import os
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["PYTORCH_JIT_DISABLE"] = "1"
import cv2
import numpy as np
import torch
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from PIL.ExifTags import TAGS
import io
import time
import base64
from collections import Counter

from usda_client import calculate_macros, get_food_density
from container_calibrator import (
    detect_container_boxes,
    estimate_plate_contour_fallback,
    compute_container_scale,
    apply_scale_calibration,
    get_container_options
)
from volume_integrator import (
    fit_global_container_plane,
    calculate_volumetric_weight,
    get_food_porosity
)

app = FastAPI(
    title="NutriLens AI - 3D Food Weight & Macro Detection Pipeline",
    description="Local 3D computer vision API for estimating food weight and macronutrients from single photos.",
    version="1.0.0"
)

# Enable CORS for React Native Expo mobile frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/mnt/d/NutriLens_Backend/sam2.1_hiera_large.pt" if os.path.exists("/mnt/d/NutriLens_Backend/sam2.1_hiera_large.pt")
    else ("sam2.1_hiera_large.pt" if os.path.exists("sam2.1_hiera_large.pt") else "./checkpoints/sam2.1_hiera_large.pt")
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

BOX_THRESHOLD = 0.25

# Global models cache dictionary
models = {}

def get_models():
    """Lazy model loader ensuring models are initialized once and cached."""
    if "dino" not in models:
        print("Loading Grounding DINO on CPU...")
        models["dino_proc"] = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        models["dino"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cpu")
    if "sam2" not in models:
        if not os.path.exists(SAM2_CHECKPOINT):
            raise FileNotFoundError(
                f"SAM 2.1 checkpoint not found at '{SAM2_CHECKPOINT}'. "
                "Download sam2.1_hiera_large.pt into ./checkpoints/ or set SAM2_CHECKPOINT."
            )
        print(f"Loading SAM 2.1 from {SAM2_CHECKPOINT} on {DEVICE}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE, apply_postprocessing=False)
        models["sam2"] = SAM2ImagePredictor(sam2_model)
    if "metric3d" not in models:
        print(f"Loading Metric3D ViT-Large on {DEVICE}...")
        depth_model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True)
        models["metric3d"] = depth_model.to(DEVICE).eval()
    return models

def get_camera_intrinsics(image_pil, resized_width, focal_override=None):
    """Extract 35mm equivalent focal length from EXIF or fallback to standard phone lens (23mm)."""
    if focal_override and focal_override > 0:
        focal_length_35mm = float(focal_override)
    else:
        exif_data = image_pil._getexif() if hasattr(image_pil, '_getexif') else None
        focal_length_35mm = None
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'FocalLengthIn35mmFilm':
                    focal_length_35mm = float(value)
                    break
        if focal_length_35mm is None:
            focal_length_35mm = 23.0  # Standard modern smartphone wide camera equivalent
    f_pixel = (focal_length_35mm / 36.0) * resized_width
    return f_pixel, f_pixel

def backproject_pixels(depth_map, fx, fy, cx, cy):
    """Backprojects depth map pixels into camera frame 3D point cloud [X, Y, Z]."""
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)

@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status and GPU availability."""
    return {
        "status": "online",
        "device": DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models_loaded": list(models.keys()),
        "sam2_checkpoint_found": os.path.exists(SAM2_CHECKPOINT)
    }

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    ingredients: Optional[str] = Form(None),
    container_type: Optional[str] = Form("dinner_plate"),
    focal_length_35mm: Optional[float] = Form(None)
):
    """
    Analyzes an uploaded food photo using NutriLens 3D AI pipeline.
    Estimates 3D depth, performs container-based scale calibration,
    computes porosity-adjusted net volume and weight, queries USDA FoodData Central,
    and returns nutrition breakdown plus visual annotated overlay.
    """
    start_time = time.time()
    
    # 1. Read and parse image
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid image file: {e}"})
    
    W_orig, H_orig = image_pil.size
    image_rgb = np.array(image_pil)
    
    # 2. Formulate dynamic DINO prompt
    candidate_items = []
    if ingredients:
        try:
            parsed = json.loads(ingredients)
            if isinstance(parsed, list):
                candidate_items = [str(x).strip() for x in parsed if str(x).strip()]
            elif isinstance(parsed, str):
                candidate_items = [str(x).strip() for x in parsed.split(",") if str(x).strip()]
        except Exception:
            candidate_items = [str(x).strip() for x in ingredients.split(",") if str(x).strip()]
            
    if not candidate_items and os.path.exists("recipe.json"):
        try:
            with open("recipe.json", "r", encoding="utf-8") as f:
                r_data = json.load(f)
            if isinstance(r_data, dict) and "ingredients" in r_data:
                candidate_items = [str(x).strip() for x in r_data["ingredients"] if str(x).strip()]
            elif isinstance(r_data, list):
                candidate_items = [str(x).strip() for x in r_data if str(x).strip()]
        except Exception:
            pass

    if candidate_items:
        dino_prompt = " . ".join(candidate_items) + " . plate . bowl . container ."
    else:
        dino_prompt = "chicken . meat . rice . broccoli . french fries . pasta . salad . potatoes . plate . bowl . container ."

    # 3. Load models
    m = get_models()
    
    # 4. Run Grounding DINO detection (on CPU)
    inputs = m["dino_proc"](images=image_pil, text=dino_prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        dino_outputs = m["dino"](**inputs)
    
    dino_results = m["dino_proc"].post_process_grounded_object_detection(
        dino_outputs, inputs.input_ids, threshold=BOX_THRESHOLD, target_sizes=[(H_orig, W_orig)]
    )[0]
    
    raw_boxes = dino_results["boxes"].cpu().numpy()
    raw_scores = dino_results["scores"].cpu().numpy()
    raw_labels = dino_results["text_labels"]
    
    food_boxes, food_labels, food_scores, container_boxes, container_labels, container_scores = detect_container_boxes(
        raw_boxes, raw_labels, raw_scores, (H_orig, W_orig)
    )
    
    # Fallback container detection if DINO missed it or detected oversized box (table/room)
    img_area = H_orig * W_orig
    valid_cboxes, valid_clabels = [], []
    for cb, cl in zip(container_boxes, container_labels):
        cb_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
        if cb_area < 0.70 * img_area:
            valid_cboxes.append(cb)
            valid_clabels.append(cl)

    if len(valid_cboxes) == 0:
        fallback_cbox = estimate_plate_contour_fallback(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), food_boxes=food_boxes)
        if fallback_cbox is not None:
            container_boxes = np.array([fallback_cbox])
            container_labels = ["plate"]
        else:
            container_boxes = np.array([])
            container_labels = []
    else:
        container_boxes = np.array(valid_cboxes)
        container_labels = valid_clabels
    
    if len(food_boxes) == 0:
        return {
            "status": "no_food_detected",
            "message": "No food items detected in the image.",
            "prompt_used": dino_prompt,
            "container_found": len(container_boxes) > 0
        }
        
    # 5. Run SAM 2.1 Full-Box Segmentation
    m["sam2"].set_image(image_rgb)
    masks, _, _ = m["sam2"].predict(box=food_boxes, multimask_output=False)
    if masks.ndim == 4:
        food_masks = (masks[:, 0, :, :] > 0.0)
    elif masks.ndim == 3:
        food_masks = (masks > 0.0)
    else:
        food_masks = np.array([masks > 0.0])
    
    # 6. Run Metric3D ViT-Large Metric Depth Estimation
    fx, fy = get_camera_intrinsics(image_pil, W_orig, focal_override=focal_length_35mm)
    cx = W_orig / 2.0
    cy = H_orig / 2.0
    intrinsic = [fx, fy, cx, cy]
    
    input_size = (616, 1064)
    scale = min(input_size[0] / H_orig, input_size[1] / W_orig)
    rgb_resized = cv2.resize(image_rgb, (int(W_orig * scale), int(H_orig * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic_scaled = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    
    padding = [123.675, 116.28, 103.53]
    h_s, w_s = rgb_resized.shape[:2]
    pad_h = input_size[0] - h_s
    pad_w = input_size[1] - w_s
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb_padded = cv2.copyMakeBorder(rgb_resized, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_tensor = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float()
    rgb_tensor = torch.div((rgb_tensor - mean), std)[None, :, :, :].to(DEVICE)
    
    with torch.no_grad():
        pred_depth, _, _ = m["metric3d"].inference({'input': rgb_tensor})
        
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (H_orig, W_orig), mode='bilinear').squeeze()
    
    # Official Metric3D canonical de-normalization
    canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
    depth_map_metres = (pred_depth * canonical_to_real_scale).cpu().numpy()
    
    # 7. Container-Based Scale Auto-Calibration
    points3d = backproject_pixels(depth_map_metres, fx, fy, cx, cy)
    scale_factor = 1.0
    calib_meta = {
        "container_detected": False,
        "container_type": container_type,
        "scale_factor": 1.0,
        "camera_distance_cm": round(float(depth_map_metres.mean() * 100.0), 1)
    }
    
    if len(container_boxes) > 0:
        cbox = container_boxes[0]
        scale_factor, calib_info = compute_container_scale(cbox, points3d, container_type)
        depth_map_calib, points3d = apply_scale_calibration(depth_map_metres, points3d, scale_factor)
        calib_meta["container_detected"] = True
        calib_meta["scale_factor"] = round(float(scale_factor), 4)
        calib_meta["estimated_container_cm"] = calib_info.get("estimated_diameter_cm")
        calib_meta["known_container_cm"] = calib_info.get("known_diameter_cm")
        calib_meta["cavity_depth_cm"] = calib_info.get("cavity_depth_cm", 1.2)
        calib_meta["camera_distance_cm"] = round(float(depth_map_calib.mean() * 100.0), 1)
    else:
        depth_map_calib = depth_map_metres

    # 8. Fit global container base plane
    cbox = container_boxes[0] if len(container_boxes) > 0 else None
    container_base_plane = fit_global_container_plane(cbox, points3d, food_masks)
    
    # 9. Volume, Porosity & USDA Nutrition Calculation
    items_result = []
    total_weight = 0.0
    total_cal = 0.0
    total_p = 0.0
    total_c = 0.0
    total_f = 0.0
    
    for i, mask in enumerate(food_masks):
        lbl = food_labels[i]
        vol_res = calculate_volumetric_weight(
            mask_bool=mask,
            depth_map_calibrated=depth_map_calib,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            label=lbl,
            base_plane=container_base_plane,
            cavity_depth_cm=calib_meta.get("cavity_depth_cm", 0.0)
        )
        if vol_res is not None:
            m_nutr = vol_res["macros"]
            box_coords = [int(v) for v in food_boxes[i]]
            item_entry = {
                "label": lbl,
                "weight_g": round(vol_res["weight_g"], 1),
                "bulk_volume_cm3": round(vol_res["bulk_volume_cm3"], 1),
                "net_volume_cm3": round(vol_res["net_volume_cm3"], 1),
                "density_g_cm3": round(vol_res["density_g_cm3"], 2),
                "porosity": vol_res["porosity"],
                "box": box_coords,
                "macros": {
                    "calories": round(m_nutr["calories"], 1),
                    "protein_g": round(m_nutr["protein_g"], 1),
                    "carbs_g": round(m_nutr["carbs_g"], 1),
                    "fat_g": round(m_nutr["fat_g"], 1),
                    "source": m_nutr.get("source", "USDA FoodData Central"),
                    "matched_desc": m_nutr.get("matched_desc", lbl)
                }
            }
            items_result.append(item_entry)
            total_weight += vol_res["weight_g"]
            total_cal += m_nutr["calories"]
            total_p += m_nutr["protein_g"]
            total_c += m_nutr["carbs_g"]
            total_f += m_nutr["fat_g"]
            
    # 10. Generate Visual Annotation Overlay
    annotated_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    palette = [
        (0, 165, 255),   # Orange
        (50, 205, 50),   # Lime Green
        (238, 130, 238), # Violet
        (0, 215, 255),   # Gold
        (255, 144, 30),  # Dodger Blue
    ]
    
    for idx, item in enumerate(items_result):
        mask = food_masks[idx]
        col = palette[idx % len(palette)]
        
        # Semi-transparent mask overlay
        annotated_bgr[mask] = (annotated_bgr[mask] * 0.55 + np.array(col) * 0.45).astype(np.uint8)
        
        # Box & Badge
        bx1, by1, bx2, by2 = item["box"]
        cv2.rectangle(annotated_bgr, (bx1, by1), (bx2, by2), col, 2)
        
        badge_text = f"{item['label'].title()}: {item['weight_g']:.0f}g | {item['macros']['calories']:.0f} kcal"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated_bgr, (bx1, max(0, by1 - 24)), (bx1 + tw + 10, by1), col, -1)
        cv2.putText(annotated_bgr, badge_text, (bx1 + 5, max(16, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    
    if len(container_boxes) > 0:
        cbx1, cby1, cbx2, cby2 = container_boxes[0].astype(int)
        cv2.rectangle(annotated_bgr, (cbx1, cby1), (cbx2, cby2), (255, 255, 255), 1, cv2.LINE_AA)
        ref_text = f"Container Ref: {calib_meta.get('known_container_cm', 25.0)}cm"
        cv2.putText(annotated_bgr, ref_text, (cbx1 + 5, cby2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
    # Save output image
    out_filename = f"nutrilens_result_{int(time.time())}.jpg"
    cv2.imwrite(out_filename, annotated_bgr)
    
    # Base64 encode for direct rendering in React Native Expo Image component
    _, buffer = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    
    elapsed = round(time.time() - start_time, 2)
    
    return {
        "status": "success",
        "elapsed_seconds": elapsed,
        "total_meal": {
            "weight_g": round(total_weight, 1),
            "calories": round(total_cal, 1),
            "protein_g": round(total_p, 1),
            "carbs_g": round(total_c, 1),
            "fat_g": round(total_f, 1)
        },
        "items": items_result,
        "calibration": calib_meta,
        "container_options": get_container_options(selected_type=container_type),
        "annotated_image_path": out_filename,
        "annotated_image_base64": annotated_b64
    }