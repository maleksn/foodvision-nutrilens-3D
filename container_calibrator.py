import cv2
import numpy as np

# Standard physical container dimensions in centimeters (cm) with cavity depths
STANDARD_CONTAINERS = {
    "salad_plate": {
        "type": "circular",
        "diameter_cm": 20.0,
        "cavity_depth_cm": 1.2,
        "name_ar": "صحن وسط / مقعر (20 سم)",
        "name_en": "Salad / Side Plate (20cm)"
    },
    "dinner_plate": {
        "type": "circular",
        "diameter_cm": 25.0,
        "cavity_depth_cm": 0.6,
        "name_ar": "صحن عشاء كبير (25 سم)",
        "name_en": "Dinner Plate (25cm)"
    },
    "round_bowl": {
        "type": "circular",
        "diameter_cm": 15.0,
        "cavity_depth_cm": 3.5,
        "name_ar": "زبدية شوربة / حبوب (15 سم)",
        "name_en": "Deep Bowl (15cm)"
    },
    "small_glass_bowl": {
        "type": "circular",
        "diameter_cm": 12.0,
        "cavity_depth_cm": 2.2,
        "name_ar": "كاسة / صحن صغير (12 سم)",
        "name_en": "Small Bowl (12cm)"
    },
    "flat_tray": {
        "type": "rectangular",
        "length_cm": 25.0,
        "width_cm": 18.0,
        "cavity_depth_cm": 0.0,
        "name_ar": "صينية / لوح مسطح (0 سم تجويف)",
        "name_en": "Flat Board / Tray (0cm cavity)"
    },
    "default_plate": {
        "type": "circular",
        "diameter_cm": 20.0,
        "cavity_depth_cm": 1.2,
        "name_ar": "صحن قياسي (20 سم)",
        "name_en": "Standard Plate (20cm)"
    }
}

def get_container_options(selected_type="salad_plate"):
    """Returns interactive container chip options for React Native Expo UI."""
    options = []
    for key, info in STANDARD_CONTAINERS.items():
        if key == "default_plate":
            continue
        options.append({
            "id": key,
            "name_ar": info["name_ar"],
            "name_en": info["name_en"],
            "diameter_cm": info.get("diameter_cm", info.get("length_cm")),
            "cavity_depth_cm": info["cavity_depth_cm"],
            "is_selected": (key == selected_type)
        })
    return options

def detect_container_boxes(boxes, labels, scores, img_shape):
    """
    Separates container/plate boxes from food item boxes.
    Applies background filtering and area guard to reject tables, counters, and full-image boxes.
    """
    container_keywords = ["plate", "bowl", "container", "dish", "tray", "box", "tupperware"]
    background_keywords = ["table", "counter", "desk", "background", "cloth", "floor", "napkin", "fork", "knife", "spoon", "cutting board"]
    
    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w
    
    food_boxes, food_labels, food_scores = [], [], []
    container_boxes, container_labels, container_scores = [], [], []

    for box, label, score in zip(boxes, labels, scores):
        lbl = label.lower().strip()
        
        # 1. Ignore empty or background labels (e.g. wooden table, counter, etc.)
        if not lbl or any(bg in lbl for bg in background_keywords):
            continue
            
        # 2. Area guard: food items cannot cover more than 55% of the total camera frame
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area > 0.55 * img_area:
            continue
            
        # 3. Classify container vs food
        is_container = any(kw in lbl for kw in container_keywords)
        if is_container:
            container_boxes.append(box)
            container_labels.append(label)
            container_scores.append(score)
        else:
            food_boxes.append(box)
            food_labels.append(label)
            food_scores.append(score)

    # 4. Apply Non-Maximum Suppression (NMS) on food boxes to merge overlapping duplicate predictions
    if len(food_boxes) > 1:
        indices = cv2.dnn.NMSBoxes(
            [b.tolist() for b in food_boxes],
            [float(s) for s in food_scores],
            score_threshold=0.20,
            nms_threshold=0.35
        )
        if len(indices) > 0:
            indices = indices.flatten()
            food_boxes = [food_boxes[i] for i in indices]
            food_labels = [food_labels[i] for i in indices]
            food_scores = [food_scores[i] for i in indices]

    # Sort containers by bounding box area descending so primary container is at index 0
    if len(container_boxes) > 1:
        c_indices = cv2.dnn.NMSBoxes(
            [b.tolist() for b in container_boxes],
            [float(s) for s in container_scores],
            score_threshold=0.20,
            nms_threshold=0.35
        )
        if len(c_indices) > 0:
            c_indices = c_indices.flatten()
            container_boxes = [container_boxes[i] for i in c_indices]
            container_labels = [container_labels[i] for i in c_indices]
            container_scores = [container_scores[i] for i in c_indices]

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in container_boxes]
        sorted_indices = np.argsort(areas)[::-1]
        container_boxes = [container_boxes[i] for i in sorted_indices]
        container_labels = [container_labels[i] for i in sorted_indices]
        container_scores = [container_scores[i] for i in sorted_indices]

    return (
        np.array(food_boxes), food_labels, np.array(food_scores),
        np.array(container_boxes), container_labels, np.array(container_scores)
    )


def estimate_plate_contour_fallback(img_bgr, food_masks=None, food_boxes=None):
    """
    Robust plate detector that first searches for circular plates via HoughCircles
    enclosing the food items, and falls back to contour analysis with aspect ratio guards.
    Rejects full-frame/table contours.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 1. Determine food centroid if food boxes or masks provided
    fcx, fcy = w / 2.0, h / 2.0
    if food_boxes is not None and len(food_boxes) > 0:
        fcx = float((food_boxes[0][0] + food_boxes[0][2]) / 2.0)
        fcy = float((food_boxes[0][1] + food_boxes[0][3]) / 2.0)
    elif food_masks is not None and len(food_masks) > 0:
        all_food = np.zeros((h, w), dtype=bool)
        for fm in food_masks:
            all_food |= fm.astype(bool)
        if np.any(all_food):
            ys, xs = np.where(all_food)
            fcx, fcy = float(np.mean(xs)), float(np.mean(ys))

    # 2. Primary: Circular Hough Transform for round plates/bowls
    min_r = int(min(h, w) * 0.22)
    max_r = int(max(h, w) * 0.48)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r,
        param1=50, param2=50, minRadius=min_r, maxRadius=max_r
    )
    if circles is not None and len(circles) > 0:
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            dist_food = np.hypot(cx - fcx, cy - fcy)
            # The plate must reasonably enclose the food (food center within 45% of plate radius)
            if dist_food < (r * 0.45):
                x1 = max(0.0, float(cx - r))
                y1 = max(0.0, float(cy - r))
                x2 = min(float(w), float(cx + r))
                y2 = min(float(h), float(cy + r))
                return np.array([x1, y1, x2, y2], dtype=np.float32)

        # Fallback to top-voted circle if food center check was too strict
        c = circles[0][0]
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        x1 = max(0.0, float(cx - r))
        y1 = max(0.0, float(cy - r))
        x2 = min(float(w), float(cx + r))
        y2 = min(float(h), float(cy + r))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    # 3. Secondary: Canny Edge + Morphological Closure for non-circular containers
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (0.15 * img_area) < area < (0.70 * img_area):
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = min(bw, bh) / max(bw, bh)
            if aspect > 0.60 and area > max_area:
                max_area = area
                best_cnt = cnt

    if best_cnt is not None:
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        return np.array([x, y, x + bw, y + bh], dtype=np.float32)
    return None

def compute_container_scale(container_box, points3d, container_type="dinner_plate"):
    """
    Calculates the 3D physical scale factor s by comparing the estimated 3D
    container diameter with its known physical diameter (e.g. 25.0 cm).
    s = D_known / D_estimated
    """
    x1, y1, x2, y2 = container_box.astype(int)
    h_max, w_max = points3d.shape[:2]
    x1 = max(0, min(x1, w_max - 1))
    x2 = max(0, min(x2, w_max))
    y1 = max(0, min(y1, h_max - 1))
    y2 = max(0, min(y2, h_max))

    # Points inside container bounding box
    container_pts = points3d[y1:y2, x1:x2]
    valid_mask = ~np.isnan(container_pts).any(axis=-1) & (container_pts[:, :, 2] > 0.05)
    
    if np.sum(valid_mask) < 100:
        return 1.0, {"status": "insufficient_points"}

    pts_valid = container_pts[valid_mask]
    
    # Measure 3D spans in centimeters (points3d is in meters)
    x_span_cm = (np.percentile(pts_valid[:, 0], 98) - np.percentile(pts_valid[:, 0], 2)) * 100.0
    y_span_cm = (np.percentile(pts_valid[:, 1], 98) - np.percentile(pts_valid[:, 1], 2)) * 100.0
    estimated_diameter_cm = max(x_span_cm, y_span_cm)

    target_info = STANDARD_CONTAINERS.get(container_type, STANDARD_CONTAINERS["salad_plate"])
    known_diameter_cm = target_info.get("diameter_cm", 20.0)
    cavity_depth_cm = target_info.get("cavity_depth_cm", 1.2)

    if estimated_diameter_cm <= 1.0:
        return 1.0, {"status": "invalid_span", "estimated_cm": estimated_diameter_cm, "cavity_depth_cm": cavity_depth_cm}

    scale_factor = known_diameter_cm / estimated_diameter_cm
    
    # Clip scale factor to reasonable physical boundaries (0.1x to 2.0x)
    scale_factor = float(np.clip(scale_factor, 0.15, 1.5))

    return scale_factor, {
        "status": "success",
        "container_type": container_type,
        "known_diameter_cm": known_diameter_cm,
        "cavity_depth_cm": cavity_depth_cm,
        "estimated_diameter_cm": round(estimated_diameter_cm, 1),
        "scale_factor": round(scale_factor, 4)
    }

def apply_scale_calibration(depth_map, points3d, scale_factor):
    """
    Applies the scale calibration factor s to the depth map and 3D point cloud.
    Z_calibrated = Z * s
    X_calibrated = X * s
    Y_calibrated = Y * s
    """
    calibrated_depth = depth_map * scale_factor
    calibrated_points3d = points3d * scale_factor
    return calibrated_depth, calibrated_points3d
