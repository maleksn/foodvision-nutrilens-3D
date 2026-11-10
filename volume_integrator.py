import numpy as np
import cv2
from usda_client import calculate_macros, get_food_density

# Scientific porosity / void-fraction table (percentage of air space in bulk volume)
FOOD_POROSITY_TABLE = {
    "french fries": 0.40,
    "fries": 0.40,
    "broccoli": 0.45,
    "steamed broccoli": 0.45,
    "chicken": 0.12,
    "cooked chicken": 0.12,
    "shredded cooked chicken": 0.15,
    "grilled chicken": 0.10,
    "rice": 0.18,
    "cooked rice": 0.18,
    "green beans": 0.35,
    "chickpeas": 0.25,
    "hummus": 0.00,
    "sauce": 0.00,
    "salad": 0.60,
    "popcorn": 0.70,
    "bread": 0.30,
    "default": 0.15
}

def get_food_porosity(label: str) -> float:
    name_lower = label.lower().strip()
    if name_lower in FOOD_POROSITY_TABLE:
        return FOOD_POROSITY_TABLE[name_lower]
    for k, v in FOOD_POROSITY_TABLE.items():
        if k in name_lower or name_lower in k:
            return v
    return FOOD_POROSITY_TABLE["default"]

def fit_global_container_plane(container_box, points3d, food_masks=None):
    """
    Fits a robust base plane across the container rim/surface excluding food masks.
    """
    if container_box is None:
        return None
    x1, y1, x2, y2 = container_box.astype(int)
    h_max, w_max = points3d.shape[:2]
    x1 = max(0, min(x1, w_max - 1))
    x2 = max(0, min(x2, w_max))
    y1 = max(0, min(y1, h_max - 1))
    y2 = max(0, min(y2, h_max))

    container_mask = np.zeros((h_max, w_max), dtype=bool)
    container_mask[y1:y2, x1:x2] = True

    # Exclude food regions from plane fitting so we only fit the container/table surface
    if food_masks is not None and len(food_masks) > 0:
        for fmask in food_masks:
            container_mask &= (~fmask.astype(bool))

    surface_pts = points3d[container_mask]
    valid = ~np.isnan(surface_pts).any(axis=-1) & (surface_pts[:, 2] > 0.05)
    pts_valid = surface_pts[valid]

    if pts_valid.shape[0] < 50:
        return None

    # Subsample if too many points for fast RANSAC
    if pts_valid.shape[0] > 5000:
        idx = np.random.choice(pts_valid.shape[0], 5000, replace=False)
        pts_valid = pts_valid[idx]

    # SVD Plane fit
    centered = pts_valid - np.mean(pts_valid, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[2, :]
    if n[2] < 0:
        n = -n
    d = -np.dot(n, np.mean(pts_valid, axis=0))
    return n, d

def calculate_volumetric_weight(mask_bool, depth_map_calibrated, fx, fy, cx, cy, label, base_plane=None, cavity_depth_cm=0.0):
    """
    Calculates the 3D volume (cm^3) and true physical weight (g) of a segmented food item,
    accounting for metric calibration, material porosity, and concave dish cavity volume.
    """
    if not np.any(mask_bool):
        return None

    food_indices = np.where(mask_bool)
    v_food = food_indices[0]
    u_food = food_indices[1]
    
    # Calibrated food depth in centimeters
    Z_food_cm = depth_map_calibrated[food_indices] * 100.0

    # Determine reference base height (cm)
    if base_plane is not None:
        n, d_plane = base_plane
        ray_dir = np.stack([
            (u_food - cx) / fx,
            (v_food - cy) / fy,
            np.ones_like(u_food)
        ], axis=1)
        denom = np.dot(ray_dir, n)
        valid_ray = denom > 0.01
        if np.sum(valid_ray) > (0.5 * len(u_food)):
            t = -d_plane / denom[valid_ray]
            Z_plane_cm = (t * ray_dir[valid_ray, 2]) * 100.0
            heights = Z_plane_cm - Z_food_cm[valid_ray]
            heights = np.maximum(heights, 0.0)
            pixel_areas_cm2 = (Z_plane_cm ** 2) / (fx * fy)
            bulk_volume = float(np.sum(pixel_areas_cm2 * heights))
            
            # If food item is outside or below tilted single container plane, fallback to percentile
            if bulk_volume < 1.0:
                z_ref = np.percentile(Z_food_cm, 95)
                heights = np.maximum(0.0, z_ref - Z_food_cm)
                pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
                bulk_volume = float(np.sum(pixel_areas_cm2 * heights))
        else:
            # Fallback to percentile reference
            z_ref = np.percentile(Z_food_cm, 95)
            heights = np.maximum(0.0, z_ref - Z_food_cm)
            pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
            bulk_volume = float(np.sum(pixel_areas_cm2 * heights))
    else:
        # Robust percentile baseline
        z_ref = np.percentile(Z_food_cm, 95)
        heights = np.maximum(0.0, z_ref - Z_food_cm)
        pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
        bulk_volume = float(np.sum(pixel_areas_cm2 * heights))

    # Add concave container cavity volume underneath base plane
    contact_area_cm2 = float(np.sum(pixel_areas_cm2))
    cavity_volume_cm3 = 0.0
    if cavity_depth_cm > 0.0:
        # Parabolic dish cavity model: average submerged depth is ~0.70 of max cavity depth
        cavity_volume_cm3 = contact_area_cm2 * (float(cavity_depth_cm) * 0.70)
        bulk_volume += cavity_volume_cm3

    # Apply physical porosity / void-fraction correction
    porosity = get_food_porosity(label)
    net_volume = bulk_volume * (1.0 - porosity)
    
    # Calculate physical mass using real food density
    density = get_food_density(label)
    weight_g = net_volume * density

    # Query USDA FoodData Central for complete macronutrients
    macros = calculate_macros(label, weight_g)

    return {
        "label": label,
        "pixel_count": int(np.sum(mask_bool)),
        "mean_depth_cm": float(np.mean(Z_food_cm)),
        "bulk_volume_cm3": round(bulk_volume, 1),
        "porosity": porosity,
        "net_volume_cm3": round(net_volume, 1),
        "density_g_cm3": density,
        "weight_g": round(weight_g, 1),
        "macros": macros
    }
