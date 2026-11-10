import numpy as np
import cv2
from usda_client import calculate_macros, get_food_density

# Inter-granular packing void-fraction table Phi (fraction of macroscopic air space between pieces in bulk pile).
# Combined with intrinsic cooked food density rho_intrinsic from FOOD_DENSITY_TABLE:
# rho_bulk = (1.0 - Phi) * rho_intrinsic
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
    Uses an inscribed circular/elliptical radial mask to reject background table corners
    outside the plate boundary, followed by RANSAC outlier rejection and SVD refinement.
    """
    if container_box is None:
        return None
    x1, y1, x2, y2 = container_box.astype(int)
    h_max, w_max = points3d.shape[:2]
    x1 = max(0, min(x1, w_max - 1))
    x2 = max(0, min(x2, w_max))
    y1 = max(0, min(y1, h_max - 1))
    y2 = max(0, min(y2, h_max))

    # Inscribed circular/elliptical radial mask inside [x1, y1, x2, y2]
    # Excludes the ~21.5% background table corners (1 - pi/4) outside circular tableware
    cx_box = 0.5 * (x1 + x2)
    cy_box = 0.5 * (y1 + y2)
    rx_box = max(1.0, 0.5 * (x2 - x1))
    ry_box = max(1.0, 0.5 * (y2 - y1))

    uu, vv = np.meshgrid(np.arange(w_max), np.arange(h_max))
    norm_rad_sq = ((uu - cx_box) / rx_box) ** 2 + ((vv - cy_box) / ry_box) ** 2
    container_mask = (norm_rad_sq <= 1.0)

    # Exclude food regions from plane fitting so we only fit the container rim/surface
    if food_masks is not None and len(food_masks) > 0:
        for fmask in food_masks:
            container_mask &= (~fmask.astype(bool))

    surface_pts = points3d[container_mask]
    valid = ~np.isnan(surface_pts).any(axis=-1) & (surface_pts[:, 2] > 0.05)
    pts_valid = surface_pts[valid]

    if pts_valid.shape[0] < 50:
        return None

    # Deterministic subsample for fast RANSAC + SVD
    rng = np.random.default_rng(42)
    if pts_valid.shape[0] > 5000:
        idx = rng.choice(pts_valid.shape[0], 5000, replace=False)
        pts_valid = pts_valid[idx]

    # Lightweight RANSAC loop to reject residual rim/background outliers before SVD refinement
    best_inliers = None
    best_count = 0
    n_pts = pts_valid.shape[0]
    inlier_thresh = 0.015  # 1.5 cm plane tolerance in meters

    for _ in range(100):
        sample_idx = rng.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts_valid[sample_idx]
        n_cand = np.cross(p2 - p1, p3 - p1)
        norm_val = np.linalg.norm(n_cand)
        if norm_val < 1e-9:
            continue
        n_cand = n_cand / norm_val
        d_cand = -np.dot(n_cand, p1)
        dists = np.abs(pts_valid @ n_cand + d_cand)
        inliers = dists < inlier_thresh
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    pts_fit = pts_valid[best_inliers] if (best_inliers is not None and best_count >= 30) else pts_valid

    # SVD Least-Squares Plane refinement on RANSAC inliers
    mean_pt = np.mean(pts_fit, axis=0)
    centered = pts_fit - mean_pt
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[2, :]
    if n[2] < 0:
        n = -n
    d = -np.dot(n, mean_pt)
    return n, d


def calculate_volumetric_weight(mask_bool, depth_map_calibrated, fx, fy, cx, cy, label, base_plane=None, cavity_depth_cm=0.0):
    """
    Calculates the 3D volume (cm^3) and true physical weight (g) of a segmented food item,
    accounting for perspective frustum integration, surface tilt correction, superquadric
    concave dish cavity volume, and inter-granular material porosity.
    """
    if not np.any(mask_bool):
        return None

    food_indices = np.where(mask_bool)
    v_food = food_indices[0]
    u_food = food_indices[1]
    
    # Calibrated food depth in centimeters
    Z_food_cm = depth_map_calibrated[food_indices] * 100.0
    cos_tilt = 1.0

    # Determine reference base height (cm) and integrate perspective frustum columns
    if base_plane is not None:
        n, d_plane = base_plane
        cos_tilt = float(np.clip(np.abs(n[2]), 0.75, 1.0))
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
            Z_top_cm = np.minimum(Z_food_cm[valid_ray], Z_plane_cm)
            heights = np.maximum(Z_plane_cm - Z_top_cm, 0.0)
            # Exact perspective frustum cross-sectional area: (Z_plane^2 + Z_plane*Z_top + Z_top^2) / (3 * fx * fy)
            frustum_areas_cm2 = (Z_plane_cm ** 2 + Z_plane_cm * Z_top_cm + Z_top_cm ** 2) / (3.0 * fx * fy)
            pixel_areas_cm2 = (Z_plane_cm ** 2) / (fx * fy * cos_tilt)
            bulk_volume = float(np.sum(frustum_areas_cm2 * heights))
            
            # If food item is outside or below tilted single container plane, fallback to percentile
            if bulk_volume < 1.0:
                z_ref = np.percentile(Z_food_cm, 95)
                Z_top_cm = np.minimum(Z_food_cm, z_ref)
                heights = np.maximum(0.0, z_ref - Z_top_cm)
                frustum_areas_cm2 = (z_ref ** 2 + z_ref * Z_top_cm + Z_top_cm ** 2) / (3.0 * fx * fy)
                pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
                bulk_volume = float(np.sum(frustum_areas_cm2 * heights))
        else:
            # Fallback to percentile reference
            z_ref = np.percentile(Z_food_cm, 95)
            Z_top_cm = np.minimum(Z_food_cm, z_ref)
            heights = np.maximum(0.0, z_ref - Z_top_cm)
            frustum_areas_cm2 = (z_ref ** 2 + z_ref * Z_top_cm + Z_top_cm ** 2) / (3.0 * fx * fy)
            pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
            bulk_volume = float(np.sum(frustum_areas_cm2 * heights))
    else:
        # Robust percentile baseline
        z_ref = np.percentile(Z_food_cm, 95)
        Z_top_cm = np.minimum(Z_food_cm, z_ref)
        heights = np.maximum(0.0, z_ref - Z_top_cm)
        frustum_areas_cm2 = (z_ref ** 2 + z_ref * Z_top_cm + Z_top_cm ** 2) / (3.0 * fx * fy)
        pixel_areas_cm2 = (Z_food_cm ** 2) / (fx * fy)
        bulk_volume = float(np.sum(frustum_areas_cm2 * heights))

    # Add concave container cavity volume underneath base plane
    contact_area_cm2 = float(np.sum(pixel_areas_cm2))
    cavity_volume_cm3 = 0.0
    if cavity_depth_cm > 0.0:
        # Superquadric flat-bottomed ceramic dish profile: z(r) = H_cavity * (r / R)^n
        # Volume ratio alpha = n / (n + 2) = 0.70 for flat-floored ceramic plates (n = 4.67),
        # compared to alpha = 0.50 (n = 2) for a sharp-apex paraboloid.
        # Scale effective cavity depth proportionally when contact_area_cm2 is smaller than reference dish well base (145 cm^2).
        dish_base_ref_cm2 = 145.0
        coverage_scale = float(np.clip(contact_area_cm2 / dish_base_ref_cm2, 0.0, 1.0))
        effective_cavity_depth_cm = float(cavity_depth_cm) * coverage_scale
        cavity_volume_cm3 = contact_area_cm2 * (effective_cavity_depth_cm * 0.70)
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
