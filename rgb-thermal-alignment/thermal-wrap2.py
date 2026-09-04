import os
import struct
import numpy as np
import cv2
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = "/home/aayushi/exodia/depth_test/experiment_007/images"          # same DATASET_ROOT as the COLMAP notebook
SPARSE_MODEL_DIR = "/home/aayushi/exodia/colmap_output2/sparse/0"   # exported TEXT model (cameras.txt, images.txt) -- must be the rig-corrected (v3/v4) reconstruction
FUSED_PLY_PATH = "/home/aayushi/exodia/colmap_output2/dense/fused.ply"          # must be rebuilt AFTER the rig fix
OUTPUT_DIR = "thermal_wraparound_outputs2"

# --- RGB intrinsics (same values fed into COLMAP's feature_extractor) ---
LEFT_INTR = dict(fx=595.607, fy=594.714, cx=941.101, cy=458.698, skew=0.895965,
                  k1=-0.00257185, k2=-0.0145025)
RIGHT_INTR = dict(fx=598.517, fy=597.363, cx=970.896, cy=424.176, skew=0.866411,
                   k1=0.00464308, k2=-0.0195245)

# raw discocal camera0to1 pose (rgb1=camera0, rgb2=camera1): X_rgb1 = R@X_rgb2 + t
# used ONLY to compute the known physical baseline for the scale VERIFICATION check.
RGB_PAIR_RVEC_RAW = np.array([0.0171, -0.0259, 3.1381])
RGB_PAIR_TVEC_RAW = np.array([-0.1482, 0.0009, 0.0015]).reshape(3, 1)

# --- Thermal-left ---
IMG_SIZE_THERMAL = (256, 192)
THERMAL_LEFT_INTR = dict(fx=267.489, fy=267.459, cx=129.32, cy=84.6925, skew=0.866478,
                          k1=-0.308768, k2=-0.0475762)
# thermal_left_to_rgb_left, already a point-transform: X_thermal_left = R@X_rgb_left + t
THERMAL_LEFT_RVEC = np.array([-0.0864, 0.0069, 0.0031])
THERMAL_LEFT_TVEC = np.array([0.0548, -0.0016, 0.0004]).reshape(3, 1)

# --- Thermal-right ---
THERMAL_RIGHT_INTR = dict(fx=263.845, fy=263.421, cx=133.939, cy=96.2706, skew=0.351657,
                           k1=-0.367637, k2=0.116875)
# raw discocal camera0to1 pose (camera0=rgb_left, camera1=thermal_right)
THERMAL_RIGHT_RVEC_RAW = np.array([0.006482, 0.042439, -0.017448])
THERMAL_RIGHT_TVEC_RAW = np.array([0.095655, -0.000551, 0.000212]).reshape(3, 1)

# --- Projection / occlusion (same semantics as the original script) ---
MIN_Z_M = 0.05
OCCLUSION_TOL_M = 0.25       # z-buffer tolerance; tune per scene scale, same as before
FOV_MARGIN = 1.15

# --- Aggregation ---
AGGREGATE = "median"           # "median" or "mean"
# All points are always kept (toggle mode needs matching point sets in both RGB and
# thermal views) -- points with zero thermal coverage just render gray in thermal mode.

# --- Thermal image filenames per frame folder (same convention as before) ---
THERMAL_LEFT_NAME = "thermal1_gray.png"
THERMAL_RIGHT_NAME = "thermal2_gray.png"

# --- Scale verification tolerance ---
SCALE_VERIFY_TOL_FRAC = 0.03   # warn if reconstructed baseline is off by more than this fraction


# ============================================================
# GEOMETRY HELPERS 
# ============================================================

def invert_transform(R, t):
    """Pose (X_A = R@X_B + t) -> point-transform for the inverse direction."""
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return np.ascontiguousarray(R.T), np.ascontiguousarray(-R.T @ t)


def undo_normalize_es(R, t, angle_threshold_deg=90.0):
    """Undo discocal's normalize_Es 180deg flip if the raw rotation signals it fired."""
    rvec, _ = cv2.Rodrigues(R)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    if np.degrees(np.linalg.norm(rvec)) < angle_threshold_deg:
        return np.ascontiguousarray(R), np.ascontiguousarray(t)
    Rz_pi, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi]))
    return np.ascontiguousarray(R @ Rz_pi), np.ascontiguousarray(-t)


def build_K(i):
    return np.array([[i["fx"], i["skew"], i["cx"]],
                      [0, i["fy"], i["cy"]],
                      [0, 0, 1]])


def build_dist(i):
    return np.array([i["k1"], i["k2"], 0.0, 0.0, 0.0])


def build_pose(rx, ry, rz, tx, ty, tz):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    t = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
    return R, t


def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


# ============================================================
# COLMAP SPARSE MODEL PARSING
# ============================================================

def parse_colmap_images_txt(path):
    """Returns dict: image_name -> dict(camera_id, qvec, tvec, R, t)
    where X_cam = R @ X_world + t  (COLMAP's world-to-camera convention).
    """
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # every image occupies 2 lines: pose line, then a POINTS2D line -- we only need the pose line
    for line in lines[::2]:
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))   # qw, qx, qy, qz
        tvec = np.array(list(map(float, parts[5:8])))   # tx, ty, tz
        camera_id = int(parts[8])
        name = parts[9]
        R = qvec2rotmat(qvec)
        t = tvec.reshape(3, 1)
        images[name] = dict(image_id=image_id, camera_id=camera_id, qvec=qvec, R=R, t=t)
    return images


def frame_id_from_name(name, tag):
    """'rgb1_000003.png' -> '000003'"""
    prefix = f"{tag}_"
    if name.startswith(prefix):
        return name[len(prefix):-4]   # strip prefix and ".png"
    return name


# ============================================================
# SCALE VERIFICATION (no rescaling -- COLMAP output is already metric)
# ============================================================

def compute_known_rgb_baseline_m():
    """Physical rgb1<->rgb2 baseline in meters, from the calibrated stereo extrinsics."""
    R_raw, _ = cv2.Rodrigues(RGB_PAIR_RVEC_RAW)
    R_pose, t_pose = undo_normalize_es(R_raw, RGB_PAIR_TVEC_RAW)
    R, T = invert_transform(R_pose, t_pose)
    return float(np.linalg.norm(T)), R, T


def verify_metric_scale(images, ref_frame_id, tol_frac=SCALE_VERIFY_TOL_FRAC):
    """Sanity check only -- no rescaling. With the rig-constrained COLMAP reconstruction
    (v3/v4), the sparse model's world coordinates are ALREADY in meters, since the known
    rgb1<->rgb2 extrinsic was locked/refined during bundle adjustment rather than left to
    float. This just confirms that held, by comparing the reconstructed baseline against
    the known physical one, and warns loudly if they've drifted apart.
    """
    known_baseline_m, _, _ = compute_known_rgb_baseline_m()

    name1 = f"rgb1_{ref_frame_id}.png"
    name2 = f"rgb2_{ref_frame_id}.png"
    if name1 not in images or name2 not in images:
        print(f"WARNING: can't verify scale -- frame {ref_frame_id} doesn't have both "
              f"rgb1 and rgb2 registered. Skipping verification, assuming metric scale.")
        return

    R1, t1 = images[name1]["R"], images[name1]["t"]
    R2, t2 = images[name2]["R"], images[name2]["t"]
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    colmap_baseline = float(np.linalg.norm(C1 - C2))

    ratio = colmap_baseline / known_baseline_m
    print(f"Known physical rgb1<->rgb2 baseline: {known_baseline_m:.4f} m")
    print(f"COLMAP-reconstructed baseline (frame {ref_frame_id}): {colmap_baseline:.4f} m")
    print(f"Ratio (should be ~1.0 if the rig constraint took effect): {ratio:.4f}")
    if abs(ratio - 1.0) > tol_frac:
        print(f"WARNING: reconstructed baseline is {abs(ratio - 1) * 100:.1f}% off the "
              f"calibrated value -- check SPARSE_MODEL_DIR actually points at the "
              f"rig-corrected reconstruction, not an old unconstrained one.")


# ============================================================
# PROJECTION + Z-BUFFER (ported from project_rgb_depth_to_thermal,
# adapted for a sparse 3D point set instead of a dense per-pixel depth map)
# ============================================================

def project_points_to_thermal_zbuffer(points_world, R_w2th, t_w2th,
                                       K_thermal, D_thermal, thermal_img,
                                       min_z_m=MIN_Z_M, occlusion_tol_m=OCCLUSION_TOL_M,
                                       fov_margin=FOV_MARGIN):
    """Project points_world (N,3) into one thermal camera, z-buffer per thermal pixel
    to reject occluded points, bilinear-sample thermal intensity at surviving points.

    Returns:
        kept_indices: indices into points_world that survived (in front, in FOV, frontmost)
        thermal_values: sampled thermal intensity per kept index
    """
    pts_thermal = points_world @ R_w2th.T + t_w2th.reshape(1, 3)
    z_thermal = pts_thermal[:, 2]

    img_pts, _ = cv2.projectPoints(
        pts_thermal.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_thermal, D_thermal,
    )
    img_pts = img_pts.reshape(-1, 2)

    Ht, Wt = thermal_img.shape
    in_front = z_thermal > min_z_m
    in_bounds = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < Wt) & \
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < Ht)

    # pre-distortion FOV gate, same rationale as the original script: guards against
    # cv2.projectPoints folding an out-of-FOV ray back onto a plausible in-bounds pixel
    corners = np.array([[0, 0], [Wt - 1, 0], [0, Ht - 1], [Wt - 1, Ht - 1]], dtype=np.float64)
    corners_norm = cv2.undistortPoints(corners.reshape(-1, 1, 2), K_thermal, D_thermal).reshape(-1, 2)
    xn_max = np.abs(corners_norm[:, 0]).max() * fov_margin
    yn_max = np.abs(corners_norm[:, 1]).max() * fov_margin
    xn = pts_thermal[:, 0] / np.where(in_front, z_thermal, np.nan)
    yn = pts_thermal[:, 1] / np.where(in_front, z_thermal, np.nan)
    in_fov_angle = (np.abs(xn) <= xn_max) & (np.abs(yn) <= yn_max)

    candidate = in_front & in_bounds & in_fov_angle
    cand_idx = np.where(candidate)[0]
    if cand_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # z-buffer: bin candidates into the thermal camera's own pixel grid, keep frontmost
    ui = np.clip(np.round(img_pts[cand_idx, 0]).astype(np.int64), 0, Wt - 1)
    vi = np.clip(np.round(img_pts[cand_idx, 1]).astype(np.int64), 0, Ht - 1)
    flat_bin = vi * Wt + ui

    zbuf = np.full(Ht * Wt, np.inf, dtype=np.float64)
    np.minimum.at(zbuf, flat_bin, z_thermal[cand_idx])

    z_at_bin = zbuf[flat_bin]
    is_frontmost = z_thermal[cand_idx] <= z_at_bin + occlusion_tol_m
    kept_idx = cand_idx[is_frontmost]

    # bilinear-sample thermal intensity at the exact (sub-pixel) projected location.
    # Manual numpy bilinear instead of cv2.remap: remap's output is capped at
    # SHRT_MAX (32767) per axis, which a single (1, N) sample batch can exceed here.
    thermal_values = bilinear_sample(thermal_img, img_pts[kept_idx, 0], img_pts[kept_idx, 1])

    return kept_idx, thermal_values


def bilinear_sample(img, xs, ys):
    """Bilinear-sample img (H,W) at floating-point (xs, ys) pixel coords, clamped to bounds.
    No size limit, unlike cv2.remap -- works for arbitrarily many sample points.
    """
    img_f = img.astype(np.float32)
    H, W = img_f.shape

    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = xs - x0
    wy = ys - y0

    v00 = img_f[y0, x0]
    v01 = img_f[y0, x1]
    v10 = img_f[y1, x0]
    v11 = img_f[y1, x1]

    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bot * wy


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Load fused point cloud ----------
    pcd = o3d.io.read_point_cloud(FUSED_PLY_PATH)
    points_world_colmap = np.asarray(pcd.points)
    n_points = points_world_colmap.shape[0]
    print(f"Loaded fused point cloud: {n_points} points")

    if pcd.has_colors():
        rgb_colors = np.asarray(pcd.colors)  # (N,3), 0-1 range
    else:
        print("WARNING: fused.ply has no color -- rgb_colors will be mid-gray")
        rgb_colors = np.full((n_points, 3), 0.5)

    # ---------- Load COLMAP sparse model ----------
    images = parse_colmap_images_txt(os.path.join(SPARSE_MODEL_DIR, "images.txt"))
    rgb1_frames = sorted(
        frame_id_from_name(name, "rgb1") for name in images if name.startswith("rgb1_")
    )
    print(f"COLMAP-registered rgb1 frames: {rgb1_frames}")
    if not rgb1_frames:
        raise RuntimeError("No registered rgb1 frames found in images.txt")

    # ---------- Scale verification (no rescaling -- already metric via rig-constrained BA) ----------
    REF_FRAME_ID = rgb1_frames[0]   # first registered frame with both rgb1+rgb2, adjust if needed
    verify_metric_scale(images, REF_FRAME_ID)
    points_world = points_world_colmap   # already metric, no scale_factor applied

    # ---------- Fixed rig extrinsics: rgb1 -> thermal_left / thermal_right ----------
    K_thermal_left = build_K(THERMAL_LEFT_INTR)
    D_thermal_left = build_dist(THERMAL_LEFT_INTR)
    R_rgb1_to_thL, t_rgb1_to_thL = build_pose(
        *THERMAL_LEFT_RVEC.tolist(), *THERMAL_LEFT_TVEC.ravel().tolist()
    )

    K_thermal_right = build_K(THERMAL_RIGHT_INTR)
    D_thermal_right = build_dist(THERMAL_RIGHT_INTR)
    R_raw_tr, _ = cv2.Rodrigues(THERMAL_RIGHT_RVEC_RAW)
    R_pose_tr, t_pose_tr = undo_normalize_es(R_raw_tr, THERMAL_RIGHT_TVEC_RAW)
    R_rgb1_to_thR, t_rgb1_to_thR = invert_transform(R_pose_tr, t_pose_tr)

    # ---------- Accumulate thermal samples per point, across all frames/cameras ----------
    accum = [[] for _ in range(n_points)]

    for frame_id in rgb1_frames:
        rgb1_name = f"rgb1_{frame_id}.png"
        if rgb1_name not in images:
            print(f"  skip {frame_id}: rgb1 not registered")
            continue

        R_w2rgb1 = images[rgb1_name]["R"]
        t_w2rgb1 = images[rgb1_name]["t"]   # already metric, no scale_factor needed

        # world -> thermal_left_i  =  (rgb1 -> thermal_left) o (world -> rgb1_i)
        R_w2thL = R_rgb1_to_thL @ R_w2rgb1
        t_w2thL = R_rgb1_to_thL @ t_w2rgb1 + t_rgb1_to_thL

        R_w2thR = R_rgb1_to_thR @ R_w2rgb1
        t_w2thR = R_rgb1_to_thR @ t_w2rgb1 + t_rgb1_to_thR

        frame_dir = os.path.join(DATASET_ROOT, frame_id)
        thermal_left_path = os.path.join(frame_dir, THERMAL_LEFT_NAME)
        thermal_right_path = os.path.join(frame_dir, THERMAL_RIGHT_NAME)

        n_contrib = 0
        if os.path.exists(thermal_left_path):
            thermal_left_img = cv2.imread(thermal_left_path, cv2.IMREAD_GRAYSCALE)
            assert thermal_left_img.shape[::-1] == IMG_SIZE_THERMAL
            kept_idx, vals = project_points_to_thermal_zbuffer(
                points_world, R_w2thL, t_w2thL, K_thermal_left, D_thermal_left, thermal_left_img
            )
            for idx, v in zip(kept_idx, vals):
                accum[idx].append(v)
            n_contrib += len(kept_idx)

        if os.path.exists(thermal_right_path):
            thermal_right_img = cv2.imread(thermal_right_path, cv2.IMREAD_GRAYSCALE)
            assert thermal_right_img.shape[::-1] == IMG_SIZE_THERMAL
            kept_idx, vals = project_points_to_thermal_zbuffer(
                points_world, R_w2thR, t_w2thR, K_thermal_right, D_thermal_right, thermal_right_img
            )
            for idx, v in zip(kept_idx, vals):
                accum[idx].append(v)
            n_contrib += len(kept_idx)

        print(f"  frame {frame_id}: {n_contrib} point-view contributions")

    # ---------- Aggregate ----------
    thermal_value = np.full(n_points, np.nan, dtype=np.float32)
    coverage_count = np.zeros(n_points, dtype=np.int32)
    for i, vals in enumerate(accum):
        if vals:
            coverage_count[i] = len(vals)
            thermal_value[i] = np.median(vals) if AGGREGATE == "median" else np.mean(vals)

    covered = coverage_count > 0
    print(f"\nPoints with thermal coverage: {covered.sum()} / {n_points} "
          f"({100 * covered.sum() / n_points:.1f}%)")
    print(f"Mean views per covered point: {coverage_count[covered].mean():.2f}")

    # ---------- Build toggle-mode color arrays ----------
    # Same points, same order, in BOTH modes -- only the color array differs.
    # Covered points get a colormapped thermal color; uncovered points are GRAY_COLOR
    # in thermal mode (so gaps are visually obvious), but keep their real RGB color
    # in RGB mode.
    GRAY_COLOR = np.array([0.5, 0.5, 0.5])

    thermal_colors = np.tile(GRAY_COLOR, (n_points, 1))
    if covered.any():
        lo, hi = np.percentile(thermal_value[covered], [1, 99])
        norm = np.clip((thermal_value[covered] - lo) / max(hi - lo, 1e-6), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO).reshape(-1, 3)[:, ::-1]  # BGR->RGB
        thermal_colors[covered] = colored.astype(np.float64) / 255.0

    # ---------- Save toggle bundle (points + both color sets) ----------
    toggle_path = os.path.join(OUTPUT_DIR, "fused_toggle.npz")
    np.savez(toggle_path,
             points=points_world,
             rgb_colors=rgb_colors,
             thermal_colors=thermal_colors,
             covered=covered,
             thermal_intensity=thermal_value)
    print(f"\nSaved toggle bundle: {toggle_path}")
    print(f"  {n_points} points total, {covered.sum()} with thermal coverage "
          f"({100 * covered.sum() / n_points:.1f}%), rest shown gray in thermal mode")

    # also write two standalone .ply files (same points/order) for tools that can't
    # read the .npz bundle -- open either directly, or use view_toggle.py for live switching
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points_world)
    pcd_rgb.colors = o3d.utility.Vector3dVector(rgb_colors)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_rgb_mode.ply"), pcd_rgb)

    pcd_thermal = o3d.geometry.PointCloud()
    pcd_thermal.points = o3d.utility.Vector3dVector(points_world)
    pcd_thermal.colors = o3d.utility.Vector3dVector(thermal_colors)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "fused_thermal_mode.ply"), pcd_thermal)
    print(f"  also wrote fused_rgb_mode.ply and fused_thermal_mode.ply")


if __name__ == "__main__":
    main()