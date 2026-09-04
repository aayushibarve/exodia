"""RGB stereo depth -> thermal-left / thermal-right registration pipeline."""

import os
import cv2
import matplotlib
matplotlib.use("Agg")  # headless: we only save figures, never show them
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import cg


# ============================================================
# CONFIG
# ============================================================

IMG_DIR = "/home/aayushi/exodia/depth_test/experiment_005/images/000003"
OUTPUT_DIR = "pipeline_outputs"

# --- RGB stereo pair ---
LEFT_IMG = os.path.join(IMG_DIR, "rgb1.png")
RIGHT_IMG = os.path.join(IMG_DIR, "rgb2.png")
IMG_SIZE = (1920, 1080)

LEFT_INTR = dict(fx=595.607, fy=594.714, cx=941.101, cy=458.698, skew=0.895965,
                  k1=-0.00257185, k2=-0.0145025)
RIGHT_INTR = dict(fx=598.517, fy=597.363, cx=970.896, cy=424.176, skew=0.866411,
                   k1=0.00464308, k2=-0.0195245)

# raw discocal camera0to1 pose (rgb1=camera0, rgb2=camera1): X_rgb1 = R@X_rgb2 + t
RGB_PAIR_RVEC_RAW = np.array([0.0171, -0.0259, 3.1381])
RGB_PAIR_TVEC_RAW = np.array([-0.1482, 0.0009, 0.0015]).reshape(3, 1)

# --- Thermal-left ---
THERMAL_LEFT_IMG = os.path.join(IMG_DIR, "thermal1_gray.png")
IMG_SIZE_THERMAL = (256, 192)

THERMAL_LEFT_INTR = dict(fx=267.489, fy=267.459, cx=129.32, cy=84.6925, skew=0.866478,
                          k1=-0.308768, k2=-0.0475762)

# thermal_left_to_rgb_left, already a point-transform: X_thermal_left = R@X_rgb_left + t
THERMAL_LEFT_RVEC = np.array([-0.0864, 0.0069, 0.0031])
THERMAL_LEFT_TVEC = np.array([0.0548, -0.0016, 0.0004]).reshape(3, 1)

# --- Thermal-right ---
THERMAL_RIGHT_IMG = os.path.join(IMG_DIR, "thermal2_gray.png")

THERMAL_RIGHT_INTR = dict(fx=263.845, fy=263.421, cx=133.939, cy=96.2706, skew=0.351657,
                           k1=-0.367637, k2=0.116875)

# raw discocal camera0to1 pose (camera0=rgb_left, camera1=thermal_right)
THERMAL_RIGHT_RVEC_RAW = np.array([0.006482, 0.042439, -0.017448])
THERMAL_RIGHT_TVEC_RAW = np.array([0.095655, -0.000551, 0.000212]).reshape(3, 1)

# --- Depth / stereo matching ---
MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 15.0
BLOCK_SIZE, NUM_DISP = 7, 16 * 16
COLORIZATION_SIGMA = 0.05
DEPTH_FILL = False   # if False, Z_filled keeps only directly-measured depth (raw + validity mask)
                     # -- e.g. for downstream 3D reconstruction with multi-view redundancy, where
                     # standard practice is to pass only trustworthy depth and let the reconstruction's
                     # own multi-view fusion handle completeness, rather than feeding it single-view
                     # interpolated values. Set True for the thermal-fusion path, which needs dense
                     # coverage. See colorization_fill_depth's docstring for the tradeoff.

# --- Thermal FOV crop / projection ---
DEPTH_NEAR_M, DEPTH_FAR_M = 0.1, 15.0
CROP_PAD_FRAC = 0.02
OCCLUSION_TOL_M = 0.25   # z-buffer occlusion tolerance, tune per scene scale

# --- Depth-aware hole filling (for pixels the occlusion z-buffer left unassigned) ---
THERMAL_FILL = False
THERMAL_FILL_KERNEL_RADIUS_PX = 8    # fixed local kernel radius, in RGB pixels, per diffusion pass
THERMAL_FILL_DEPTH_SIGMA_M = 0.05    # stiff: only near-exact depth matches contribute at all
THERMAL_FILL_MAX_PASSES = 50         # safety cap; fill also stops early on convergence/stall


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def invert_transform(R, t):
    """Pose (X_A = R@X_B + t) -> point-transform for cv2.stereoRectify."""
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
    """Build 3x3 intrinsics matrix from an intrinsics dict."""
    return np.array([[i["fx"], i["skew"], i["cx"]],
                      [0, i["fy"], i["cy"]],
                      [0, 0, 1]])


def build_dist(i):
    """Build OpenCV distortion vector (k1,k2,0,0,0) from an intrinsics dict."""
    return np.array([i["k1"], i["k2"], 0.0, 0.0, 0.0])


def build_pose(rx, ry, rz, tx, ty, tz):
    """Extrinsics -> (R, t) exactly as given: X_A = R@X_B + t."""
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    t = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
    return R, t


# ============================================================
# DEPTH ESTIMATION
# ============================================================

def compute_stereo_depth(gray_left, gray_right, Q, block_size, num_disp,
                          min_depth_m, max_depth_m, speckle_window=100, use_wls=True):
    """Run SGBM (+ optional WLS filtering) and back-project to metric depth Z."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=speckle_window, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp_l = stereo.compute(gray_left, gray_right)
    if use_wls:
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disp_r = right_matcher.compute(gray_right, gray_left)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        disp_out = wls.filter(disp_l, gray_left, disparity_map_right=disp_r)
    else:
        disp_out = disp_l
    disparity = disp_out.astype(np.float32) / 16.0

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    Z = points_3d[..., 2]
    valid = (disparity > 0) & (Z >= min_depth_m) & (Z <= max_depth_m)
    return disparity, Z, valid


def colorization_fill_depth(depth, valid_mask, guide, sigma=0.05, conn8=True, maxiter=500):
    """RGB-guided weighted-least-squares hole filling (Levin et al. colorization, adapted)."""
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    guide = guide.astype(np.float32) / 255.0

    H, W = depth.shape
    N = H * W
    idx = np.arange(N).reshape(H, W)

    rows, cols, vals = [], [], []
    diag = np.zeros(N, dtype=np.float64)

    offsets = [(0, 1), (1, 0)]
    if conn8:
        offsets += [(1, 1), (1, -1)]

    for dy, dx in offsets:
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        gi = guide[y0:y1, x0:x1]
        gj = guide[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        w = np.exp(-((gi - gj) ** 2) / (2 * sigma ** 2)).astype(np.float64).ravel()

        i_idx = idx[y0:y1, x0:x1].ravel()
        j_idx = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()

        rows.append(i_idx); cols.append(j_idx); vals.append(-w)
        rows.append(j_idx); cols.append(i_idx); vals.append(-w)
        np.add.at(diag, i_idx, w)
        np.add.at(diag, j_idx, w)

    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(N, N)).tocsr()

    valid_flat = valid_mask.ravel()
    b = np.zeros(N)
    b[valid_flat] = depth.ravel()[valid_flat]

    # enforce "known pixels stay fixed" (A row -> identity) via vectorized sparse ops,
    # NOT a per-row Python loop over ~1M+ pixels through a LIL conversion (was the dominant cost)
    D_unknown = sp.diags((~valid_flat).astype(np.float64))  # zeroes out rows of known pixels
    D_known = sp.diags(valid_flat.astype(np.float64))        # adds back identity rows there
    A = (D_unknown @ A + D_known).tocsr()

    x, _ = cg(A, b, rtol=1e-5, maxiter=maxiter)
    filled = x.reshape(H, W).astype(depth.dtype)
    filled[valid_mask] = depth[valid_mask]
    return filled


def depth_to_points_rectified(Z, P):
    """Dense (H,W,3) point cloud from a dense depth map using rectified pinhole intrinsics."""
    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    H, W = Z.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (us - cx) / fx * Z
    Y = (vs - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1)


# ============================================================
# RGB <-> THERMAL REGISTRATION
# ============================================================

def compute_thermal_fov_crop_in_rgb(K_thermal, D_thermal, P_rgb_rect, R1_rect,
                                     R_th_to_rgb, t_th_to_rgb,
                                     depth_near, depth_far, W_th, H_th, W_rgb, H_rgb,
                                     pad_frac=0.02, n_per_edge=60):
    """Bounding box in the rectified RGB frame that a thermal camera could possibly see."""
    top_u = np.linspace(0, W_th - 1, n_per_edge); top_v = np.zeros(n_per_edge)
    bot_u = np.linspace(0, W_th - 1, n_per_edge); bot_v = np.full(n_per_edge, H_th - 1)
    left_u = np.zeros(n_per_edge); left_v = np.linspace(0, H_th - 1, n_per_edge)
    right_u = np.full(n_per_edge, W_th - 1); right_v = np.linspace(0, H_th - 1, n_per_edge)
    u_perim = np.concatenate([top_u, bot_u, left_u, right_u])
    v_perim = np.concatenate([top_v, bot_v, left_v, right_v])

    pix = np.stack([u_perim, v_perim], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    norm = cv2.undistortPoints(pix, K_thermal, D_thermal)
    xn, yn = norm[:, 0, 0], norm[:, 0, 1]

    fx, fy, cx, cy = P_rgb_rect[0, 0], P_rgb_rect[1, 1], P_rgb_rect[0, 2], P_rgb_rect[1, 2]

    all_pts = []
    for d in (depth_near, depth_far):
        X_th = np.stack([xn * d, yn * d, np.full_like(xn, d)], axis=-1)
        X_rgb_raw = X_th @ R_th_to_rgb.T + t_th_to_rgb.reshape(1, 3)
        X_rgb_rect = X_rgb_raw @ R1_rect.T
        u = fx * X_rgb_rect[:, 0] / X_rgb_rect[:, 2] + cx
        v = fy * X_rgb_rect[:, 1] / X_rgb_rect[:, 2] + cy
        all_pts.append(np.stack([u, v], axis=-1))
    pts_rgb = np.concatenate(all_pts, axis=0)

    x_min, y_min = pts_rgb.min(axis=0)
    x_max, y_max = pts_rgb.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    x_min -= pad_frac * w; x_max += pad_frac * w
    y_min -= pad_frac * h; y_max += pad_frac * h

    x0 = int(max(0, np.floor(x_min))); y0 = int(max(0, np.floor(y_min)))
    x1 = int(min(W_rgb, np.ceil(x_max))); y1 = int(min(H_rgb, np.ceil(y_max)))
    return x0, y0, x1, y1, pts_rgb


def project_rgb_depth_to_thermal(points_3d_rect, valid_mask, R1_rect,
                                  R_rgb_to_thermal, t_rgb_to_thermal,
                                  K_thermal, D_thermal, thermal_img,
                                  min_z_m=0.05, occlusion_tol_m=0.25, fov_margin=1.15):
    """Project dense RGB depth into a thermal image, z-buffered to reject occluded points.

    fov_margin: normalized-coordinate FOV gate (relative to the thermal image's own corner
    rays) applied BEFORE distortion, to reject points whose ray direction is genuinely
    outside the lens's FOV -- guards against cv2.projectPoints folding an out-of-FOV ray
    back onto a plausible-looking in-bounds pixel (non-monotonic distortion polynomial).
    """
    H, W = valid_mask.shape
    ys, xs = np.where(valid_mask)
    pts_rect = points_3d_rect[valid_mask]

    pts_rgb_raw = pts_rect @ R1_rect
    pts_thermal = pts_rgb_raw @ R_rgb_to_thermal.T + t_rgb_to_thermal.reshape(1, 3)

    img_pts, _ = cv2.projectPoints(
        pts_thermal.reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3),
        K_thermal, D_thermal,
    )
    img_pts = img_pts.reshape(-1, 2)
    z_thermal = pts_thermal[:, 2]

    Ht, Wt = thermal_img.shape
    in_front = z_thermal > min_z_m
    in_bounds = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < Wt) & \
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < Ht)

    # pre-distortion FOV gate: legitimate normalized-coordinate range from the thermal
    # image's own corners, undistorted -- rejects rays no real thermal pixel could produce
    corners = np.array([[0, 0], [Wt - 1, 0], [0, Ht - 1], [Wt - 1, Ht - 1]], dtype=np.float64)
    corners_norm = cv2.undistortPoints(corners.reshape(-1, 1, 2), K_thermal, D_thermal).reshape(-1, 2)
    xn_max = np.abs(corners_norm[:, 0]).max() * fov_margin
    yn_max = np.abs(corners_norm[:, 1]).max() * fov_margin
    xn = pts_thermal[:, 0] / np.where(in_front, z_thermal, np.nan)
    yn = pts_thermal[:, 1] / np.where(in_front, z_thermal, np.nan)
    in_fov_angle = (np.abs(xn) <= xn_max) & (np.abs(yn) <= yn_max)

    candidate = in_front & in_bounds & in_fov_angle

    # true FOV footprint: every pixel whose ray genuinely reaches the real thermal sensor,
    # BEFORE the z-buffer discards occluded ones -- this is what distinguishes "occluded"
    # (inside true_fov_mask, lost the z-buffer) from "sensor never saw this at all" (outside it)
    true_fov_mask = np.zeros((H, W), dtype=bool)
    true_fov_mask[ys[candidate], xs[candidate]] = True

    # z-buffer: keep only the frontmost point per thermal pixel bin
    ui = np.clip(np.round(img_pts[:, 0]).astype(np.int64), 0, Wt - 1)
    vi = np.clip(np.round(img_pts[:, 1]).astype(np.int64), 0, Ht - 1)
    flat_bin = vi * Wt + ui

    zbuf = np.full(Ht * Wt, np.inf, dtype=np.float64)
    cand_idx = np.where(candidate)[0]
    np.minimum.at(zbuf, flat_bin[cand_idx], z_thermal[cand_idx])

    z_at_bin = zbuf[flat_bin[cand_idx]]
    is_frontmost = z_thermal[cand_idx] <= z_at_bin + occlusion_tol_m

    keep = np.zeros_like(candidate)
    keep[cand_idx[is_frontmost]] = True

    map_x = np.full((H, W), -1, dtype=np.float32)
    map_y = np.full((H, W), -1, dtype=np.float32)
    map_x[ys[keep], xs[keep]] = img_pts[keep, 0]
    map_y[ys[keep], xs[keep]] = img_pts[keep, 1]

    thermal_warped = cv2.remap(thermal_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_proj = np.zeros((H, W), dtype=bool)
    valid_proj[ys[keep], xs[keep]] = True
    return thermal_warped, valid_proj, map_x, map_y, true_fov_mask


def depth_aware_thermal_fill(thermal_warped, valid_proj, Z_filled, region_mask,
                              kernel_radius_px=8, depth_sigma_m=0.05, spatial_sigma_px=None,
                              min_weight_sum=1e-6, max_passes=50):
    """Diffusion-style hole fill: repeatedly average each unfilled hole pixel against
    currently-valid pixels within a small FIXED local kernel, weighted by depth similarity
    (so a closer object like a hand can't bleed onto a farther one like the floor behind it,
    regardless of spatial closeness) and by distance within the kernel.

    Because the kernel is small and fixed, hole pixels deep in the interior have no valid
    neighbors on pass 1. Each pass fills whatever it can from currently-valid data (original
    or already-filled), then the next pass treats those newly-filled pixels as valid too --
    the fill front propagates inward roughly one kernel-radius per pass until the hole closes
    or no further progress is possible (remaining pixels have no matching-depth data reachable
    at all, e.g. an isolated bad depth reading with nothing similar nearby).
    """
    if spatial_sigma_px is None:
        spatial_sigma_px = kernel_radius_px / 2.0

    hole_mask_full = region_mask & ~valid_proj
    filled_full = thermal_warped.astype(np.float32).copy()
    filled_mask_full = valid_proj.copy()
    if not hole_mask_full.any():
        return filled_full, filled_mask_full

    # restrict work to a tight box around the HOLES (plus kernel-radius margin for valid
    # context) rather than the full region bounding box -- holes are usually a small,
    # localized area, and processing the whole region every pass wastes most of the work
    # on pixels that are already filled and never change
    hole_ys, hole_xs = np.where(hole_mask_full)
    H_full, W_full = region_mask.shape
    y0 = max(0, hole_ys.min() - kernel_radius_px)
    y1 = min(H_full, hole_ys.max() + 1 + kernel_radius_px)
    x0 = max(0, hole_xs.min() - kernel_radius_px)
    x1 = min(W_full, hole_xs.max() + 1 + kernel_radius_px)

    therm = filled_full[y0:y1, x0:x1].copy()
    valid = filled_mask_full[y0:y1, x0:x1].copy()
    depth = Z_filled[y0:y1, x0:x1].astype(np.float32)
    region = region_mask[y0:y1, x0:x1]

    # fixed circular kernel offsets (precomputed once)
    offsets = []
    for dy in range(-kernel_radius_px, kernel_radius_px + 1):
        for dx in range(-kernel_radius_px, kernel_radius_px + 1):
            if dy == 0 and dx == 0:
                continue
            dist = (dy * dy + dx * dx) ** 0.5
            if dist <= kernel_radius_px:
                offsets.append((dy, dx, dist))

    total_holes = int((region & ~valid).sum())
    filled_count = 0
    n_passes = 0
    R = kernel_radius_px

    for n_passes in range(1, max_passes + 1):
        unfilled_ys, unfilled_xs = np.where(region & ~valid)
        if len(unfilled_ys) == 0:
            break

        # pad once per pass; neighbor lookups become cheap fancy-indexing below
        valid_pad = np.pad(valid, R, constant_values=False)
        therm_pad = np.pad(therm, R, constant_values=0.0)
        depth_pad = np.pad(depth, R, constant_values=0.0)

        hole_depth_now = depth[unfilled_ys, unfilled_xs]      # (M,) -- shrinks every pass

        numerator = np.zeros(len(unfilled_ys), dtype=np.float32)
        denominator = np.zeros(len(unfilled_ys), dtype=np.float32)

        for dy, dx, dist in offsets:
            # look up each still-unfilled pixel's neighbor directly -- work scales with M
            # (the shrinking unfilled count), not H*W (the fixed bounding-box area)
            ny = unfilled_ys - dy + R
            nx = unfilled_xs - dx + R
            n_valid = valid_pad[ny, nx]
            n_therm = therm_pad[ny, nx]
            n_depth = depth_pad[ny, nx]

            depth_diff = hole_depth_now - n_depth
            spatial_w = np.exp(-(dist ** 2) / (2 * spatial_sigma_px ** 2))
            depth_w = np.exp(-(depth_diff ** 2) / (2 * depth_sigma_m ** 2))
            w = spatial_w * depth_w * n_valid

            numerator += w * n_therm
            denominator += w

        can_fill = denominator > min_weight_sum
        if not can_fill.any():
            break  # stalled: remaining holes have no matching-depth data within kernel reach

        fill_ys, fill_xs = unfilled_ys[can_fill], unfilled_xs[can_fill]
        therm[fill_ys, fill_xs] = numerator[can_fill] / denominator[can_fill]
        valid[fill_ys, fill_xs] = True
        filled_count += int(can_fill.sum())

    filled_full[y0:y1, x0:x1] = therm
    filled_mask_full[y0:y1, x0:x1] = valid

    print(f"  depth-aware fill (diffusion): {filled_count}/{total_holes} hole pixels filled "
          f"over {n_passes} passes ({total_holes - filled_count} left unfilled -- no "
          f"matching-depth data reachable within kernel_radius={kernel_radius_px}px)")
    return filled_full, filled_mask_full


# ============================================================
# DIAGNOSTICS / PLOTTING (all figures auto-saved to OUTPUT_DIR)
# ============================================================

def save_fig(fig, name):
    """Save a figure to OUTPUT_DIR and close it."""
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def epipolar_error(gray_left, gray_right, max_corners=500, half=9,
                    search_v=60, max_disp=260, corr_thresh=0.85):
    """Template-match corners left->right to estimate residual vertical (epipolar) error."""
    pts = cv2.goodFeaturesToTrack(gray_left, maxCorners=max_corners,
                                   qualityLevel=0.01, minDistance=12)
    if pts is None:
        return np.array([])
    pts = pts.reshape(-1, 2)
    rows = []
    for x, y in pts:
        x, y = int(x), int(y)
        if x - half < 0 or y - half < 0 or x + half >= gray_left.shape[1] or y + half >= gray_left.shape[0]:
            continue
        templ = gray_left[y - half:y + half + 1, x - half:x + half + 1]
        y0, y1 = max(0, y - search_v), min(gray_right.shape[0], y + search_v)
        x0, x1 = max(0, x - max_disp), max(0, x)
        if x1 - x0 < templ.shape[1] + 1 or y1 - y0 < templ.shape[0] + 1:
            continue
        region = gray_right[y0:y1, x0:x1]
        res = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        if maxval < corr_thresh:
            continue
        mx, my = x0 + maxloc[0] + half, y0 + maxloc[1] + half
        rows.append((x, y, x - mx, my - y, maxval))
    return np.array(rows)


def plot_rectified_pair(rect_left, rect_right, title, name, line_step=40):
    """Side-by-side rectified pair with horizontal guide lines."""
    stack = np.hstack([rect_left, rect_right])
    stack = cv2.cvtColor(stack, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(stack)
    for y in range(0, stack.shape[0], line_step):
        ax.axhline(y, color="lime", linewidth=0.5)
    ax.set_title(title)
    save_fig(fig, name)


def plot_epipolar_diagnostic(gray_left, gray_right, name, **kwargs):
    """Scatter plot of vertical epipolar error vs x and y."""
    res = epipolar_error(gray_left, gray_right, **kwargs)
    if len(res) < 5:
        print(f"only {len(res)} confident matches")
        return
    x, y, dx, dy, conf = res.T
    print(f"{len(res)} matches | dy mean={dy.mean():.2f}px std={dy.std():.2f}px")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(x, dy, s=6, alpha=0.5); axes[0].axhline(0, color="r")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("vertical error")
    axes[1].scatter(y, dy, s=6, alpha=0.5); axes[1].axhline(0, color="r")
    axes[1].set_xlabel("y")
    save_fig(fig, name)


def plot_disparity_and_depth(disparity, Z, valid, min_depth_m, max_depth_m,
                              name_prefix, cmap_disp="turbo", cmap_depth="inferno"):
    """Disparity map + validity-masked depth map."""
    print(f"valid: {valid.sum()} / {valid.size} px ({100 * valid.mean():.1f}%)")
    if valid.any():
        print(f"depth range: {Z[valid].min():.3f}-{Z[valid].max():.3f} m, median {np.median(Z[valid]):.3f} m")

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(np.clip(disparity, 0, None), cmap=cmap_disp)
    plt.colorbar(label="disparity (px)"); plt.title("Disparity (WLS-filtered)")
    save_fig(fig, f"{name_prefix}_disparity.png")

    depth_display = np.where(valid, Z, np.nan)
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(depth_display, cmap=cmap_depth, vmin=min_depth_m, vmax=max_depth_m)
    plt.colorbar(label="depth (m)"); plt.title(f"Depth ({min_depth_m}-{max_depth_m} m)")
    save_fig(fig, f"{name_prefix}_depth.png")


def plot_filled_depth(Z_filled, min_depth_m, max_depth_m, name, filled=True):
    """Depth map -- dense (colorization-filled) or raw (measured-only), per DEPTH_FILL."""
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(Z_filled, cmap="inferno", vmin=min_depth_m, vmax=max_depth_m)
    plt.colorbar(label="depth (m)")
    title = "Z_filled: colorization-filled depth" if filled else "Z_filled: raw measured depth (unfilled)"
    plt.title(title)
    plt.axis("off")
    save_fig(fig, name)


def plot_fov_crop(rect_left, perimeter_pts, x0, y0, x1, y1, title, name):
    """RGB view with the projected thermal-FOV perimeter and resulting crop box."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB))
    ax.scatter(perimeter_pts[:, 0], perimeter_pts[:, 1], s=3, c="cyan")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2))
    ax.set_title(title)
    ax.axis("off")
    save_fig(fig, name)


def plot_thermal_overlay(rect_rgb_bgr, thermal_warped, valid_proj, name, alpha=0.55, cmap="inferno"):
    """3-panel figure: rectified RGB, warped thermal, and alpha-blended overlay. Returns the blend."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(rect_rgb_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Rectified RGB"); axes[0].axis("off")

    im = axes[1].imshow(np.ma.masked_where(~valid_proj, thermal_warped), cmap=cmap)
    axes[1].set_title("Thermal warped into RGB view"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    rgb_disp = cv2.cvtColor(rect_rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    therm_rgb = plt.get_cmap(cmap)(thermal_warped / 255.0)[..., :3]
    blended = rgb_disp.copy()
    blended[valid_proj] = (1 - alpha) * rgb_disp[valid_proj] + alpha * therm_rgb[valid_proj]
    axes[2].imshow(blended)
    axes[2].set_title("Overlay"); axes[2].axis("off")

    save_fig(fig, name)
    return blended


def plot_side_by_side_overlays(rect_left, thermal_warped_left, valid_proj_left,
                                thermal_warped_right, valid_proj_right, name, alpha=0.55):
    """Thermal-left overlay next to thermal-right overlay, for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    rgb_disp = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cmap = plt.get_cmap("inferno")

    therm_left_rgb = cmap(thermal_warped_left / 255.0)[..., :3]
    blended_left = rgb_disp.copy()
    blended_left[valid_proj_left] = (1 - alpha) * rgb_disp[valid_proj_left] + alpha * therm_left_rgb[valid_proj_left]
    axes[0].imshow(blended_left); axes[0].set_title("Thermal-LEFT overlay"); axes[0].axis("off")

    therm_right_rgb = cmap(thermal_warped_right / 255.0)[..., :3]
    blended_right = rgb_disp.copy()
    blended_right[valid_proj_right] = (1 - alpha) * rgb_disp[valid_proj_right] + alpha * therm_right_rgb[valid_proj_right]
    axes[1].imshow(blended_right); axes[1].set_title("Thermal-RIGHT overlay"); axes[1].axis("off")

    save_fig(fig, name)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- RGB stereo rectification ----------
    R_raw, _ = cv2.Rodrigues(RGB_PAIR_RVEC_RAW)
    R_pose, t_pose = undo_normalize_es(R_raw, RGB_PAIR_TVEC_RAW)
    R, T = invert_transform(R_pose, t_pose)

    K_left, D_left = build_K(LEFT_INTR), build_dist(LEFT_INTR)
    K_right, D_right = build_K(RIGHT_INTR), build_dist(RIGHT_INTR)

    img_left = cv2.imread(LEFT_IMG)
    img_right = cv2.imread(RIGHT_IMG)
    assert img_left is not None and img_right is not None, "check your RGB image paths"

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, IMG_SIZE, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, IMG_SIZE, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, IMG_SIZE, cv2.CV_32FC1)
    rect_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    print("baseline:", np.linalg.norm(T), "m | ROI L/R:", roi1, roi2)
    plot_rectified_pair(rect_left, rect_right, "RGB rectified pair", "01_rectified_pair.png")
    plot_epipolar_diagnostic(gray_left, gray_right, "02_epipolar_diagnostic.png")

    # ---------- RGB stereo depth ----------
    disparity, Z, valid = compute_stereo_depth(
        gray_left, gray_right, Q, BLOCK_SIZE, NUM_DISP, MIN_DEPTH_M, MAX_DEPTH_M
    )
    plot_disparity_and_depth(disparity, Z, valid, MIN_DEPTH_M, MAX_DEPTH_M, "03")

    if DEPTH_FILL:
        Z_filled = colorization_fill_depth(Z, valid, rect_left, sigma=COLORIZATION_SIGMA)
    else:
        Z_filled = np.where(valid, Z, 0.0).astype(Z.dtype)  # unmeasured px left at 0, unused
        depth_trust_mask = valid                             # only directly-measured px trusted
        print(f"depth fill disabled: {depth_trust_mask.sum()} directly-measured px usable downstream "
              f"(vs {depth_trust_mask.size} total)")
    plot_filled_depth(Z_filled, MIN_DEPTH_M, MAX_DEPTH_M, "04_depth_filled.png", filled=DEPTH_FILL)

    points_3d_filled = depth_to_points_rectified(Z_filled, P1)
    H_rgb, W_rgb = Z_filled.shape

    # ---------- Thermal-left: RGB -> thermal extrinsic ----------
    K_thermal_left = build_K(THERMAL_LEFT_INTR)
    D_thermal_left = build_dist(THERMAL_LEFT_INTR)
    thermal_left_img = cv2.imread(THERMAL_LEFT_IMG, cv2.IMREAD_GRAYSCALE)
    assert thermal_left_img is not None, "check thermal-left image path"
    assert thermal_left_img.shape[::-1] == IMG_SIZE_THERMAL

    # already the rgb->thermal point-transform, no inversion
    R_rgb_to_thermal_left, t_rgb_to_thermal_left = build_pose(
        *THERMAL_LEFT_RVEC.tolist(), *THERMAL_LEFT_TVEC.ravel().tolist()
    )
    R_thermal_left_to_rgb, t_thermal_left_to_rgb = invert_transform(
        R_rgb_to_thermal_left, t_rgb_to_thermal_left
    )

    x0_l, y0_l, x1_l, y1_l, perim_l = compute_thermal_fov_crop_in_rgb(
        K_thermal_left, D_thermal_left, P1, R1,
        R_thermal_left_to_rgb, t_thermal_left_to_rgb,
        depth_near=DEPTH_NEAR_M, depth_far=DEPTH_FAR_M,
        W_th=IMG_SIZE_THERMAL[0], H_th=IMG_SIZE_THERMAL[1],
        W_rgb=W_rgb, H_rgb=H_rgb, pad_frac=CROP_PAD_FRAC,
    )
    print("thermal-LEFT FOV crop box in RGB:", (x0_l, y0_l, x1_l, y1_l))
    plot_fov_crop(rect_left, perim_l, x0_l, y0_l, x1_l, y1_l,
                  "Thermal-LEFT FOV perimeter + crop box", "05_thermal_left_fov_crop.png")

    crop_mask_left = np.zeros((H_rgb, W_rgb), dtype=bool)
    crop_mask_left[y0_l:y1_l, x0_l:x1_l] = True
    if not DEPTH_FILL:
        n_before = crop_mask_left.sum()
        crop_mask_left &= depth_trust_mask
        print(f"thermal-LEFT crop restricted by depth trust: {n_before} -> {crop_mask_left.sum()} px")

    thermal_warped_left, valid_proj_left, _, _, true_fov_left = project_rgb_depth_to_thermal(
        points_3d_filled, crop_mask_left, R1,
        R_rgb_to_thermal_left, t_rgb_to_thermal_left,
        K_thermal_left, D_thermal_left, thermal_left_img,
        occlusion_tol_m=OCCLUSION_TOL_M,
    )
    print(f"projected into thermal-LEFT: {valid_proj_left.sum()} / {crop_mask_left.sum()} "
          f"({100 * valid_proj_left.sum() / crop_mask_left.sum():.1f}%)")

    if THERMAL_FILL:
        thermal_warped_left, valid_proj_left = depth_aware_thermal_fill(
            thermal_warped_left, valid_proj_left, Z_filled, true_fov_left,
            kernel_radius_px=THERMAL_FILL_KERNEL_RADIUS_PX,
            depth_sigma_m=THERMAL_FILL_DEPTH_SIGMA_M,
            max_passes=THERMAL_FILL_MAX_PASSES,
        )

    plot_thermal_overlay(rect_left, thermal_warped_left, valid_proj_left, "06_thermal_left_overlay.png")

    # ---------- Thermal-right: RGB -> thermal extrinsic (direct calibration, needs inverting) ----------
    K_thermal_right = build_K(THERMAL_RIGHT_INTR)
    D_thermal_right = build_dist(THERMAL_RIGHT_INTR)
    thermal_right_img = cv2.imread(THERMAL_RIGHT_IMG, cv2.IMREAD_GRAYSCALE)
    assert thermal_right_img is not None, "check thermal-right image path"
    assert thermal_right_img.shape[::-1] == IMG_SIZE_THERMAL

    R_raw_tr, _ = cv2.Rodrigues(THERMAL_RIGHT_RVEC_RAW)
    R_pose_tr, t_pose_tr = undo_normalize_es(R_raw_tr, THERMAL_RIGHT_TVEC_RAW)
    R_rgb_to_thermal_right, t_rgb_to_thermal_right = invert_transform(R_pose_tr, t_pose_tr)
    R_thermal_right_to_rgb, t_thermal_right_to_rgb = invert_transform(
        R_rgb_to_thermal_right, t_rgb_to_thermal_right
    )

    x0_r, y0_r, x1_r, y1_r, perim_r = compute_thermal_fov_crop_in_rgb(
        K_thermal_right, D_thermal_right, P1, R1,
        R_thermal_right_to_rgb, t_thermal_right_to_rgb,
        depth_near=DEPTH_NEAR_M, depth_far=DEPTH_FAR_M,
        W_th=IMG_SIZE_THERMAL[0], H_th=IMG_SIZE_THERMAL[1],
        W_rgb=W_rgb, H_rgb=H_rgb, pad_frac=CROP_PAD_FRAC,
    )
    print("thermal-RIGHT FOV crop box in RGB:", (x0_r, y0_r, x1_r, y1_r))
    plot_fov_crop(rect_left, perim_r, x0_r, y0_r, x1_r, y1_r,
                  "Thermal-RIGHT FOV perimeter + crop box", "07_thermal_right_fov_crop.png")

    crop_mask_right = np.zeros((H_rgb, W_rgb), dtype=bool)
    crop_mask_right[y0_r:y1_r, x0_r:x1_r] = True
    if not DEPTH_FILL:
        n_before = crop_mask_right.sum()
        crop_mask_right &= depth_trust_mask
        print(f"thermal-RIGHT crop restricted by depth trust: {n_before} -> {crop_mask_right.sum()} px")

    thermal_warped_right, valid_proj_right, _, _, true_fov_right = project_rgb_depth_to_thermal(
        points_3d_filled, crop_mask_right, R1,
        R_rgb_to_thermal_right, t_rgb_to_thermal_right,
        K_thermal_right, D_thermal_right, thermal_right_img,
        occlusion_tol_m=OCCLUSION_TOL_M,
    )
    print(f"projected into thermal-RIGHT: {valid_proj_right.sum()} / {crop_mask_right.sum()} "
          f"({100 * valid_proj_right.sum() / crop_mask_right.sum():.1f}%)")

    if THERMAL_FILL:
        thermal_warped_right, valid_proj_right = depth_aware_thermal_fill(
            thermal_warped_right, valid_proj_right, Z_filled, true_fov_right,
            kernel_radius_px=THERMAL_FILL_KERNEL_RADIUS_PX,
            depth_sigma_m=THERMAL_FILL_DEPTH_SIGMA_M,
            max_passes=THERMAL_FILL_MAX_PASSES,
        )

    plot_thermal_overlay(rect_left, thermal_warped_right, valid_proj_right, "08_thermal_right_overlay.png")

    # ---------- Combined left+right coverage ----------
    both_valid = valid_proj_left & valid_proj_right
    only_left = valid_proj_left & ~valid_proj_right
    only_right = valid_proj_right & ~valid_proj_left
    any_valid = valid_proj_left | valid_proj_right

    thermal_combined = np.zeros_like(thermal_warped_left, dtype=np.float32)
    thermal_combined[only_left] = thermal_warped_left[only_left]
    thermal_combined[only_right] = thermal_warped_right[only_right]
    thermal_combined[both_valid] = 0.5 * (thermal_warped_left[both_valid].astype(np.float32)
                                           + thermal_warped_right[both_valid].astype(np.float32))
    thermal_combined_u8 = thermal_combined.astype(np.uint8)

    print(f"left-only: {only_left.sum()}  right-only: {only_right.sum()}  "
          f"both: {both_valid.sum()}  union: {any_valid.sum()}")
    plot_thermal_overlay(rect_left, thermal_combined_u8, any_valid, "09_thermal_combined_overlay.png")

    plot_side_by_side_overlays(
        rect_left, thermal_warped_left, valid_proj_left,
        thermal_warped_right, valid_proj_right, "10_left_vs_right_overlay.png"
    )

    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()