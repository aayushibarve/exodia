"""
Runs classical cv2.StereoSGBM + WLS filtering (ported from rgb-thermal-align.py's
compute_stereo_depth) on the exact same rectified pair Fast-FoundationStereo
has been struggling with -- no GPU, no FFS repo needed at all.

Point of this: if the classical method ALSO produces a warped/wedge-shaped
point cloud on this scene, the problem isn't specific to FFS's network --
it's something about the scene (long textureless floor, depth range, or
still-unresolved calibration) that both methods would trip on equally. If
classical stereo looks clean while FFS is warped, the issue is specific to
how the foundation model handles this scene.

Requires opencv-contrib (for cv2.ximgproc -- WLS filtering) and plyfile:
    pip install opencv-contrib-python plyfile

Usage:
    python check_classical_stereo.py --manifest extracted/manifest.json --index 3
"""
import argparse
import importlib
import os

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from rectify_utils_classical import build_rectification, rectify_pair


def compute_stereo_depth(gray_left, gray_right, Q, block_size, num_disp,
                          min_depth_m, max_depth_m, speckle_window=100, use_wls=True):
    """Ported verbatim from rgb-thermal-align.py."""
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
    return disparity, points_3d, valid


def write_ply(path, points, colors_bgr):
    """points: (N,3) float, colors_bgr: (N,3) uint8 in BGR (as read by cv2.imread)."""
    colors_rgb = colors_bgr[:, ::-1]
    vertex = np.zeros(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x'], vertex['y'], vertex['z'] = points[:, 0], points[:, 1], points[:, 2]
    vertex['red'], vertex['green'], vertex['blue'] = colors_rgb[:, 0], colors_rgb[:, 1], colors_rgb[:, 2]
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--calib", default="calibration")
    ap.add_argument("--out", default="classical_stereo_check")
    ap.add_argument("--block-size", type=int, default=7)
    ap.add_argument("--num-disp", type=int, default=16 * 16, help="Must be a multiple of 16")
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=15.0)
    ap.add_argument("--zfar", type=float, default=10.0, help="Cap on saved point cloud, meters")
    args = ap.parse_args()

    import json
    with open(args.manifest) as f:
        manifest = json.load(f)
    entry = next((e for e in manifest if e["pair_index"] == args.index), None)
    if entry is None:
        raise SystemExit(f"pair_index {args.index} not in manifest. "
                          f"Available: {[e['pair_index'] for e in manifest]}")

    calib_mod = importlib.import_module(args.calib)
    rect = build_rectification(calib_mod.CALIB)
    print(f"baseline: {rect['baseline_m']:.5f} m")

    img1 = cv2.imread(entry["rgb1_path"], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(entry["rgb2_path"], cv2.IMREAD_UNCHANGED)
    r1, r2 = rectify_pair(img1, img2, rect)
    rect_left, rect_right = (r1, r2) if rect["left_is"] == "rgb1" else (r2, r1)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    disparity, points_3d, valid = compute_stereo_depth(
        gray_left, gray_right, rect["Q"], args.block_size, args.num_disp,
        args.min_depth, args.max_depth,
    )
    print(f"valid: {valid.sum()} / {valid.size} px ({100 * valid.mean():.1f}%)")

    os.makedirs(args.out, exist_ok=True)

    disp_vis = cv2.applyColorMap(
        cv2.normalize(np.clip(disparity, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    cv2.imwrite(os.path.join(args.out, "disparity_classical.png"), disp_vis)
    cv2.imwrite(os.path.join(args.out, "rect_left.png"), rect_left)
    cv2.imwrite(os.path.join(args.out, "rect_right.png"), rect_right)

    zfar_mask = valid & (points_3d[..., 2] <= args.zfar)
    pts = points_3d[zfar_mask]
    colors = rect_left[zfar_mask]
    ply_path = os.path.join(args.out, "cloud_classical.ply")
    write_ply(ply_path, pts, colors)
    print(f"Saved {len(pts)} points -> {ply_path}")
    print(f"Saved disparity visualization -> {os.path.join(args.out, 'disparity_classical.png')}")


if __name__ == "__main__":
    main()
