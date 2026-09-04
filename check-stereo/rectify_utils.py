"""
Rectification helpers.

Fast-FoundationStereo requires rectified, undistorted left/right images
(horizontal epipolar lines) -- see repo tips. /rgb1/image_raw and
/rgb2/image_raw are raw camera output, so every pair gets rectified here
before being handed to the network.
"""
import cv2
import numpy as np


def build_rectification(calib):
    """Compute rectification maps once from a CALIB dict (see calibration_template.py).

    Returns a dict with the remap tables for both cameras, the new (shared)
    intrinsic matrix, the post-rectification baseline in meters -- which
    is exactly what Fast-FoundationStereo's --intrinsic_file needs -- and Q,
    OpenCV's 4x4 reprojection matrix for cv2.reprojectImageTo3D (used by the
    classical-stereo comparison script, not by the FFS path).
    """
    K1, D1 = calib["K1"], calib["D1"]
    K2, D2 = calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]
    size = calib["image_size"]  # (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)

    # Baseline in meters: P2[0,3] = -fx' * baseline  (OpenCV convention)
    fx_rect = P1[0, 0]
    baseline_m = abs(P2[0, 3] / fx_rect)

    return {
        "map1x": map1x, "map1y": map1y,
        "map2x": map2x, "map2y": map2y,
        "K_rect": P1[:3, :3],
        "Q": Q,
        "baseline_m": baseline_m,
        "left_is": calib["left_is"],
    }


def rectify_pair(img1, img2, rect):
    """img1/img2 correspond to rgb1/rgb2 respectively (independent of which is
    physically 'left' -- that's handled when picking left_file/right_file)."""
    r1 = cv2.remap(img1, rect["map1x"], rect["map1y"], cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, rect["map2x"], rect["map2y"], cv2.INTER_LINEAR)
    return r1, r2


def write_intrinsic_file(path, K_rect, baseline_m):
    """Fast-FoundationStereo format: line 1 = flattened 1x9 K, line 2 = baseline (m)."""
    flat = np.asarray(K_rect, dtype=np.float64).flatten()
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.10f}" for v in flat) + "\n")
        f.write(f"{baseline_m:.10f}\n")
