"""
RGB stereo calibration for rgb1 (left) / rgb2 (right), ported verbatim from
the RGB-pair section of rgb-thermal-align.py -- including undo_normalize_es
(discocal's 180deg flip correction) and invert_transform (pose -> the
point-transform cv2.stereoRectify actually wants). Nothing here is
re-derived; it's the same math, same order of operations, just repackaged
into the CALIB dict rectify_utils.py expects.
"""
import numpy as np
import cv2
#0.895965
IMAGE_SIZE = (1920, 1080)   # (width, height) -- rgb1.png / rgb2.png
LEFT_IS = "rgb1"            # rgb1 = camera0 = LEFT_IMG in the source script

LEFT_INTR = dict(fx=595.607, fy=594.714, cx=941.101, cy=458.698, skew=0.0,
                  k1=-0.00257185, k2=-0.0145025)
                  #0.866411
RIGHT_INTR = dict(fx=598.517, fy=597.363, cx=970.896, cy=424.176, skew=0.0,
                   k1=0.00464308, k2=-0.0195245)

# raw discocal camera0to1 pose (rgb1=camera0, rgb2=camera1): X_rgb1 = R@X_rgb2 + t
RGB_PAIR_RVEC_RAW = np.array([0.0171, -0.0259, 3.1381])
RGB_PAIR_TVEC_RAW = np.array([-0.1482, 0.0009, 0.0015]).reshape(3, 1)


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
    return np.array([[i["fx"], i["skew"], i["cx"]],
                      [0, i["fy"], i["cy"]],
                      [0, 0, 1]])


def build_dist(i):
    return np.array([i["k1"], i["k2"], 0.0, 0.0, 0.0])


K1, D1 = build_K(LEFT_INTR), build_dist(LEFT_INTR)
K2, D2 = build_K(RIGHT_INTR), build_dist(RIGHT_INTR)

# rvec/tvec -> R -> undo the 180deg normalize_Es flip if the raw magnitude signals
# it fired (RGB_PAIR_RVEC_RAW has norm ~3.138 rad = ~179.8deg, i.e. right at the
# threshold -- this DOES trigger the flip; verify by uncommenting the print below
# after import if you ever swap in different raw rvecs).
_R_raw, _ = cv2.Rodrigues(RGB_PAIR_RVEC_RAW)
_R_pose, _t_pose = undo_normalize_es(_R_raw, RGB_PAIR_TVEC_RAW)
# _R_pose/_t_pose is a POSE: X_rgb1 = R@X_rgb2 + t. cv2.stereoRectify wants the
# point-transform the other way (X_right = R@X_left + T), so invert it.
R, T = invert_transform(_R_pose, _t_pose)

CALIB = dict(
    image_size=IMAGE_SIZE,
    left_is=LEFT_IS,
    K1=K1, D1=D1,
    K2=K2, D2=D2,
    R=R, T=T,
)

if __name__ == "__main__":
    print("K1=\n", K1, "\nD1=", D1)
    print("K2=\n", K2, "\nD2=", D2)
    print("rvec norm (deg):", np.degrees(np.linalg.norm(cv2.Rodrigues(_R_raw)[0])))
    print("R=\n", R, "\nT=\n", T, "\nbaseline (m):", np.linalg.norm(T))
