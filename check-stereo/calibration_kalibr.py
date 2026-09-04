"""
RGB stereo calibration for rgb1 (left) / rgb2 (right), sourced from Kalibr's
camchain output (rgb-apriltag-camchain.yaml) instead of discocal.

Unlike calibration.py, no undo_normalize_es / invert_transform is needed here:
Kalibr's T_cn_cnm1 is already the point-transform cv2.stereoRectify wants
directly -- X_rgb2 = R @ X_rgb1 + T -- since cam0=rgb1, cam1=rgb2.

Same CALIB dict interface as calibration.py, so this is a drop-in swap via
    --calib calibration_kalibr
on run_stereo.py / check_rectification.py / check_classical_stereo.py --
none of those need to change.
"""
import numpy as np
import cv2

IMAGE_SIZE = (1920, 1080)   # (width, height) -- matches camchain's resolution: [1920, 1080]
LEFT_IS = "rgb1"            # cam0 = rgb1 in the camchain

# intrinsics: [fx, fy, cx, cy]
LEFT_INTR = dict(fx=595.5131125646116, fy=595.4347842307814,
                  cx=937.1438588828621, cy=459.52906005794046)
RIGHT_INTR = dict(fx=604.5403230213659, fy=604.3848715029433,
                   cx=967.638578208722, cy=424.0641291666773)

# distortion_model: radtan == OpenCV's plumb-bob (k1, k2, p1, p2[, k3])
LEFT_DIST_RADTAN = [-0.006053561997736177, -0.010116448442781898,
                     -0.0007332457524041055, -0.0017822070415447345]
RIGHT_DIST_RADTAN = [0.03459723604219442, -0.030767694481543133,
                      -0.002550466204814176, 0.004007846658771188]


def build_K(i):
    return np.array([[i["fx"], 0.0, i["cx"]],
                      [0.0, i["fy"], i["cy"]],
                      [0.0, 0.0, 1.0]])


def build_dist_radtan(coeffs):
    """[k1, k2, p1, p2] -> OpenCV's [k1, k2, p1, p2, k3] with k3=0."""
    k1, k2, p1, p2 = coeffs
    return np.array([k1, k2, p1, p2, 0.0])


K1, D1 = build_K(LEFT_INTR), build_dist_radtan(LEFT_DIST_RADTAN)
K2, D2 = build_K(RIGHT_INTR), build_dist_radtan(RIGHT_DIST_RADTAN)

# T_cn_cnm1 from the camchain: point-transform cam0(rgb1) -> cam1(rgb2),
# i.e. X_rgb2 = R @ X_rgb1 + T. This is exactly cv2.stereoRectify's convention
# already -- no inversion or flip-correction needed.
R = np.array([
    [0.9999960692456313, 5.6045239682392934e-05, 0.0028032752643997677],
    [-0.00010995513721470303, 0.9998149922261677, 0.019234584207786273],
    [-0.002801678629801559, -0.019234816835876807, 0.9998110683614859],
])
T = np.array([[-0.14982557125354823], [-0.00011735766719689479], [0.006025489182246096]])

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
    print("R=\n", R, "\nT=\n", T, "\nbaseline (m):", np.linalg.norm(T))
    rvec, _ = cv2.Rodrigues(R)
    print("rotation angle (deg):", np.degrees(np.linalg.norm(rvec)))
