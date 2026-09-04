"""
RGB Camera Calibration using a Chessboard Pattern
Usage
-----
1. Live capture from a webcam:
    python calibrate_camera.py --live --rows 6 --cols 9 --square-size 0.025

2. Calibrate from a folder of already-captured images:
    python calibrate_camera.py --images-dir ./calib_images --rows 6 --cols 9 --square-size 0.025

Notes
-----
- rows/cols refer to INNER corners of the checkerboard (not the number of
  squares). A standard "9x6" chessboard has 9x6 inner corners, which
  corresponds to a 10x7 grid of squares.
- square-size is the physical side length of one square, in meters
  (or any consistent unit you want your calibration output in).
- Press 'c' to capture a frame during live mode, 'q' to quit and calibrate
  with whatever frames have been captured so far.
"""

import argparse
import glob
import os
import subprocess
import cv2
import numpy as np


def find_corners(gray, pattern_size, use_sb=True):
    """Find chessboard corners in a grayscale image, refined to subpixel accuracy.

    use_sb=True uses OpenCV's newer findChessboardCornersSB detector, which is
    generally more robust than the classic detector for screen-displayed
    patterns, varying lighting, and slight blur.
    """
    if use_sb:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if found:
            return True, corners
        # Fall back to the classic detector if SB fails to find the pattern
        return find_corners(gray, pattern_size, use_sb=False)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def build_object_points(pattern_size, square_size):
    """3D points of the chessboard corners in the board's own coordinate frame (Z=0 plane)."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate(object_points, image_points, image_size):
    """Run OpenCV's camera calibration and return results as a dict."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    # Per-view reprojection error (useful for spotting bad frames)
    per_view_errors = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(error)

    return {
        "rms_reprojection_error": ret,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_view_errors": per_view_errors,
    }


def save_calibration(result, image_size, pattern_size, square_size, out_path):
    np.savez(
        out_path,
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
        rms_reprojection_error=result["rms_reprojection_error"],
        per_view_errors=np.array(result["per_view_errors"]),
        image_width=image_size[0],
        image_height=image_size[1],
        pattern_cols=pattern_size[0],
        pattern_rows=pattern_size[1],
        square_size=square_size,
    )
    print(f"\nSaved calibration to: {out_path}")


def print_summary(result):
    print("\n===== CALIBRATION RESULTS =====")
    print(f"RMS reprojection error: {result['rms_reprojection_error']:.4f} px")
    print("\nCamera matrix (K):")
    print(result["camera_matrix"])
    print("\nDistortion coefficients (k1, k2, p1, p2, k3, ...):")
    print(result["dist_coeffs"].ravel())

    errors = result["per_view_errors"]
    print(f"\nPer-view reprojection error: mean={np.mean(errors):.4f}px, "
          f"max={np.max(errors):.4f}px (over {len(errors)} views)")
    worst = int(np.argmax(errors))
    print(f"Worst view index: {worst} (error={errors[worst]:.4f}px) "
          f"— consider discarding this frame and re-running if error is high.")


def run_live_capture(cap, pattern_size, min_frames, display_scale=1.0):
    """Interactive capture loop: shows live preview with detected corners,
    lets the user press 'c' to accept a frame, 'q' to finish."""
    object_points = []
    image_points = []
    image_size = None

    print("\nLive capture mode.")
    print("  'c' = capture current frame (only works when corners are detected)")
    print("  'q' = quit capture and run calibration")
    print(f"  Need at least {min_frames} good frames for a usable calibration.\n")

    win_main = "Calibration capture"
    win_gray = "Grayscale feed (check focus/contrast here)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_gray, cv2.WINDOW_NORMAL)
    # Place windows side by side near the top-left so they don't cover the
    # whole screen or overlap a terminal/editor window.
    cv2.moveWindow(win_main, 20, 20)
    cv2.moveWindow(win_gray, 20, 20)  # resized/repositioned properly once we know frame size below

    windows_positioned = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read failed, stopping capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = find_corners(gray, pattern_size, use_sb=True)

        display = frame.copy()
        #print(display.shape)
        cv2.drawChessboardCorners(display, pattern_size, corners, found) if found else None
        expected = pattern_size[0] * pattern_size[1]
        status = (f"Captured: {len(image_points)}  |  Board detected: {found}  |  "
                  f"Looking for {pattern_size[1]}x{pattern_size[0]} inner corners ({expected} pts)")
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0) if found else (0, 0, 255), 2)

        if display_scale != 1.0:
            display_small = cv2.resize(display, None, fx=display_scale, fy=display_scale,
                                        interpolation=cv2.INTER_AREA)
            gray_small = cv2.resize(gray, None, fx=display_scale, fy=display_scale,
                                     interpolation=cv2.INTER_AREA)
        else:
            display_small, gray_small = display, gray

        cv2.imshow(win_main, display_small)
        cv2.imshow(win_gray, gray_small)

        # Once we know the actual on-screen size, position the two windows
        # side by side instead of stacked on top of each other.
        if not windows_positioned:
            cv2.resizeWindow(win_main, display_small.shape[1], display_small.shape[0])
            cv2.resizeWindow(win_gray, gray_small.shape[1], gray_small.shape[0])
            cv2.moveWindow(win_main, 20, 20)
            cv2.moveWindow(win_gray, 20 + display_small.shape[1] + 20, 20)
            windows_positioned = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and found:
            objp = build_object_points(pattern_size, square_size=1.0)  # scaled later
            object_points.append(objp)
            image_points.append(corners)
            print(f"  Captured frame #{len(image_points)}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return object_points, image_points, image_size


def run_folder_capture(images_dir, pattern_size):
    object_points = []
    image_points = []
    image_size = None

    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.tiff"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Found {len(paths)} images in {images_dir}")

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  Skipping unreadable file: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = find_corners(gray, pattern_size, use_sb=True)
        if found:
            objp = build_object_points(pattern_size, square_size=1.0)  # scaled later
            object_points.append(objp)
            image_points.append(corners)
            print(f"  [OK]   {os.path.basename(path)}")
        else:
            print(f"  [FAIL] {os.path.basename(path)} — no chessboard found")

    return object_points, image_points, image_size


def main():
    parser = argparse.ArgumentParser(description="RGB camera calibration using a chessboard.")
    parser.add_argument("--rows", type=int, required=True,
                         help="Number of INNER corners per column (chessboard rows).")
    parser.add_argument("--cols", type=int, required=True,
                         help="Number of INNER corners per row (chessboard columns).")
    parser.add_argument("--square-size", type=float, required=True,
                         help="Physical size of one chessboard square (e.g. meters).")
    parser.add_argument("--live", action="store_true",
                         help="Capture frames live from a camera device.")
    parser.add_argument("--device", type=int, default=0,
                         help="Camera device index for live capture (default: 0).")
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Folder of pre-captured images to calibrate from instead of live capture.")
    parser.add_argument("--min-frames", type=int, default=10,
                         help="Minimum number of good frames recommended before calibrating (default: 10).")
    parser.add_argument("--display-scale", type=float, default=1.0,
                         help="Scale factor for the live preview windows, e.g. 0.5 for half size (default: 1.0).")
    parser.add_argument("--out", type=str, default="camera_calibration.npz",
                         help="Output file path for calibration results (.npz).")
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)  # OpenCV expects (cols, rows) i.e. (points_per_row, points_per_col)

    pixel_format = "MJPG"  # change to "UYVY", "GREY", "RGBP", or "BGR3" as needed

    width=1920
    height=1080

    subprocess.run([
        "v4l2-ctl",
        "--device=/dev/video4",
        f"--set-fmt-video=width={width},height={height},pixelformat={pixel_format}"
    ], check=False)

    if args.live and args.images_dir:
        parser.error("Choose either --live or --images-dir, not both.")
    if not args.live and not args.images_dir:
        parser.error("You must specify either --live or --images-dir.")

    if args.live:
        fourcc=cv2.VideoWriter.fourcc(*"MJPG")
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        if not cap.isOpened():
            print(f"ERROR: could not open camera device {args.device}")
            return
        object_points, image_points, image_size = run_live_capture(
            cap, pattern_size, args.min_frames, args.display_scale
        )
        cap.release()
    else:
        object_points, image_points, image_size = run_folder_capture(
            args.images_dir, pattern_size
        )

    if len(image_points) < 4:
        print(f"\nOnly {len(image_points)} good frames found — need at least a handful "
              "(ideally >= 10-15) covering different positions/angles/tilts. Aborting.")
        return
    if len(image_points) < args.min_frames:
        print(f"\nWarning: only {len(image_points)} good frames (recommended >= {args.min_frames}). "
              "Calibration will run, but consider capturing more for a robust result.")

    # Scale object points to real square size now that we know it
    object_points = [pts * args.square_size for pts in object_points]

    result = calibrate(object_points, image_points, image_size)
    print_summary(result)
    save_calibration(result, image_size, pattern_size, args.square_size, args.out)


if __name__ == "__main__":
    main()