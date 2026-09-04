import os
import subprocess
import cv2
import numpy as np

pixel_format1 = "Y16"
pixel_format2 = "UYVY"

width1 = 160
height1 = 120

width2 = 512
height2= 192

device1 = "/dev/video4"
device2 = "/dev/video6"
camera1 = 'rgb2'
camera2 = 'brick1'

subprocess.run([
    "v4l2-ctl",
    f"--device={device1}",
    f"--set-fmt-video=width={width1},height={height1},pixelformat={pixel_format1}"
], check=False)

subprocess.run([
    "v4l2-ctl",
    f"--device={device2}",
    f"--set-fmt-video=width={width2},height={height2},pixelformat={pixel_format2}"
], check=False)

fourcc_map = {
    "Y16": cv2.VideoWriter.fourcc('Y', '1', '6', ' '),
    "MJPG": cv2.VideoWriter.fourcc(*"MJPG"),
    "UYVY": cv2.VideoWriter.fourcc('U','Y','V','Y'),
    "YUYV": cv2.VideoWriter.fourcc('Y','U','Y','V'),
    "GREY": cv2.VideoWriter.fourcc('G','R','E','Y'),
    "RGBP": cv2.VideoWriter.fourcc('R','G','B','P'),
    "BGR3": cv2.VideoWriter.fourcc('B','G','R','3'),
}

fourcc1 = fourcc_map[pixel_format1]
fourcc2 = fourcc_map[pixel_format2]

cap1 = cv2.VideoCapture(device1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(device2, cv2.CAP_V4L2)

if not cap1.isOpened():
    print("Failed to open camera 1")
    exit()

if not cap2.isOpened():
    print("Failed to open camera 2")
    exit()


cap1.set(cv2.CAP_PROP_FOURCC, fourcc1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width1)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height1)
cap1.set(cv2.CAP_PROP_CONVERT_RGB, 0)

cap2.set(cv2.CAP_PROP_FOURCC, fourcc2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width2)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height2)
cap2.set(cv2.CAP_PROP_CONVERT_RGB, 0)

print("Camera 1:", cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera 2:", cap2.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

folder1 = "/home/aayushi/exodia/rgb-thermal-calib/thermal-right/rgb"
folder2 = "/home/aayushi/exodia/rgb-thermal-calib/thermal-right/thermal"

os.makedirs(folder1, exist_ok=True)
os.makedirs(folder2, exist_ok=True)

img_count = 0

def build_display_frame(frame, fmt_name, camera, height, width):
    if frame is None:
        return None, None

    raw_frame = frame

    if fmt_name == "Y16":
        gray = raw_frame
        gray16 = gray.astype(np.uint16, copy=False) if gray.dtype != np.uint16 else gray
        display_frame = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        display_frame = cv2.applyColorMap(display_frame, cv2.COLORMAP_INFERNO)
        return gray16, display_frame

    if fmt_name == "GREY":
        gray = raw_frame
        display_frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return gray, display_frame

    if fmt_name == "UYVY":
        display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_UYVY)
        if camera=='brick1':
            display_frame=display_frame[:,256:]
        return raw_frame, display_frame
    
    if fmt_name == "YUYV":
        display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_YUYV)
        display_frame= cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)  # collapse to 1 channel
        #display_frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)  # now apply colormap
        #What exactly is going on with brick 1?
        if camera=='brick1':
            display_frame=display_frame[:,256:]+display_frame[:,256:]
            display_frame = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return raw_frame, display_frame

    if fmt_name == "RGBP":
        raw_8bit = np.frombuffer(raw_frame, dtype=np.uint8)
        frame_rgb565 = raw_8bit.reshape((height, width, 2))
        bgr_image = cv2.cvtColor(frame_rgb565, cv2.COLOR_BGR5652BGR)
        return frame_rgb565, bgr_image

    if fmt_name == "BGR3":
        display_frame = raw_frame.astype(np.uint8, copy=False)
        return raw_frame, display_frame

    if fmt_name == "MJPG":
        display_frame = raw_frame
        return raw_frame, display_frame

    return raw_frame, raw_frame

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    raw1, disp1 = build_display_frame(frame1, pixel_format1, camera1, height1, width1)
    raw2, disp2 = build_display_frame(frame2, pixel_format2, camera2, height2, width2)

    if disp1 is None or disp2 is None:
        continue

    if disp1.shape!=disp2.shape:
        rgb_h, rgb_w = disp1.shape[:2]
        th_h, th_w = disp2.shape[:2]

        # Scale thermal image to RGB height
        scale = rgb_h / th_h
        new_w = int(round(th_w * scale))

        thermal_resized = cv2.resize(disp2, (new_w, rgb_h), interpolation=cv2.INTER_NEAREST)

        # Pad on the right if needed
        if new_w < rgb_w:
            pad = np.zeros((rgb_h, rgb_w - new_w, 3), dtype=thermal_resized.dtype)
            thermal_display = np.hstack((thermal_resized, pad))
        else:
            thermal_display = thermal_resized[:, :rgb_w]

        display = np.hstack((disp1, thermal_display))

        display_scale = min(
            1600 / display.shape[1],
            900 / display.shape[0],
            1.0
        )

        display_small = cv2.resize(display, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

        cv2.imshow("Stereo Cameras", display_small)

    else:
        display = np.hstack((disp1, disp2))
        # Resize only for display
        display_scale = 2  # change this value as needed
        display_small = cv2.resize(display, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

        cv2.imshow("Stereo Cameras", display_small)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:

            _, disp1 = build_display_frame(frame1, pixel_format1, camera1, height1, width1)
            _, disp2 = build_display_frame(frame2, pixel_format2, camera2, height2, width2)

            path1 = os.path.join(folder1, f"cap_{img_count:02d}.png")
            path2 = os.path.join(folder2, f"cap_{img_count:02d}.png")

            cv2.imwrite(path1, disp1)
            cv2.imwrite(path2, disp2)

            print(f"Saved {path1}")
            print(f"Saved {path2}")

            img_count += 1

    elif key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()