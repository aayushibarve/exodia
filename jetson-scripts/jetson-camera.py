# Configure the camera to use one of the Lepton pixel formats before opening it with OpenCV.
import os
import subprocess
import cv2
import numpy as np
import sys
import threading
import termios
import tty

global capture_requested 
capture_requested = False
global stopcode 
stopcode = False

def keyboard_listener():
    global capture_requested

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while True:
            key = sys.stdin.read(1)

            if key == 'c':
                global capture_requested
                capture_requested = True
            elif key == 'q':
                global stopcode
                stopcode=True
                break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
threading.Thread(target=keyboard_listener, daemon=True).start()

pixel_format = "UYVY"  # change to "UYVY", "GREY", "RGBP", or "BGR3" as needed

width=512
height=192
camera='brick1'

subprocess.run([
    "v4l2-ctl",
    "--device=/dev/video0",
    f"--set-fmt-video=width={width},height={height},pixelformat={pixel_format}"
], check=False)

device = "/dev/video0"



# Query basic info
info = subprocess.run(["v4l2-ctl", "--device", device, "--all"],
                      capture_output=True, text=True)
#print(info.stdout)

fourcc_map = {
    "Y16": cv2.VideoWriter.fourcc('Y', '1', '6', ' '),
    "UYVY": cv2.VideoWriter.fourcc('U', 'Y', 'V', 'Y'),
    "YUYV": cv2.VideoWriter.fourcc('Y','U','Y','V'),
    "GREY": cv2.VideoWriter.fourcc('G', 'R', 'E', 'Y'),
    "RGBP": cv2.VideoWriter.fourcc('R', 'G', 'B', 'P'),
    "BGR3": cv2.VideoWriter.fourcc('B', 'G', 'R', '3'),
}
fourcc = fourcc_map[pixel_format]

print('Format set')
cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
print(cap.getBackendName())
if not cap.isOpened():
    print("ERROR: Camera did not open")
    exit()
print('Starting capture')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

print("Negotiated:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("CAP_PROP_FOURCC:", cap.get(cv2.CAP_PROP_FOURCC))
print("CAP_PROP_CONVERT_RGB:", cap.get(cv2.CAP_PROP_CONVERT_RGB))

folder = "/home/disal/capture/brick1"
os.makedirs(folder, exist_ok=True)

frame_buf = []
img_count = 0


def build_display_frame(frame, fmt_name):
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
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)  # collapse to 1 channel
        display_frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)  # now apply colormap
        return raw_frame, display_frame

    if fmt_name == "RGBP":
        raw_8bit = np.frombuffer(raw_frame, dtype=np.uint8)
        frame_rgb565 = raw_8bit.reshape((height, width, 2))
        bgr_image = cv2.cvtColor(frame_rgb565, cv2.COLOR_BGR5652BGR)
        return frame_rgb565, bgr_image

    if fmt_name == "BGR3":
        display_frame = raw_frame.astype(np.uint8, copy=False)
        return raw_frame, display_frame


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    raw_frame, display_frame = build_display_frame(frame, pixel_format)
    if raw_frame is None or display_frame is None:
        continue

    #print(f"Frame shape={frame.shape}, dtype={frame.dtype}, processed_shape={display_frame.shape}, processed_dtype={display_frame.dtype}")
    
    # Write video to memory
    frame_buf.append(raw_frame)

    key = cv2.waitKey(1) & 0xFF

    if capture_requested:
        capture_requested = False

        proc_path = os.path.join(folder, f"cap_{img_count:02d}.tiff")
        success = cv2.imwrite(proc_path, display_frame)

        if success:
            print(f"Saved processed: {proc_path}")
        else:
            print("Failed to save image")

        img_count += 1
    if stopcode:
        break

cap.release()
