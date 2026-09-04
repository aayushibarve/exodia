# ROS 2 Sensor Packages and scripts for the Jetson

ROS 2 (Jazzy) packages and scripts for running the camera + IMU hardware stack on the Jetson Orin Nano

## Packages

- **`rgb_camera/`** — ROS 2 driver/publisher package for the RGB camera(s).
- **`thermal_camera/`** — ROS 2 driver/publisher package for the thermal camera(s).
- **`wit_ros2_imu/`** — ROS 2 driver package for the IMU, taken from [this Google Drive folder](https://drive.google.com/drive/folders/1AWsB-WSwWK3zLfkPwECs9csat4eT4c1l).
- **`listener_test/`** — Minimal test node for subscribing to and verifying sensor topics during bring-up.

## Scripts

- **`jetson-camera.py`** — Runs the camera pipeline on the Jetson.
- **`decode_uyvy.py`** — Decodes raw UYVY-format camera frames.
- **`data_capture.py`** — Captures/records sensor data (topics) to disk.
