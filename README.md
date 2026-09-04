# Multi-Modal Camera + IMU Perception Platform

## Project Description

This project explores multi-modal passive sensing for computer vision — RGB and thermal augmented with IMU data. The broader goal was to build a hardware prototype (RGB stereo + thermal cameras + IMU on a Jetson Orin Nano, running ROS 2 Jazzy) and use it to investigate cross-modal fusion and scene reconstruction.

This repo includes scripts and resources relevant to the hardware and platform being used for the following: **IMU characterization, camera/IMU calibration, RGB–thermal alignment (both for images as well as for reconstructed 3D pointclouds), and 3D reconstruction from stereo**.

## Repository Structure

### `allan_variance_imu/`
Scripts for characterizing IMU noise and drift. Includes recording raw IMU data as a rosbag, running an Allan variance analysis to extract noise/bias parameters, and plotting the resulting curves and drift behavior.

### `camera_calib_utils/`
General-purpose camera calibration utilities and results. Covers RGB and stereo-thermal intrinsic/extrinsic calibration, auto-calibration helpers, and stored extrinsic results for the RGB, thermal, and right-thermal camera configurations. This includes the .yaml files needed as input for Discocal with the calibration target parameters and camera intrinsics, as well as the resultant computed extrinsics for the RGB and thermal stereo pairs.

### `data_utils/`
Shared helpers for capturing and processing sensor data from ROS 2 bags — camera and stereo capture utilities, IMU data extraction, a script to convert the .yaml file from ROS2 Jazzy to ROS 2 Humble readable, and general rosbag extraction script.

### `imu_camera_kalibr/`
IMU–camera spatiotemporal calibration using [Kalibr](#citations). Contains calibration target definitions (AprilGrid, CircleGrid), IMU/camera config YAMLs, and the resulting calibration reports, chain files, and target PDFs for both the RGB and thermal cameras.

### `platform_step_files/`
CAD models (STEP format) for the physical camera/IMU mounting platform — the base assembly (`Exodia_basev3.step`), the DiscoCal calibration target mount (`discocal_target.step`), and the RGB camera flange (`rgb_flange.step`).

### `reconstruction_scripts/`
3D scene reconstruction from stereo RGB imagery using [COLMAP](#citations).

### `rgb-thermal-alignment/`
Scripts for aligning and registering RGB and thermal imagery, including a variant built on the real-time [Fast-FoundationStereo](#citations) model (`*-ffs.py`) alongside a standard alignment pipeline, adding thermal information to reconstructed RGB pointclouds, and tools for toggling between/visualizing aligned views and point clouds.

## Citations

The underlying tools were used for this project:

**Kalibr** (IMU–camera calibration)
```bibtex
@inproceedings{furgale2013kalibr,
  author={Paul Furgale and Joern Rehder and Roland Siegwart},
  title={Unified Temporal and Spatial Calibration for Multi-Sensor Systems},
  booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2013}
}
```

**DiscoCal** (circular-pattern camera calibration)
```bibtex
@article{song2024discocal,
  title={Unbiased Estimator for Distorted Conics in Camera Calibration},
  author={Song, Chaehyeon and Shin, Jaeho and Jeon, Myung-Hwan and Lim, Jongwoo and Kim, Ayoung},
  journal={arXiv preprint arXiv:2403.04583},
  year={2024}
}
```

**Fast-FoundationStereo** (real-time zero-shot stereo matching)
```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```

**COLMAP** (structure-from-motion / multi-view stereo reconstruction)
```bibtex
@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
@inproceedings{schoenberger2016mvs,
  author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
  title={Pixelwise View Selection for Unstructured Multi-View Stereo},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016}
}
```
