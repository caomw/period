# Period3D

PERIOD stands for Pose Estimation and Recognition for Instance-level Object Detection in 3D

TODO: Write a project description

## Current Features

* Captures synchronized RGB-D frames from Intel® RealSense™ F200 Camera in real-time
* Capture frames in single-camera mode or multi-camera mode (with anti-interference)
* Saves RGB-D frames and intrinsics to disk (depth aligned to color) 
 * Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
 * Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG)
* Scripts for flushing USB ports if necessary
* Performs SIFT-based frame-to-frame alignment for 3D reconstruction and camera pose estimation
 * Pose: frame-XXXXXX.pose.txt (camera-to-world, 4x4 matrix in homogeneous coordinates).
* CPU implmentation of Kinect Fusion to compute projective TSDF (Truncated Signed Distance Fields) for 3D volumetric representation
* Code for training data collection and detection using Marvin with PERIOD's deep learning network

## Dependencies

* Intel® RealSense™ F200 Camera
* [librealsense](https://github.com/IntelRealSense/librealsense)
* png++ `sudo apt-get install libpng++-dev`
* Matlab
* [marvin](https://github.com/PrincetonVision/marvin)

## Compilation

`./compile.sh`

## Usage

* Capture RGB-D frames with `./capture`
* Run `/tools/matlab/recon/demo.m` for camera pose estimation
* Run `./kinfu` for Kinect Fusion and to create a point cloud from the fused TSDF volume

TODO: Write usage instructions