# Period3D

PERIOD stands for Pose Estimation and Recognition for Instance-level Object Detection in 3D

TODO: Write a project description

## Current Functionalities

* Captures synchronized RGB-D frames from Intel® RealSense™ F200 Camera in real-time
* Saves RGB-D frames and intrinsics to disk (depth aligned to color) 
 * Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
 * Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG)

## Dependencies

* Intel® RealSense™ F200 Camera
* [librealsense](https://github.com/IntelRealSense/librealsense)
* png++ `sudo apt-get install libpng++-dev`
* [marvin](https://github.com/PrincetonVision/marvin)

## Compilation

`./compile.sh`

## Usage

Capture RGB-D frames with `./capture`

TODO: Write usage instructions