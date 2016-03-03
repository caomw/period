# Period3D

PERIOD stands for Pose Estimation and Recognition for Instance-level Object Detection in 3D

## Features

* Standalone C++ object detection algorithm using Marvin with Period’s deep learning network
 * Contains a pretrained model for glue bottle object detection

## Hardware Dependencies

* Intel® RealSense™ F200 Camera
* NVIDIA GPU for CUDA and cuDNN

## Installation Instructions
1. Install [librealsense](https://github.com/IntelRealSense/librealsense) (instructions can be found [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md))
 * Note: install with the Video4Linux backend
2. Install PNG++, a library for PNG I/O
 * `sudo apt-get install libpng++-dev`
3. Install the dependencies required by [Marvin](https://github.com/PrincetonVision/marvin) (CUDA 7.5 and cuDNN 4rc)
 * Note: instead of installing cuDNN 4 as suggested by the current Marvin README, install cuDNN 4rc (older version of the installation instructions for Marvin can be found [here](https://github.com/PrincetonVision/marvin/blob/f1a628efb34e858c8dddb66ac928dde2341cccd1/README.md))
4. Compile Marvin
 * Navigate to `/tools/marvin` and run `./compile.sh`
5. If you have not yet done so, install OpenCV (we are using OpenCV 2.4.11) 
6. Download Period's training data for Marvin [here](https://drive.google.com/folderview?id=0B4mCa-2YGnp7bE4yN1BwWVpjUWM&usp=sharing)
 * Place `data2DTrain.tensor`, `data3DTrain.tensor`, `labelsTrain.tensor` in folder `data/tensors` (if the folder doesn't exist, create it)
7. Download Period's pre-trained weights for Marvin [here](https://drive.google.com/folderview?id=0B4mCa-2YGnp7bE4yN1BwWVpjUWM&usp=sharing)
 * Place `PeriodNet.marvin` in folder `tools/marvin`

## Compilation

`./compile.sh`

## Standalone Usage

* Run `./standalone`
 * Note: on running the executable, a window will pop up and stream data from the sensor
 * Press the spacebar key to run object detection
 * Note: detection dialogue will display itself on the command prompt
 * If an object is successfully detected, a second window will pop up displaying the cube bounding box of the object
 * Detection dialogue will return information about the object's 3D location with respect to the camera
 * Press the spacebar key again to continue streaming data from the sensor

## Other Features

* Stream synchronized RGB-D frames from Intel® RealSense™ F200 Camera in real-time
* Capture frames in single-camera mode or multi-camera mode (with anti-interference)
* Saves RGB-D frames and intrinsics to disk (depth aligned to color) 
 * Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
 * Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG)
* Scripts for flushing USB ports if necessary
* Performs SIFT-based frame-to-frame alignment for 3D reconstruction and camera pose estimation
 * Pose: frame-XXXXXX.pose.txt (camera-to-world, 4x4 matrix in homogeneous coordinates).
* CPU implmentation of Kinect Fusion to compute projective TSDF (Truncated Signed Distance Fields) for 3D volumetric representation

## Additional Usage Instructions for Other Features

* Stream RGB-D frames with `./multicam`
 * Press spacebar key to save frames
* Capture and save RGB-D frames with `./capture`
 * RGB-D frames and intrinsics will be saved to the `data` folder
* Run Kinect fusion and generate point cloud of fused TSDF volume `./kinfu`
* Run camera pose estimation with `/tools/matlab/recon/demo.m`