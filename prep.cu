#include "util/util.hpp"
#include "kinfu.hpp"
#include <opencv2/opencv.hpp>

////////////////////////////////////////////////////////////////////////////////

void show_object_pose(float* K, float* object_pose, cv::Mat& display_frame) {

  // Compute center of ground truth object in 3D camera coordinates
  float object_center_display_3D[3] = {0};
  for (int i = 0; i < 3; i++)
    object_center_display_3D[i] = object_pose[i * 4 + 3];

  // Compute axis endpoints of ground truth object pose in 3D camera coordinates
  float object_pose_display_3D[18] = {0};
  for (int i = 0; i < 3; i++) {
    object_pose_display_3D[0 * 6 + i * 2 + 0] = object_pose[0 * 4 + 3] - object_pose[0 * 4 + i] * 0.1f;
    object_pose_display_3D[0 * 6 + i * 2 + 1] = object_pose[0 * 4 + 3] + object_pose[0 * 4 + i] * 0.1f;
    object_pose_display_3D[1 * 6 + i * 2 + 0] = object_pose[1 * 4 + 3] - object_pose[1 * 4 + i] * 0.1f;
    object_pose_display_3D[1 * 6 + i * 2 + 1] = object_pose[1 * 4 + 3] + object_pose[1 * 4 + i] * 0.1f;
    object_pose_display_3D[2 * 6 + i * 2 + 0] = object_pose[2 * 4 + 3] - object_pose[2 * 4 + i] * 0.1f;
    object_pose_display_3D[2 * 6 + i * 2 + 1] = object_pose[2 * 4 + 3] + object_pose[2 * 4 + i] * 0.1f;
  }

  // Project endpoints of ground truth object pose axis from 3D to 2D
  float object_pose_display_2D[12] = {0};
  for (int i = 0; i < 6; i++) {
    object_pose_display_2D[0 * 6 + i] = (object_pose_display_3D[0 * 6 + i] * K[0]) / (object_pose_display_3D[2 * 6 + i]) + K[2];
    object_pose_display_2D[1 * 6 + i] = (object_pose_display_3D[1 * 6 + i] * K[4]) / (object_pose_display_3D[2 * 6 + i]) + K[5];
  }
  // for (int i = 0; i < 12; i++)
  //   std::cout << object_pose_display_2D[i] << std::endl;

  // Project center of ground truth object from 3D to 2D
  float object_center_display_2D[2] = {0};
  object_center_display_2D[0] = (object_center_display_3D[0] * K[0]) / (object_center_display_3D[2]) + K[2];
  object_center_display_2D[1] = (object_center_display_3D[1] * K[4]) / (object_center_display_3D[2]) + K[5];
  // for (int i = 0; i < 12; i++)
  //   std::cout << object_pose_display_2D[i] << std::endl;

  // Display ground truth object pose
  cv::line(display_frame, cv::Point(object_pose_display_2D[0], object_pose_display_2D[6]), cv::Point(object_pose_display_2D[1], object_pose_display_2D[7]), cv::Scalar(0, 0, 255), 3);
  cv::line(display_frame, cv::Point(object_pose_display_2D[2], object_pose_display_2D[8]), cv::Point(object_pose_display_2D[3], object_pose_display_2D[9]), cv::Scalar(0, 255, 0), 3);
  cv::line(display_frame, cv::Point(object_pose_display_2D[4], object_pose_display_2D[10]), cv::Point(object_pose_display_2D[5], object_pose_display_2D[11]), cv::Scalar(255, 0, 0), 3);
  cv::circle(display_frame, cv::Point(object_center_display_2D[0], object_center_display_2D[1]), 6, cv::Scalar(0, 255, 255), -1);
  cv::namedWindow("Object Pose", CV_WINDOW_AUTOSIZE);
  cv::imshow("Object Pose", display_frame);
}

////////////////////////////////////////////////////////////////////////////////

__global__
void gen_hypothesis_labels(int num_hypothesis, unsigned short* tmp_hypothesis_locations, char* tmp_hypothesis_labels, unsigned short* tmp_hypothesis_crop_2D, float* tmp_object_center_cam, float* tmp_K, float tmp_vox_unit, int* tmp_vox_size, float* tmp_vox_range_cam, float* tmp_vox_tsdf) {

  // Check kernel index
  int hypothesis_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (hypothesis_idx >= num_hypothesis)
    return;

  float tsdf_surface_threshold = 0.2f;
  int cube_dim = 30;

  // Fetch hypothesis location
  int x = (int)tmp_hypothesis_locations[0 * num_hypothesis + hypothesis_idx];
  int y = (int)tmp_hypothesis_locations[1 * num_hypothesis + hypothesis_idx];
  int z = (int)tmp_hypothesis_locations[2 * num_hypothesis + hypothesis_idx];

  // Check cube occupancy
  int cube_occ = 0;
  for (int i = -15; i < 15; i++)
    for (int j = -15; j < 15; j++)
      for (int k = -15; k < 15; k++) {
        int vox_idx = (z + k) * tmp_vox_size[0] * tmp_vox_size[1] + (y + j) * tmp_vox_size[0] + (x + i);
        if (tmp_vox_tsdf[vox_idx] < tsdf_surface_threshold)
          cube_occ++;
      }

  // Skip near empty cubes
  if (cube_occ < cube_dim * cube_dim / 2)
    return;

  // Convert cube location from grid to camera coordinates
  float x_cam = ((float)x + 1) * tmp_vox_unit + tmp_vox_range_cam[0 * 2 + 0];
  float y_cam = ((float)y + 1) * tmp_vox_unit + tmp_vox_range_cam[1 * 2 + 0];
  float z_cam = ((float)z + 1) * tmp_vox_unit + tmp_vox_range_cam[2 * 2 + 0];

  // If cube 2D projection is not in image bounds, cube is invalid
  float cube_rad = ((float) cube_dim) * tmp_vox_unit / 2;
  float cube_front[12] = {(x_cam + cube_rad), (x_cam + cube_rad), (x_cam - cube_rad), (x_cam - cube_rad),
                          (y_cam + cube_rad), (y_cam - cube_rad), (y_cam - cube_rad), (y_cam + cube_rad),
                          (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad)
                         };
  float cube_front_2D[8] = {};
  for (int i = 0; i < 4; i++) {
    cube_front_2D[0 * 4 + i] = cube_front[0 * 4 + i] * tmp_K[0] / cube_front[2 * 4 + i] + tmp_K[2];
    cube_front_2D[1 * 4 + i] = cube_front[1 * 4 + i] * tmp_K[4] / cube_front[2 * 4 + i] + tmp_K[5];
  }
  for (int i = 0; i < 8; i++)
    cube_front_2D[i] = roundf(cube_front_2D[i]);
  if (fmin(fmin(cube_front_2D[0], cube_front_2D[1]), fmin(cube_front_2D[2], cube_front_2D[3])) < 0 ||
      fmax(fmax(cube_front_2D[0], cube_front_2D[1]), fmax(cube_front_2D[2], cube_front_2D[3])) >= 640 ||
      fmin(fmin(cube_front_2D[4], cube_front_2D[5]), fmin(cube_front_2D[6], cube_front_2D[7])) < 0 ||
      fmax(fmax(cube_front_2D[4], cube_front_2D[5]), fmax(cube_front_2D[6], cube_front_2D[7])) >= 480)
    return;

  // Check distance between center of cube to center of object bbox
  float obj_dist = sqrtf((tmp_object_center_cam[0] - x_cam) * (tmp_object_center_cam[0] - x_cam) +
                         (tmp_object_center_cam[1] - y_cam) * (tmp_object_center_cam[1] - y_cam) +
                         (tmp_object_center_cam[2] - z_cam) * (tmp_object_center_cam[2] - z_cam));

  // Save label (positive case if dist to ground truth object center < some threshold)
  if (obj_dist < 0.01f)
    tmp_hypothesis_labels[hypothesis_idx] = (char)1;
  else
    tmp_hypothesis_labels[hypothesis_idx] = (char)2;

  // Save 2D patch of cube's 2D project to image
  tmp_hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx] = (unsigned short)roundf(cube_front_2D[2]);
  tmp_hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx] = (unsigned short)roundf(cube_front_2D[6]);
  tmp_hypothesis_crop_2D[2 * num_hypothesis + hypothesis_idx] = (unsigned short)roundf(cube_front_2D[1] - cube_front_2D[2]);
  tmp_hypothesis_crop_2D[3 * num_hypothesis + hypothesis_idx] = (unsigned short)roundf(cube_front_2D[4] - cube_front_2D[6]);
}

////////////////////////////////////////////////////////////////////////////////

void generate_train_labels(const std::string &sequence_directory) {

  std::vector<std::string> frame_names;
  get_files_in_directory(sequence_directory, frame_names, ".color.png");
  std::sort(frame_names.begin(), frame_names.end());

  // Process all RGB-D frames in directory
  for (int i = 0; i < frame_names.size(); i++) {
    std::string curr_frame_name = frame_names[i];
    curr_frame_name = curr_frame_name.substr(0, curr_frame_name.length() - 10);
    std::cout << "Preparing Training Frame: " << sequence_directory << "/" << curr_frame_name << std::endl;

    // Load intrinsics (3x3 matrix)
    std::string intrinsic_filename = sequence_directory + "/intrinsics.K.txt";
    std::vector<float> K_vec = load_matrix_from_file(intrinsic_filename, 3, 3);
    // float K[9];
    // for (int i = 0; i < 9; i++)
    //   K[i] = K_vec[i];
    float * K = &K_vec[0];
    // for (int i = 0; i < 9; i++)
    //   std::cout << K[i] << std::endl;

    // Load RGB-D frame
    std::string curr_frame_color_filename = sequence_directory + "/" + curr_frame_name + ".color.png";
    cv::Mat curr_frame_color = cv::imread(curr_frame_color_filename.c_str(), 1);
    std::string curr_frame_depth_filename = sequence_directory + "/" + curr_frame_name + ".depth.png";
    cv::Mat curr_frame_depth = cv::imread(curr_frame_depth_filename.c_str(), CV_LOAD_IMAGE_UNCHANGED);

    // Read ground truth object pose from file
    std::string object_pose_filename = sequence_directory + "/object.pose.txt";
    std::vector<float> object_pose_raw = load_matrix_from_file(object_pose_filename, 4, 4);
    float * object_pose_arr = &object_pose_raw[0];
    // for (int i = 0; i < 12; i++)
    //   std::cout << object_pose[i] << std::endl;

    // Compute ground truth object pose w.r.t. current camera pose
    std::string curr_cam_pose_filename = sequence_directory + "/" + curr_frame_name + ".pose.txt";
    std::vector<float> curr_cam_pose_raw = load_matrix_from_file(curr_cam_pose_filename, 4, 4);
    float * curr_cam_pose_arr = &curr_cam_pose_raw[0];
    // for (int i = 0; i < 16; i++)
    //   std::cout << curr_cam_pose_arr[i] << std::endl;
    float curr_cam_pose_inv[16] = {0};
    invert_matrix(curr_cam_pose_arr, curr_cam_pose_inv);
    // for (int i = 0; i < 16; i++)
    //   std::cout << curr_cam_pose_inv[i] << std::endl;
    float object_pose[16] = {0};
    multiply_matrix(curr_cam_pose_inv, object_pose_arr, object_pose);
    // for (int i = 0; i < 4; i++) {
    //   for (int j = 0; j < 4; j++)
    //     std::cout << object_pose[i * 4 + j] << " ";
    //   std::cout << std::endl;
    // }

    // Display ground truth object pose
    show_object_pose(K, object_pose, curr_frame_color);
    cv::waitKey(0);

    // Compute center of ground truth object in 3D camera coordinates
    float object_center_cam[3] = {0};
    for (int i = 0; i < 3; i++)
      object_center_cam[i] = object_pose[i * 4 + 3];

    // Convert pose from rotation matrix to axis/angle (radians) representation (x, y, z, theta)
    float object_pose_axis[3] = {0};
    float object_pose_angle = std::acos(0.5f * (object_pose[0] + object_pose[5] + object_pose[10] - 1));
    object_pose_axis[0] = (object_pose[9] - object_pose[6]) / (2 * std::sin(object_pose_angle));
    object_pose_axis[1] = (object_pose[2] - object_pose[8]) / (2 * std::sin(object_pose_angle));
    object_pose_axis[2] = (object_pose[4] - object_pose[1]) / (2 * std::sin(object_pose_angle));
    // for (int i = 0; i < 3; i++)
    //   std::cout << object_pose_axis[i] << std::endl;

    // Convert axis/angle to pose
    float object_pose_rotation[9] = {0};
    object_pose_rotation[0 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[0] + std::cos(object_pose_angle);
    object_pose_rotation[0 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[1] - object_pose_axis[2] * std::sin(object_pose_angle);
    object_pose_rotation[0 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[2] + object_pose_axis[1] * std::sin(object_pose_angle);
    object_pose_rotation[1 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[0] + object_pose_axis[2] * std::sin(object_pose_angle);
    object_pose_rotation[1 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[1] + std::cos(object_pose_angle);
    object_pose_rotation[1 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[2] - object_pose_axis[0] * std::sin(object_pose_angle);
    object_pose_rotation[2 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[0] - object_pose_axis[1] *  std::sin(object_pose_angle);
    object_pose_rotation[2 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[1] + object_pose_axis[0] * std::sin(object_pose_angle);
    object_pose_rotation[2 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[2] + std::cos(object_pose_angle);
    // for (int i = 0; i < 3; i++) {
    //   for (int j = 0; j < 3; j++)
    //     std::cout << object_pose_rotation[i * 3 + j] << " ";
    //   std::cout << std::endl;
    // }

    // Bin axis into one of 42 bins
    float axis_sphere_bin[42 * 3] = { -0.85065, -1, -0.85065, -0.80902, -0.80902, -0.80902, -0.80902, -0.52573, -0.52573, -0.5, -0.5, -0.5, -0.5, -0.30902, -0.30902, -0.30902, -0.30902, 0, 0, 0, 0, 0, 0, 0, 0, 0.30902, 0.30902, 0.30902, 0.30902, 0.5, 0.5, 0.5, 0.5, 0.52573, 0.52573, 0.80902, 0.80902, 0.80902, 0.80902, 0.85065, 1, 0.85065,
                                      0, 0, 0, -0.5, -0.5, 0.5, 0.5, -0.85065, 0.85065, -0.30902, -0.30902, 0.30902, 0.30902, -0.80902, -0.80902, 0.80902, 0.80902, -1, -0.52573, -0.52573, 0, 0, 0.52573, 0.52573, 1, -0.80902, -0.80902, 0.80902, 0.80902, -0.30902, -0.30902, 0.30902, 0.30902, -0.85065, 0.85065, -0.5, -0.5, 0.5, 0.5, 0, 0, 0,
                                      -0.52573, 0, 0.52573, -0.30902, 0.30902, -0.30902, 0.30902, 0, 0, -0.80902, 0.80902, -0.80902, 0.80902, -0.5, 0.5, -0.5, 0.5, 0, -0.85065, 0.85065, -1, 1, -0.85065, 0.85065, 0, -0.5, 0.5, -0.5, 0.5, -0.80902, 0.80902, -0.80902, 0.80902, 0, 0, -0.30902, 0.30902, -0.30902, 0.30902, -0.52573, 0, 0.52573
                                    };
    int closest_axis_bin = 0;
    float closest_axis_dist = 100;
    for (int i = 0; i < 42; i++) {
      float curr_axis_dist = std::sqrt((axis_sphere_bin[0 * 42 + i] - object_pose_axis[0]) * (axis_sphere_bin[0 * 42 + i] - object_pose_axis[0]) +
                                       (axis_sphere_bin[1 * 42 + i] - object_pose_axis[1]) * (axis_sphere_bin[1 * 42 + i] - object_pose_axis[1]) +
                                       (axis_sphere_bin[2 * 42 + i] - object_pose_axis[2]) * (axis_sphere_bin[2 * 42 + i] - object_pose_axis[2]));
      if (curr_axis_dist < closest_axis_dist) {
        closest_axis_dist = curr_axis_dist;
        closest_axis_bin = i;
      }
    }
    // std::cout << closest_axis_bin << std::endl;
    // std::cout << closest_axis_dist << std::endl;

    // Bin angle into one of 18 bins (10 degrees)
    float closest_angle_bin = floor(object_pose_angle / (3.14159265 / 18));
    if (closest_angle_bin > 17 || closest_axis_bin > 41) {
      std::cout << "AXIS/ANGLE BINS INCORRECTLY SET UP" << std::endl;
      exit(1);
    }

    // Load image/depth/extrinsic data for current frame
    unsigned short * depth_data = (unsigned short *) malloc(480 * 640 * sizeof(unsigned short));
    for (int i = 0; i < 480 * 640; i++)
      depth_data[i] = (((unsigned short) curr_frame_depth.data[i * 2 + 1]) << 8) + ((unsigned short) curr_frame_depth.data[i * 2 + 0]);

    // Compute relative camera pose transform between current frame and base frame
    // Compute camera view frustum bounds within the voxel volume
    float camera_relative_pose[16] = {0};
    float view_bounds[6] = {0};
    std::vector<float> curr_extrinsic;
    for (int i = 0; i < 3; i++) {
      curr_extrinsic.push_back(1.0f);
      for (int i = 0; i < 4; i++) {
        curr_extrinsic.push_back(0.0f);
      }
    }
    curr_extrinsic.push_back(1.0f);
    std::vector<std::vector<float>> extrinsics;
    extrinsics.push_back(curr_extrinsic);
    get_frustum_bounds(K, extrinsics, 0, 0, camera_relative_pose, view_bounds,
                       vox_unit, vox_size, vox_range_cam);

    // Copy fusion params to GPU
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_depth_data, depth_data, 480 * 640 * sizeof(unsigned short), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_camera_relative_pose, camera_relative_pose, 16 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_vox_range_cam, vox_range_cam, 6 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // Integrate
    int CUDA_NUM_BLOCKS = vox_size[2];
    int CUDA_NUM_THREADS = vox_size[1];
    integrate <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
        vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
    checkCUDA(__LINE__, cudaGetLastError());

    // Copy data back to memory
    cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDA(__LINE__, cudaGetLastError());

    // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
    //   std::cout << vox_tsdf[i] << std::endl;

    // Save curr volume to file
    std::string scene_ply_name = "volume.pointcloud.ply";
    save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf, vox_weight);

    // Compute bounding box of surface in TSDF volume
    float tsdf_surface_threshold = 0.2f;
    float grid_bounds[6] = {0};
    grid_bounds[0] = vox_size[0]; grid_bounds[2] = vox_size[1]; grid_bounds[4] = vox_size[2];
    for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++) {
      if (std::abs(vox_tsdf[i]) < tsdf_surface_threshold) {
        float z = (float) (floor(i / (vox_size[0] * vox_size[1])));
        float y = (float) (floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]));
        float x = (float) (i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]));
        grid_bounds[0] = std::min(x, grid_bounds[0]); grid_bounds[1] = std::max(x, grid_bounds[1]);
        grid_bounds[2] = std::min(y, grid_bounds[2]); grid_bounds[3] = std::max(y, grid_bounds[3]);
        grid_bounds[4] = std::min(z, grid_bounds[4]); grid_bounds[5] = std::max(z, grid_bounds[5]);
      }
    }

    // Double check bounding box is not near edge of TSDF volume
    grid_bounds[0] = std::max(grid_bounds[0], 15.0f); grid_bounds[1] = std::min(grid_bounds[1], (float)vox_size[0] - 15.0f - 1.0f);
    grid_bounds[2] = std::max(grid_bounds[2], 15.0f); grid_bounds[3] = std::min(grid_bounds[3], (float)vox_size[1] - 15.0f - 1.0f);
    grid_bounds[4] = std::max(grid_bounds[4], 15.0f); grid_bounds[5] = std::min(grid_bounds[5], (float)vox_size[2] - 15.0f - 1.0f);
    // std::cout << grid_bounds[0] << " " << grid_bounds[1] << std::endl;
    // std::cout << grid_bounds[2] << " " << grid_bounds[3] << std::endl;
    // std::cout << grid_bounds[4] << " " << grid_bounds[5] << std::endl;
    int grid_size[3] = {0};
    grid_size[0] = grid_bounds[1] - grid_bounds[0] + 1;
    grid_size[1] = grid_bounds[3] - grid_bounds[2] + 1;
    grid_size[2] = grid_bounds[5] - grid_bounds[4] + 1;

    // Create list of hypothesis cubes (store grid locations, and is valid or not (0 for invalid, 1 for positive, 2 for negative))
    int num_hypothesis = grid_size[0] * grid_size[1] * grid_size[2];
    // std::cout << num_hypothesis << std::endl;
    unsigned short * hypothesis_locations = new unsigned short[3 * num_hypothesis];
    char * hypothesis_labels = new char[num_hypothesis];
    memset(hypothesis_labels, 0, sizeof(char) * num_hypothesis);
    for (int z = grid_bounds[4]; z <= grid_bounds[5]; z++)
      for (int y = grid_bounds[2]; y <= grid_bounds[3]; y++)
        for (int x = grid_bounds[0]; x <= grid_bounds[1]; x++) {
          int hypothesis_idx = (z - grid_bounds[4]) * grid_size[0] * grid_size[1] + (y - grid_bounds[2]) * grid_size[0] + (x - grid_bounds[0]);
          hypothesis_locations[0 * num_hypothesis + hypothesis_idx] = (unsigned short)x;
          hypothesis_locations[1 * num_hypothesis + hypothesis_idx] = (unsigned short)y;
          hypothesis_locations[2 * num_hypothesis + hypothesis_idx] = (unsigned short)z;
        }

    // Copy list of hypothesis cubes to GPU memory
    unsigned short * d_hypothesis_locations;
    char * d_hypothesis_labels;
    cudaMalloc(&d_hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short));
    cudaMalloc(&d_hypothesis_labels, num_hypothesis * sizeof(char));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_hypothesis_locations, hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hypothesis_labels, hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // Copy hypothesis crop information and object center to GPU memory
    unsigned short * d_hypothesis_crop_2D;
    float * d_object_center_cam;
    cudaMalloc(&d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short));
    cudaMalloc(&d_object_center_cam, 3 * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(d_object_center_cam, object_center_cam, 3 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // Run kernel to get labels for hypotheses
    CUDA_NUM_THREADS = 512;
    CUDA_NUM_BLOCKS = (int)ceil(((float)num_hypothesis) / ((float)CUDA_NUM_THREADS));
    gen_hypothesis_labels <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(num_hypothesis, d_hypothesis_locations, d_hypothesis_labels, d_hypothesis_crop_2D, d_object_center_cam, d_K, vox_unit, d_vox_size, d_vox_range_cam, d_vox_tsdf);

    // Copy 2D crop information back to CPU
    unsigned short * hypothesis_crop_2D = new unsigned short[4 * num_hypothesis];
    cudaMemcpy(hypothesis_labels, d_hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(hypothesis_crop_2D, d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    int num_invalid_hypotheses = 0;
    int num_positive_hypotheses = 0;
    int num_negative_hypotheses = 0;
    for (int i = 0; i < num_hypothesis; i++) {
      if (((int)hypothesis_labels[i]) == 0)
        num_invalid_hypotheses++;
      if (((int)hypothesis_labels[i]) == 1) {
        num_positive_hypotheses++;
        // std::cout << (int)hypothesis_locations[0 * num_hypothesis + i] << " " << (int)hypothesis_locations[1 * num_hypothesis + i] << " " << (int)hypothesis_locations[2 * num_hypothesis + i] << std::endl;
        // std::cout << (int)hypothesis_crop_2D[0 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[1 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[2 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[3 * num_hypothesis + i] << std::endl;
        // std::cout << std::endl;
        // cv::Rect curr_patch_ROI((int)hypothesis_crop_2D[0 * num_hypothesis + i], (int)hypothesis_crop_2D[1 * num_hypothesis + i], (int)hypothesis_crop_2D[2 * num_hypothesis + i], (int)hypothesis_crop_2D[3 * num_hypothesis + i]);
        // cv::Mat curr_patch = curr_frame_color(curr_patch_ROI);
        // cv::resize(curr_patch, curr_patch, cv::Size(227, 227));
        // cv::imshow("Patch", curr_patch);
        // cv::waitKey(0);
      }
      if (((int)hypothesis_labels[i]) == 2)
        num_negative_hypotheses++;
    }
    int num_valid_hypotheses = num_positive_hypotheses + num_negative_hypotheses;
    std::cout << "    Number of positive hypotheses found: " << num_positive_hypotheses << std::endl;
    std::cout << "    Number of negative hypotheses found: " << num_negative_hypotheses << std::endl;

    // Save to binary file: 8 x num_valid_hypotheses (int) (label, grid location (x,y,z), hypothesis 2D patch (x,y,width,height))
    std::string labels_filename = sequence_directory + "/" + curr_frame_name + ".labels.bin";
    int * train_labels = new int[num_valid_hypotheses * 8 + 1];
    train_labels[0] = num_valid_hypotheses;
    int train_idx = 0;
    for (int i = 0; i < num_hypothesis; i++) {
      if (((int)hypothesis_labels[i]) > 0) {
        train_labels[0 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_labels[i];
        train_labels[1 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_locations[0 * num_hypothesis + i];
        train_labels[2 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_locations[1 * num_hypothesis + i];
        train_labels[3 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_locations[2 * num_hypothesis + i];
        train_labels[4 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_crop_2D[0 * num_hypothesis + i];
        train_labels[5 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_crop_2D[1 * num_hypothesis + i];
        train_labels[6 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_crop_2D[2 * num_hypothesis + i];
        train_labels[7 * num_valid_hypotheses + train_idx + 1] = (int)hypothesis_crop_2D[3 * num_hypothesis + i];
        train_idx++;
      }
    }
    // for (int i = 0; i < num_valid_hypotheses; i++) {
    //   for (int j = 0; j < 8; j++)
    //     std::cout << train_labels[j * num_valid_hypotheses + i + 1] << " ";
    //   std::cout << std::endl;
    // }
    std::ofstream tmp_out(labels_filename, std::ios::binary | std::ios::out);
    for (int i = 0; i < num_valid_hypotheses * 8 + 1; i++)
      tmp_out.write((char*)&train_labels[i], sizeof(int));
    tmp_out.close();

    // Reset volume in GPU
    reset_vox_GPU <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_vox_size, d_vox_tsdf, d_vox_weight);
    checkCUDA(__LINE__, cudaGetLastError());

    // Free memory
    free(depth_data);
    delete [] hypothesis_locations;
    delete [] hypothesis_labels;
    delete [] hypothesis_crop_2D;
    delete [] train_labels;

    if (d_hypothesis_locations != NULL)
      cudaFree(d_hypothesis_locations);
    checkCUDA(__LINE__, cudaGetLastError());
    if (d_hypothesis_labels != NULL)
      cudaFree(d_hypothesis_labels);
    checkCUDA(__LINE__, cudaGetLastError());
    if (d_hypothesis_crop_2D != NULL)
      cudaFree(d_hypothesis_crop_2D);
    checkCUDA(__LINE__, cudaGetLastError());
    if (d_object_center_cam != NULL)
      cudaFree(d_object_center_cam);
    checkCUDA(__LINE__, cudaGetLastError());

    realloc_fusion_params();
  }
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

  // std::string data_directory = "data/train";
  // std::string object_name = "glue";
  // std::string object_directory = data_directory + "/" + object_name;

  // // Pick a random RGB-D sequence
  // std::vector<std::string> sequence_names;
  // get_files_in_directory(object_directory, sequence_names, "");
  // int rand_sequence_idx = (int)floor(gen_random_float(0, (float)sequence_names.size()));
  // std::string curr_sequence_name = sequence_names[rand_sequence_idx];
  // std::string curr_sequence_directory = object_directory + "/" + curr_sequence_name;
  // // std::cout << curr_sequence_directory << std::endl;

  init_fusion_GPU();
  generate_train_labels("data/train/glue/seq01");







  return 0;
}

