#include "util/util.hpp"
#include "kinfu.hpp"
#include <opencv2/opencv.hpp>
#define DATATYPE 1 // Marvin datatype
#include "marvin.hpp"

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
void gen_hypothesis_labels(int num_hypothesis, unsigned short* tmp_hypothesis_locations, char* tmp_hypothesis_labels, unsigned short* tmp_hypothesis_crop_2D, float* tmp_K, float tmp_vox_unit, int* tmp_vox_size, float* tmp_vox_range_cam, float* tmp_vox_tsdf) {

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

  tmp_hypothesis_labels[hypothesis_idx] = (char)1;

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
    cv::waitKey(10);

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
    kCheckCUDA(__LINE__, cudaMemcpy(d_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_depth_data, depth_data, 480 * 640 * sizeof(unsigned short), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_camera_relative_pose, camera_relative_pose, 16 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_vox_range_cam, vox_range_cam, 6 * sizeof(float), cudaMemcpyHostToDevice));

    // Integrate
    int num_blocks = vox_size[2];
    int num_threads = vox_size[1];
    integrate <<< num_blocks, num_threads >>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
        vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());
    // kCheckCUDA(__LINE__, cudaDeviceSynchronize());

    // Copy data back to memory
    kCheckCUDA(__LINE__, cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
    kCheckCUDA(__LINE__, cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
    //   std::cout << vox_tsdf[i] << std::endl;

    // // Save curr volume to file
    // std::string scene_ply_name = "volume.pointcloud.ply";
    // save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf, vox_weight);

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
    kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short)));
    kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_labels, num_hypothesis * sizeof(char)));
    kCheckCUDA(__LINE__, cudaMemcpy(d_hypothesis_locations, hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_hypothesis_labels, hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyHostToDevice));

    // Copy hypothesis crop information and object center to GPU memory
    unsigned short * d_hypothesis_crop_2D;
    float * d_object_center_cam;
    kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short)));
    kCheckCUDA(__LINE__, cudaMalloc(&d_object_center_cam, 3 * sizeof(float)));
    kCheckCUDA(__LINE__, cudaMemcpy(d_object_center_cam, object_center_cam, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel to get labels for hypotheses
    num_threads = 512;
    num_blocks = (int)ceil(((float)num_hypothesis) / ((float)num_threads));
    // gen_hypothesis_labels<<<num_blocks,num_threads>>>(num_hypothesis, d_hypothesis_locations, d_hypothesis_labels, d_hypothesis_crop_2D, d_object_center_cam, d_K, vox_unit, d_vox_size, d_vox_range_cam, d_vox_tsdf);
    kCheckCUDA(__LINE__, cudaGetLastError());

    // Copy 2D crop information back to CPU
    unsigned short * hypothesis_crop_2D = new unsigned short[4 * num_hypothesis];
    kCheckCUDA(__LINE__, cudaMemcpy(hypothesis_labels, d_hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyDeviceToHost));
    kCheckCUDA(__LINE__, cudaMemcpy(hypothesis_crop_2D, d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short), cudaMemcpyDeviceToHost));

    // Sort hypothesis lists
    std::vector<int> hypothesis_sort_idx_positive;
    std::vector<int> hypothesis_sort_idx_negative;
    int num_invalid_hypotheses = 0;
    int num_positive_hypotheses = 0;
    int num_negative_hypotheses = 0;
    for (int i = 0; i < num_hypothesis; i++) {
      if (((int)hypothesis_labels[i]) == 0)
        num_invalid_hypotheses++;
      if (((int)hypothesis_labels[i]) == 1) {
        num_positive_hypotheses++;
        hypothesis_sort_idx_positive.push_back(i);
        // std::cout << (int)hypothesis_locations[0 * num_hypothesis + i] << " " << (int)hypothesis_locations[1 * num_hypothesis + i] << " " << (int)hypothesis_locations[2 * num_hypothesis + i] << std::endl;
        // std::cout << (int)hypothesis_crop_2D[0 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[1 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[2 * num_hypothesis + i] << " " << (int)hypothesis_crop_2D[3 * num_hypothesis + i] << std::endl;
        // std::cout << std::endl;
        // cv::Rect curr_patch_ROI((int)hypothesis_crop_2D[0 * num_hypothesis + i], (int)hypothesis_crop_2D[1 * num_hypothesis + i], (int)hypothesis_crop_2D[2 * num_hypothesis + i], (int)hypothesis_crop_2D[3 * num_hypothesis + i]);
        // cv::Mat curr_patch = curr_frame_color(curr_patch_ROI);
        // cv::resize(curr_patch, curr_patch, cv::Size(227, 227));
        // cv::imshow("Patch", curr_patch);
        // cv::waitKey(0);
      }
      if (((int)hypothesis_labels[i]) == 2) {
        num_negative_hypotheses++;
        hypothesis_sort_idx_negative.push_back(i);
      }
    }
    int num_valid_hypotheses = num_positive_hypotheses + num_negative_hypotheses;
    std::cout << "    Number of positive hypotheses found: " << hypothesis_sort_idx_positive.size() << std::endl;
    std::cout << "    Number of negative hypotheses found: " << hypothesis_sort_idx_negative.size() << std::endl;

    // Save to binary file: 8 x num_valid_hypotheses (int) (label, grid location (x,y,z), hypothesis 2D patch (x,y,width,height)), positive first then negative
    std::string labels_filename = sequence_directory + "/" + curr_frame_name + ".labels.bin";
    std::ofstream tmp_out(labels_filename, std::ios::binary | std::ios::out);
    tmp_out.write((char*)&num_valid_hypotheses, sizeof(int));
    tmp_out.write((char*)&num_positive_hypotheses, sizeof(int));
    tmp_out.write((char*)&num_negative_hypotheses, sizeof(int));
    for (int i = 0; i < num_positive_hypotheses; i++) {
      unsigned short tmp_var[8];
      tmp_var[0] = (unsigned short)hypothesis_labels[hypothesis_sort_idx_positive[i]];
      tmp_var[1] = (unsigned short)hypothesis_locations[0 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[2] = (unsigned short)hypothesis_locations[1 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[3] = (unsigned short)hypothesis_locations[2 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[4] = (unsigned short)hypothesis_crop_2D[0 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[5] = (unsigned short)hypothesis_crop_2D[1 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[6] = (unsigned short)hypothesis_crop_2D[2 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      tmp_var[7] = (unsigned short)hypothesis_crop_2D[3 * num_hypothesis + hypothesis_sort_idx_positive[i]];
      for (int j = 0; j < 8; j++)
        tmp_out.write((char*)&tmp_var[j], sizeof(unsigned short));
    }
    for (int i = 0; i < num_negative_hypotheses; i++) {
      unsigned short tmp_var[8];
      tmp_var[0] = (unsigned short)hypothesis_labels[hypothesis_sort_idx_negative[i]];
      tmp_var[1] = (unsigned short)hypothesis_locations[0 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[2] = (unsigned short)hypothesis_locations[1 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[3] = (unsigned short)hypothesis_locations[2 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[4] = (unsigned short)hypothesis_crop_2D[0 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[5] = (unsigned short)hypothesis_crop_2D[1 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[6] = (unsigned short)hypothesis_crop_2D[2 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      tmp_var[7] = (unsigned short)hypothesis_crop_2D[3 * num_hypothesis + hypothesis_sort_idx_negative[i]];
      for (int j = 0; j < 8; j++)
        tmp_out.write((char*)&tmp_var[j], sizeof(unsigned short));
    }
    tmp_out.close();

    // Reset volume in GPU
    num_blocks = vox_size[2];
    num_threads = vox_size[1];
    reset_vox_whole_GPU <<< num_blocks, num_threads >>>(d_vox_size, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());

    // Free memory
    free(depth_data);
    delete [] hypothesis_locations;
    delete [] hypothesis_labels;
    delete [] hypothesis_crop_2D;

    kCheckCUDA(__LINE__, cudaFree(d_hypothesis_locations));
    kCheckCUDA(__LINE__, cudaFree(d_hypothesis_labels));
    kCheckCUDA(__LINE__, cudaFree(d_hypothesis_crop_2D));
    kCheckCUDA(__LINE__, cudaFree(d_object_center_cam));
  }
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

  init_fusion_GPU();

  // Load intrinsics (3x3 matrix)
  std::string intrinsic_filename = "TMP.intrinsics.K.txt";
  std::vector<float> K_vec = load_matrix_from_file(intrinsic_filename, 3, 3);
  float * K = &K_vec[0];
  for (int i = 0; i < 9; i++)
    std::cout << K[i] << std::endl;

  // Load RGB-D frame
  std::string curr_frame_color_filename = "TMP.frame.color.png";
  cv::Mat curr_frame_color = cv::imread(curr_frame_color_filename.c_str(), 1);
  std::string curr_frame_depth_filename = "TMP.frame.depth.png";
  cv::Mat curr_frame_depth = cv::imread(curr_frame_depth_filename.c_str(), CV_LOAD_IMAGE_UNCHANGED);

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
  kCheckCUDA(__LINE__, cudaMemcpy(d_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
  kCheckCUDA(__LINE__, cudaMemcpy(d_depth_data, depth_data, 480 * 640 * sizeof(unsigned short), cudaMemcpyHostToDevice));
  kCheckCUDA(__LINE__, cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice));
  kCheckCUDA(__LINE__, cudaMemcpy(d_camera_relative_pose, camera_relative_pose, 16 * sizeof(float), cudaMemcpyHostToDevice));
  kCheckCUDA(__LINE__, cudaMemcpy(d_vox_range_cam, vox_range_cam, 6 * sizeof(float), cudaMemcpyHostToDevice));

  // Integrate
  int num_blocks = vox_size[2];
  int num_threads = vox_size[1];
  integrate <<< num_blocks, num_threads >>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
      vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
  kCheckCUDA(__LINE__, cudaGetLastError());
  // kCheckCUDA(__LINE__, cudaDeviceSynchronize());

  // Copy data back to memory
  kCheckCUDA(__LINE__, cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
  kCheckCUDA(__LINE__, cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));

  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   std::cout << vox_tsdf[i] << std::endl;

  // Save curr volume to file
  std::string scene_ply_name = "volume.pointcloud.ply";
  save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf);

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
  int hop_size = 4;
  grid_size[0] = grid_bounds[1] - grid_bounds[0] + 1;
  grid_size[1] = grid_bounds[3] - grid_bounds[2] + 1;
  grid_size[2] = grid_bounds[5] - grid_bounds[4] + 1;
  for (int i = 0; i < 3; i++)
    grid_size[i] = (int)std::ceil(((float)grid_size[i])/((float)hop_size));

  // Create list of hypothesis cubes (store grid locations, and is valid or not (0 for invalid, 1 for positive, 2 for negative))
  int num_hypothesis = grid_size[0] * grid_size[1] * grid_size[2];
  std::cout << num_hypothesis << std::endl;
  unsigned short * hypothesis_locations = new unsigned short[3 * num_hypothesis];
  char * hypothesis_labels = new char[num_hypothesis];
  memset(hypothesis_labels, 0, sizeof(char) * num_hypothesis);
  for (int z = grid_bounds[4]; z <= grid_bounds[5]; z = z + hop_size)
    for (int y = grid_bounds[2]; y <= grid_bounds[3]; y = y + hop_size)
      for (int x = grid_bounds[0]; x <= grid_bounds[1]; x = x + hop_size) {
        int hypothesis_idx = (z - grid_bounds[4])/hop_size * grid_size[0] * grid_size[1] + 
                             (y - grid_bounds[2])/hop_size * grid_size[0] + 
                             (x - grid_bounds[0])/hop_size;
        // std::cout << x << " " << y << " " << z << std::endl;
        std::cout << hypothesis_idx << std::endl;
        hypothesis_locations[0 * num_hypothesis + hypothesis_idx] = (unsigned short)x;
        hypothesis_locations[1 * num_hypothesis + hypothesis_idx] = (unsigned short)y;
        hypothesis_locations[2 * num_hypothesis + hypothesis_idx] = (unsigned short)z;
      }
  // std::cout << num_hypothesis << std::endl;
  // for (int i = 0; i < num_hypothesis; i++) {
  //   std::cout << hypothesis_locations[0 * num_hypothesis + i] << " " << hypothesis_locations[1 * num_hypothesis + i] << " " << hypothesis_locations[2 * num_hypothesis + i] << std::endl;
  // }

  // Copy list of hypothesis cubes to GPU memory
  unsigned short * d_hypothesis_locations;
  char * d_hypothesis_labels;
  kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short)));
  kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_labels, num_hypothesis * sizeof(char)));
  kCheckCUDA(__LINE__, cudaMemcpy(d_hypothesis_locations, hypothesis_locations, 3 * num_hypothesis * sizeof(unsigned short), cudaMemcpyHostToDevice));
  kCheckCUDA(__LINE__, cudaMemcpy(d_hypothesis_labels, hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyHostToDevice));

  // Copy hypothesis crop information to GPU memory
  unsigned short * d_hypothesis_crop_2D;
  kCheckCUDA(__LINE__, cudaMalloc(&d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short)));

  // Run kernel to get labels for hypotheses
  num_threads = 512;
  num_blocks = (int)ceil(((float)num_hypothesis) / ((float)num_threads));
  gen_hypothesis_labels<<<num_blocks,num_threads>>>(num_hypothesis, d_hypothesis_locations, d_hypothesis_labels, d_hypothesis_crop_2D, d_K, vox_unit, d_vox_size, d_vox_range_cam, d_vox_tsdf);
  kCheckCUDA(__LINE__, cudaGetLastError());

  // Copy 2D crop information back to CPU
  unsigned short * hypothesis_crop_2D = new unsigned short[4 * num_hypothesis];
  kCheckCUDA(__LINE__, cudaMemcpy(hypothesis_labels, d_hypothesis_labels, num_hypothesis * sizeof(char), cudaMemcpyDeviceToHost));
  kCheckCUDA(__LINE__, cudaMemcpy(hypothesis_crop_2D, d_hypothesis_crop_2D, 4 * num_hypothesis * sizeof(unsigned short), cudaMemcpyDeviceToHost));

  int num_valid_hypotheses = 0;
  for (int i = 0; i < num_hypothesis; i++)
    if ((int)(hypothesis_labels[i]) == 1)
      num_valid_hypotheses++;


  std::cout << num_valid_hypotheses << std::endl;
  std::cout << num_hypothesis << std::endl;
  // ROS_INFO("Found %d hypothesis bounding boxes.", num_valid_hypotheses);
  // ROS_INFO("Saving hypotheses to tensors on disk for Marvin.");

  // Init tensor files for marvin
  std::string data2D_tensor_filename = "TMP.data.2D.tensor";
  std::string data3D_tensor_filename = "TMP.data.3D.tensor";
  FILE *data2D_tensor_fp = fopen(data2D_tensor_filename.c_str(), "w");
  FILE *data3D_tensor_fp = fopen(data3D_tensor_filename.c_str(), "w");

  // Write data header for 2D
  uint8_t type_id = (uint8_t)1;
  fwrite((void*)&type_id, sizeof(uint8_t), 1, data2D_tensor_fp);
  uint32_t size_of = (uint32_t)4;
  fwrite((void*)&size_of, sizeof(uint32_t), 1, data2D_tensor_fp);
  uint32_t str_len = (uint32_t)4;
  fwrite((void*)&str_len, sizeof(uint32_t), 1, data2D_tensor_fp);
  fprintf(data2D_tensor_fp, "data");
  uint32_t data_dim = (uint32_t)4;
  fwrite((void*)&data_dim, sizeof(uint32_t), 1, data2D_tensor_fp);
  uint32_t data_size = (uint32_t)num_valid_hypotheses;
  fwrite((void*)&data_size, sizeof(uint32_t), 1, data2D_tensor_fp);
  uint32_t data_chan = (uint32_t)3;
  fwrite((void*)&data_chan, sizeof(uint32_t), 1, data2D_tensor_fp);
  uint32_t patch_dim = (uint32_t)227;
  fwrite((void*)&patch_dim, sizeof(uint32_t), 1, data2D_tensor_fp);
  fwrite((void*)&patch_dim, sizeof(uint32_t), 1, data2D_tensor_fp);

  // Write data header for 3D
  type_id = (uint8_t)1;
  fwrite((void*)&type_id, sizeof(uint8_t), 1, data3D_tensor_fp);
  size_of = (uint32_t)4;
  fwrite((void*)&size_of, sizeof(uint32_t), 1, data3D_tensor_fp);
  str_len = (uint32_t)4;
  fwrite((void*)&str_len, sizeof(uint32_t), 1, data3D_tensor_fp);
  fprintf(data3D_tensor_fp, "data");
  data_dim = (uint32_t)5;
  fwrite((void*)&data_dim, sizeof(uint32_t), 1, data3D_tensor_fp);
  data_size = (uint32_t)num_valid_hypotheses;
  fwrite((void*)&data_size, sizeof(uint32_t), 1, data3D_tensor_fp);
  data_chan = (uint32_t)1;
  fwrite((void*)&data_chan, sizeof(uint32_t), 1, data3D_tensor_fp);
  uint32_t volume_dim = (uint32_t)30;
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);

  // Write hypothesis cubes and patches to tensor file
  for (int hypothesis_idx = 0; hypothesis_idx < num_hypothesis; hypothesis_idx++) {
    if ((int)(hypothesis_labels[hypothesis_idx]) == 1) {
      int x = hypothesis_locations[0 * num_hypothesis + hypothesis_idx];
      int y = hypothesis_locations[1 * num_hypothesis + hypothesis_idx];
      int z = hypothesis_locations[2 * num_hypothesis + hypothesis_idx];

      // Get 3D cube
      float * curr_cube = new float[30 * 30 * 30];
      for (int i = -15; i < 15; i++)
        for (int j = -15; j < 15; j++)
          for (int k = -15; k < 15; k++) {
            int volumeIDX = (z + k) * vox_size[0] * vox_size[1] + (y + j) * vox_size[0] + (x + i);
            curr_cube[(k + 15) * 30 * 30 + (j + 15) * 30 + (i + 15)] = vox_tsdf[volumeIDX];
          }

      // Get 2D patch of cube's 2D project to image
      cv::Rect curr_patch_ROI(hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx], hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx], hypothesis_crop_2D[2 * num_hypothesis + hypothesis_idx], hypothesis_crop_2D[3 * num_hypothesis + hypothesis_idx]);
      // std::cout << std::round(cube_front_2D[2]) << " " << std::round(cube_front_2D[6]) << " " << std::round(cube_front_2D[1]-cube_front_2D[2]) << " " << std::round(cube_front_2D[4]-cube_front_2D[6]) << std::endl;
      cv::Mat curr_patch = curr_frame_color(curr_patch_ROI);
      cv::resize(curr_patch, curr_patch, cv::Size(227, 227));

      // Write 2D image patch to data tensor file (bgr and subtract mean)
      float * patch_data = new float[3 * 227 * 227];
      for (int tmp_row = 0; tmp_row < 227; tmp_row++)
        for (int tmp_col = 0; tmp_col < 227; tmp_col++) {
          patch_data[0 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row, tmp_col)[0]) - 102.9801f; // B
          patch_data[1 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row, tmp_col)[1]) - 115.9465f; // G
          patch_data[2 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row, tmp_col)[2]) - 122.7717f; // R
        }
      fwrite(patch_data, sizeof(float), 3 * 227 * 227, data2D_tensor_fp);

      // Write 3D tsdf volume to data tensor file
      fwrite(curr_cube, sizeof(float), 30 * 30 * 30, data3D_tensor_fp);

      // Clear memory
      delete [] patch_data;
      delete [] curr_cube;
    }
  }
  fclose(data2D_tensor_fp);
  fclose(data3D_tensor_fp);

  // Clear memory
  free(depth_data);
  cudaFree(d_hypothesis_locations);
  cudaFree(d_hypothesis_labels);
  cudaFree(d_hypothesis_crop_2D);
  cudaFree(d_vox_tsdf);
  cudaFree(d_vox_weight);

  // Run marvin
  std::cout << "Running Marvin for 2D/3D deep learning." << std::endl;
  std::string class_score_tensor_filename = "TMP.class_score_response.tensor";
  std::string axis_score_tensor_filename = "TMP.axis_score_response.tensor";
  std::string angle_score_tensor_filename = "TMP.angle_score_response.tensor";
  sys_command("rm " + class_score_tensor_filename);
  sys_command("rm " + axis_score_tensor_filename);
  sys_command("rm " + angle_score_tensor_filename);
  marvin::Net net("tools/marvin/modelTest.json");
  net.Malloc(marvin::Testing);
  std::vector<std::string> models = marvin::getStringVector("tools/marvin/PeriodNet.1.3.60000.marvin");
  for (int m=0;m<models.size();++m)   net.loadWeights(models[m]);
  int itersPerSave = 0;
  net.test(marvin::getStringVector("class_score"), marvin::getStringVector("TMP.class_score_response.tensor"), itersPerSave);
  // sys_command("cd src/apc_vision/tools/marvin; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cudnn/v4rc/lib64; ./marvin test model.json PeriodNet.marvin class_score ../../../../" + class_score_tensor_filename);
  // sys_command("rm " + data2D_tensor_filename);
  // sys_command("rm " + data3D_tensor_filename);

  // Parse class scores
  std::ifstream inFile(class_score_tensor_filename, std::ios::binary | std::ios::in);
  int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  inFile.seekg(size_t(header_bytes));
  float * class_score_raw = new float[num_hypothesis * 2];
  inFile.read((char*)class_score_raw, num_hypothesis * 2 * sizeof(float));
  inFile.close();
  sys_command("rm " + class_score_tensor_filename);

  float highest_class_score = 0;
  int best_guess_IDX = 0;
  int valid_hypothesis_idx = 0;
  for (int hypothesis_idx = 0; hypothesis_idx < num_hypothesis; hypothesis_idx++) {
    if ((int)(hypothesis_labels[hypothesis_idx]) == 1) {
      if (class_score_raw[valid_hypothesis_idx * 2 + 1] > 0.5f) {
        int crop_x1 = hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx];
        int crop_y1 = hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx];
        int crop_x2 = hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx] + hypothesis_crop_2D[2 * num_hypothesis + hypothesis_idx];
        int crop_y2 = hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx] + hypothesis_crop_2D[3 * num_hypothesis + hypothesis_idx];
        cv::rectangle(curr_frame_color, cv::Point(crop_x1, crop_y1), cv::Point(crop_x2, crop_y2), cv::Scalar(255, 0, 0));
        if (class_score_raw[valid_hypothesis_idx * 2 + 1] > highest_class_score) {
          highest_class_score = class_score_raw[valid_hypothesis_idx * 2 + 1];
          best_guess_IDX = hypothesis_idx;
        }
      }
      valid_hypothesis_idx++;
    }
  }

  std::cout << "Finished running Marvin." << std::endl;

  // If no objects are detected
  if (highest_class_score == 0) {
    std::cout << "No objects detected!" << std::endl;
  }

  // Display detection results
  int crop_x1 = hypothesis_crop_2D[0 * num_hypothesis + best_guess_IDX];
  int crop_y1 = hypothesis_crop_2D[1 * num_hypothesis + best_guess_IDX];
  int crop_x2 = hypothesis_crop_2D[0 * num_hypothesis + best_guess_IDX] + hypothesis_crop_2D[2 * num_hypothesis + best_guess_IDX];
  int crop_y2 = hypothesis_crop_2D[1 * num_hypothesis + best_guess_IDX] + hypothesis_crop_2D[3 * num_hypothesis + best_guess_IDX];
  cv::rectangle(curr_frame_color, cv::Point(crop_x1, crop_y1), cv::Point(crop_x2, crop_y2), cv::Scalar(0, 255, 0));
  cv::circle(curr_frame_color, cv::Point((crop_x1 + crop_x2) / 2, (crop_y1 + crop_y2) / 2), 5, cv::Scalar(0, 255, 0), -1);
  cv::imwrite( "result.png", curr_frame_color);



















  return 0;
}

