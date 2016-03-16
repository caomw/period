#include "util.hpp"
#include "kinfu.hpp"
// #include <opencv2/opencv.hpp>

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

void patch2tensor(cv::Mat curr_patch, float* patch_data) {
  for (int tmp_row = 0; tmp_row < 227; tmp_row++)
    for (int tmp_col = 0; tmp_col < 227; tmp_col++) {
      patch_data[0 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[0]) - 102.9801f; // B
      patch_data[1 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[1]) - 115.9465f; // G
      patch_data[2 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[2]) - 122.7717f; // R
    }
}

////////////////////////////////////////////////////////////////////////////////

__global__
void process_3D_patch(int batch_idx, float* tmp_batch_3D, int crop_x, int crop_y, int crop_z, int* tmp_vox_size, float* tmp_vox_tsdf) {
  int z = crop_z - 15 + blockIdx.x;
  int y = crop_y - 15 + threadIdx.x;
  for (int x = crop_x - 15; x < crop_x + 15; x++) {
    tmp_batch_3D[(batch_idx * 30 * 30 * 30) + blockIdx.x * 30 * 30 + threadIdx.x * 30 + (x - (crop_x - 15))] = tmp_vox_tsdf[z * tmp_vox_size[0] * tmp_vox_size[1] + y * tmp_vox_size[0] + x];
  }
}

////////////////////////////////////////////////////////////////////////////////

std::vector<cv::Mat> gen_train_hypothesis_pair(int batch_idx, float* d_batch_3D, const std::string &object_directory, int* positive_hypothesis_crop_info, int* negative_hypothesis_crop_info, int* axis_angle_label) {

  bool is_hypothesis_pair_found = false;
  std::vector<cv::Mat> patches_2D;
  while (!is_hypothesis_pair_found) {

    // Pick a random RGB-D sequence
    std::vector<std::string> sequence_names;
    get_files_in_directory(object_directory, sequence_names, "");
    int rand_sequence_idx = (int)floor(gen_random_float(0, (float)sequence_names.size()));
    std::string curr_sequence_name = sequence_names[rand_sequence_idx];
    std::string curr_sequence_directory = object_directory + "/" + curr_sequence_name;
    // std::cout << curr_sequence_directory << std::endl;

    // Pick a random RGB-D frame
    std::vector<std::string> frame_names;
    get_files_in_directory(curr_sequence_directory, frame_names, ".color.png");
    // for (int i = 0; i < frame_names.size(); i++)
    //   std::cout << frame_names[i] << std::endl;
    int rand_frame_idx = (int)floor(gen_random_float(0, (float)frame_names.size()));
    std::string curr_frame_name = frame_names[rand_frame_idx];
    curr_frame_name = curr_frame_name.substr(0, curr_frame_name.length() - 10);
    std::cout << "Preparing Training Frame: " << curr_sequence_directory << "/" << curr_frame_name << std::endl;

    // Load intrinsics (3x3 matrix)
    std::string intrinsic_filename = curr_sequence_directory + "/intrinsics.K.txt";
    std::vector<float> K_vec = load_matrix_from_file(intrinsic_filename, 3, 3);
    // float K[9];
    // for (int i = 0; i < 9; i++)
    //   K[i] = K_vec[i];
    float * K = &K_vec[0];
    // for (int i = 0; i < 9; i++)
    //   std::cout << K[i] << std::endl;

    // Load RGB-D frame
    std::string curr_frame_color_filename = curr_sequence_directory + "/" + curr_frame_name + ".color.png";
    cv::Mat curr_frame_color = cv::imread(curr_frame_color_filename.c_str(), 1);
    std::string curr_frame_depth_filename = curr_sequence_directory + "/" + curr_frame_name + ".depth.png";
    cv::Mat curr_frame_depth = cv::imread(curr_frame_depth_filename.c_str(), CV_LOAD_IMAGE_UNCHANGED);

    // Read ground truth object pose from file
    std::string object_pose_filename = curr_sequence_directory + "/object.pose.txt";
    std::vector<float> object_pose_raw = load_matrix_from_file(object_pose_filename, 4, 4);
    float * object_pose_arr = &object_pose_raw[0];
    // for (int i = 0; i < 12; i++)
    //   std::cout << object_pose[i] << std::endl;

    // Compute ground truth object pose w.r.t. current camera pose
    std::string curr_cam_pose_filename = curr_sequence_directory + "/" + curr_frame_name + ".pose.txt";
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
    // show_object_pose(K, object_pose, curr_frame_color);
    // cv::waitKey(0);

    // // Compute center of ground truth object in 3D camera coordinates
    // float object_center_cam[3] = {0};
    // for (int i = 0; i < 3; i++)
    //   object_center_cam[i] = object_pose[i * 4 + 3];

    // Convert pose from rotation matrix to axis/angle (radians) representation (x, y, z, theta)
    float object_pose_axis[3] = {0};
    float object_pose_angle = std::acos(0.5f * (object_pose[0] + object_pose[5] + object_pose[10] - 1));
    object_pose_axis[0] = (object_pose[9] - object_pose[6]) / (2 * std::sin(object_pose_angle));
    object_pose_axis[1] = (object_pose[2] - object_pose[8]) / (2 * std::sin(object_pose_angle));
    object_pose_axis[2] = (object_pose[4] - object_pose[1]) / (2 * std::sin(object_pose_angle));
    // for (int i = 0; i < 3; i++)
    //   std::cout << object_pose_axis[i] << std::endl;

    // // Convert axis/angle to pose
    // float object_pose_rotation[9] = {0};
    // object_pose_rotation[0 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[0] + std::cos(object_pose_angle);
    // object_pose_rotation[0 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[1] - object_pose_axis[2] * std::sin(object_pose_angle);
    // object_pose_rotation[0 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[0] * object_pose_axis[2] + object_pose_axis[1] * std::sin(object_pose_angle);
    // object_pose_rotation[1 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[0] + object_pose_axis[2] * std::sin(object_pose_angle);
    // object_pose_rotation[1 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[1] + std::cos(object_pose_angle);
    // object_pose_rotation[1 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[1] * object_pose_axis[2] - object_pose_axis[0] * std::sin(object_pose_angle);
    // object_pose_rotation[2 * 3 + 0] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[0] - object_pose_axis[1] *  std::sin(object_pose_angle);
    // object_pose_rotation[2 * 3 + 1] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[1] + object_pose_axis[0] * std::sin(object_pose_angle);
    // object_pose_rotation[2 * 3 + 2] = (1 - std::cos(object_pose_angle)) * object_pose_axis[2] * object_pose_axis[2] + std::cos(object_pose_angle);
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
    int closest_angle_bin = (int)floor(object_pose_angle / (3.14159265 / 18));
    if (closest_angle_bin > 17 || closest_axis_bin > 41) {
      std::cout << "AXIS/ANGLE BINS INCORRECTLY SET UP" << std::endl;
      exit(1);
    }

    // Save axis and angle labels
    axis_angle_label[0] = closest_axis_bin;
    axis_angle_label[1] = closest_angle_bin;

    // Read binary files containing labels
    std::string labels_filename = curr_sequence_directory + "/" + curr_frame_name + ".labels.bin";
    std::ifstream inFile(labels_filename.c_str(), std::ios::binary | std::ios::in);
    int num_valid_hypothesis = 0;
    int num_positive_hypotheses = 0;
    int num_negative_hypotheses = 0;
    inFile.read((char*)&num_valid_hypothesis, sizeof(int));
    inFile.read((char*)&num_positive_hypotheses, sizeof(int));
    inFile.read((char*)&num_negative_hypotheses, sizeof(int));
    // int * train_labels = new int[num_valid_hypothesis * 8];
    // for (int i = 0; i < num_valid_hypothesis * 8; i++) {
    //   // inFile.read((char*)&train_labels[i], sizeof(int));
    //   unsigned short tmp_label;
    //   inFile.read((char*)&tmp_label, sizeof(unsigned short));
    //   train_labels[i] = (int) tmp_label;
    // }
    // for (int i = 0; i < num_valid_hypothesis; i++) {
    //   for (int j = 0; j < 8; j++)
    //     std::cout << train_labels[i * 8 + j] << " ";
    //   std::cout << std::endl;
    // }
    // for (int i = 0; i < 8; i++)
    //   std::cout << train_labels[rand_positive_idx * 8 + i] << std::endl;
    // for (int i = 0; i < 8; i++)
    //   std::cout << train_labels[(num_positive_hypotheses + rand_negative_idx) * 8 + i] << std::endl;
    
    // If no positive hypotheses, exit
    if (num_positive_hypotheses == 0 || num_negative_hypotheses == 0)
      continue; 

    // Randomly select a positive and a negative case
    int rand_positive_idx = (int)floor(gen_random_float(0, (float)num_positive_hypotheses));
    int rand_negative_idx = (int)floor(gen_random_float(0, (float)num_negative_hypotheses));
    inFile.seekg(3 * sizeof(int) + rand_positive_idx * 8 * sizeof(unsigned short));
    for (int i = 0; i < 8; i++) {
      unsigned short tmp_label;
      inFile.read((char*)&tmp_label, sizeof(unsigned short));
      positive_hypothesis_crop_info[i] = (int) tmp_label;
    }
    inFile.seekg(3 * sizeof(int) + num_positive_hypotheses * 8 * sizeof(unsigned short) + rand_negative_idx * 8 * sizeof(unsigned short));
    for (int i = 0; i < 8; i++) {
      unsigned short tmp_label;
      inFile.read((char*)&tmp_label, sizeof(unsigned short));
      negative_hypothesis_crop_info[i] = (int) tmp_label;
    }
    inFile.close();
    for (int i = 0; i < 8; i++)
      std::cout << positive_hypothesis_crop_info[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < 8; i++)
      std::cout << negative_hypothesis_crop_info[i] << " ";
    std::cout << std::endl;

    // Load image/depth/extrinsic data for current frame
    unsigned short * depth_data = (unsigned short *) malloc(480 * 640 * sizeof(unsigned short));
    for (int i = 0; i < 480 * 640; i++)
      depth_data[i] = (((unsigned short) curr_frame_depth.data[i * 2 + 1]) << 8) + ((unsigned short) curr_frame_depth.data[i * 2 + 0]);

    // Compute relative camera pose transform between current frame and base frame
    // Compute camera view frustum bounds within the voxel volume
    float camera_relative_pose[16] = {0};
    float view_bounds[6] = {0};
    // std::vector<float> curr_extrinsic;
    // for (int i = 0; i < 3; i++) {
    //   curr_extrinsic.push_back(1.0f);
    //   for (int i = 0; i < 4; i++) {
    //     curr_extrinsic.push_back(0.0f);
    //   }
    // }
    // curr_extrinsic.push_back(1.0f);
    // std::vector<std::vector<float>> extrinsics;
    // extrinsics.push_back(curr_extrinsic);
    // get_frustum_bounds(K, extrinsics, 0, 0, camera_relative_pose, view_bounds,
    //                    vox_unit, vox_size, vox_range_cam);
    load_identity_matrix(camera_relative_pose);

    // Set view bounds for positive cropped box
    view_bounds[0] = positive_hypothesis_crop_info[1] - 15; 
    view_bounds[1] = positive_hypothesis_crop_info[1] + 15; 
    view_bounds[2] = positive_hypothesis_crop_info[2] - 15; 
    view_bounds[3] = positive_hypothesis_crop_info[2] + 15; 
    view_bounds[4] = positive_hypothesis_crop_info[3] - 15; 
    view_bounds[5] = positive_hypothesis_crop_info[3] + 15; 

    // Show 2D image patch
    cv::Rect curr_positive_patch_ROI(positive_hypothesis_crop_info[4], positive_hypothesis_crop_info[5], positive_hypothesis_crop_info[6], positive_hypothesis_crop_info[7]);
    cv::Mat positive_patch_2D = curr_frame_color(curr_positive_patch_ROI);
    cv::resize(positive_patch_2D, positive_patch_2D, cv::Size(227, 227));

    // Copy fusion params to GPU
    kCheckCUDA(__LINE__, cudaMemcpy(d_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_depth_data, depth_data, 480 * 640 * sizeof(unsigned short), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_camera_relative_pose, camera_relative_pose, 16 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(d_vox_range_cam, vox_range_cam, 6 * sizeof(float), cudaMemcpyHostToDevice));

    // Integrate
    // int CUDA_NUM_BLOCKS = vox_size[2];
    // int CUDA_NUM_THREADS = vox_size[1];
    integrate<<<30,30>>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose, vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());
    // kCheckCUDA(__LINE__, cudaDeviceSynchronize());

    // // Copy data back to memory
    // kCheckCUDA(__LINE__, cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
    // kCheckCUDA(__LINE__, cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));

    // Crop out 3D patch from TSDF volume
    process_3D_patch<<<30,30>>>(batch_idx * 2, d_batch_3D, positive_hypothesis_crop_info[1], positive_hypothesis_crop_info[2], positive_hypothesis_crop_info[3], d_vox_size, d_vox_tsdf);
    kCheckCUDA(__LINE__, cudaGetLastError());

    // Reset voxel volume
    reset_vox_GPU<<<30,30>>>(d_view_bounds, d_vox_size, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());

    // // Save curr volume to file
    // std::string scene_ply_name = "volume.pos.pointcloud.ply";
    // save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf, vox_weight);

    // Set view bounds for positive 3D cropped box
    view_bounds[0] = negative_hypothesis_crop_info[1] - 15; 
    view_bounds[1] = negative_hypothesis_crop_info[1] + 15; 
    view_bounds[2] = negative_hypothesis_crop_info[2] - 15; 
    view_bounds[3] = negative_hypothesis_crop_info[2] + 15; 
    view_bounds[4] = negative_hypothesis_crop_info[3] - 15; 
    view_bounds[5] = negative_hypothesis_crop_info[3] + 15; 

    // Show 2D image patch
    cv::Rect curr_negative_patch_ROI(negative_hypothesis_crop_info[4], negative_hypothesis_crop_info[5], negative_hypothesis_crop_info[6], negative_hypothesis_crop_info[7]);
    cv::Mat negative_patch_2D = curr_frame_color(curr_negative_patch_ROI);
    cv::resize(negative_patch_2D, negative_patch_2D, cv::Size(227, 227));

    // Copy fusion params to GPU
    kCheckCUDA(__LINE__, cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice));

    // Integrate
    integrate<<<30,30>>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose, vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());
    // kCheckCUDA(__LINE__, cudaDeviceSynchronize());

    // // Copy data back to memory
    // kCheckCUDA(__LINE__, cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
    // kCheckCUDA(__LINE__, cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Crop out 3D patch from TSDF volume
    process_3D_patch<<<30,30>>>(batch_idx * 2 + 1, d_batch_3D, negative_hypothesis_crop_info[1], negative_hypothesis_crop_info[2], negative_hypothesis_crop_info[3], d_vox_size, d_vox_tsdf);
    kCheckCUDA(__LINE__, cudaGetLastError());
    
    // Reset voxel volume
    reset_vox_GPU<<<30,30>>>(d_view_bounds, d_vox_size, d_vox_tsdf, d_vox_weight);
    kCheckCUDA(__LINE__, cudaGetLastError());

    // // Save curr volume to file
    // scene_ply_name = "volume.neg.pointcloud.ply";
    // save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf, vox_weight);

    free(depth_data);

    // Return 2D patches
    patches_2D.push_back(positive_patch_2D);
    patches_2D.push_back(negative_patch_2D);
    is_hypothesis_pair_found = true;
  }
  return patches_2D;
}

////////////////////////////////////////////////////////////////////////////////

int sample(int argc, char **argv) {

  init_fusion_GPU();
  int positive_hypothesis_crop_info[8] = {0};
  int negative_hypothesis_crop_info[8] = {0};

  float * d_batch_2D;
  float * d_batch_3D;
  kCheckCUDA(__LINE__, cudaMalloc(&d_batch_2D, 32 * 3 * 227 * 227 * sizeof(float)));
  kCheckCUDA(__LINE__, cudaMalloc(&d_batch_3D, 32 * 30 * 30 * 30 * sizeof(float)));

  tic();
  for (int i = 0; i < 16; i++) {

    // Create positive and negative hypothesis
    std::string data_directory = "data/train";
    std::string object_name = "glue";
    std::string object_directory = data_directory + "/" + object_name;
    int axis_angle_label[2] = {0};
    std::vector<cv::Mat> patches_2D = gen_train_hypothesis_pair(i, d_batch_3D, object_directory, positive_hypothesis_crop_info, negative_hypothesis_crop_info, axis_angle_label);

    // // Show image patches
    // cv::namedWindow("Positive Patch", CV_WINDOW_AUTOSIZE);
    // cv::imshow("Positive Patch", patches_2D[0]);
    // cv::namedWindow("Negative Patch", CV_WINDOW_AUTOSIZE);
    // cv::imshow("Negative Patch", patches_2D[1]);
    // cv::waitKey(0);

    // Write 2D image patches to data tensors (bgr and subtract mean)
    float * pos_patch_data = new float[3 * 227 * 227];
    float * neg_patch_data = new float[3 * 227 * 227];
    patch2tensor(patches_2D[0], pos_patch_data);
    patch2tensor(patches_2D[1], neg_patch_data);
    kCheckCUDA(__LINE__, cudaMemcpy(&d_batch_2D[(i * 2) * 3 * 227 * 227], pos_patch_data, 3 * 227 * 227 * sizeof(float), cudaMemcpyHostToDevice));
    kCheckCUDA(__LINE__, cudaMemcpy(&d_batch_2D[(i * 2 + 1) * 3 * 227 * 227], neg_patch_data, 3 * 227 * 227 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy axis/angle label
    std::cout << axis_angle_label[0] << " " << axis_angle_label[1] << std::endl;

  }
  toc();

  float * batch_2D = new float[32 * 3 * 227 * 227];
  float * batch_3D = new float[32 * 30 * 30 * 30];
  memset(batch_2D, 0, sizeof(float) * 32 * 3 * 227 * 227);
  memset(batch_3D, 0, sizeof(float) * 32 * 30 * 30 * 30);
  kCheckCUDA(__LINE__, cudaMemcpy(batch_2D, d_batch_2D, 32 * 3 * 227 * 227 * sizeof(float), cudaMemcpyDeviceToHost));
  kCheckCUDA(__LINE__, cudaMemcpy(batch_3D, d_batch_3D, 32 * 30 * 30 * 30 * sizeof(float), cudaMemcpyDeviceToHost));

  // for (int i = 0; i < 32 * 3 * 227 * 227; i++)
  //   std::cout << "2D batch data: " << batch_2D[i] << std::endl;

  // for (int i = 0; i < 32 * 30 * 30 * 30; i++)
  //   std::cout << "3D batch data: " << batch_3D[i] << std::endl;

  // // Save curr volumes to file
  // int patch3D_size[3];
  // patch3D_size[0] = 30;
  // patch3D_size[1] = 30;
  // patch3D_size[2] = 30;
  // for (int i = 0; i < 32; i++) {
  //   std::string scene_ply_name = "patch." + std::to_string(i) + ".ply";
  //   save_volume_to_ply(scene_ply_name, patch3D_size, &batch_3D[i * 30 * 30 * 30]);
  // }


  return 0;
}

