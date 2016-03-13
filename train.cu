#include "util/util.hpp"
#include <png++/png.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>

////////////////////////////////////////////////////////////////////////////////

void get_frustum_bounds(float* K, std::vector<std::vector<float>> &extrinsic_poses, int base_frame, int curr_frame, float* camera_relative_pose, float* view_bounds,
                        float vox_unit, int* vox_size, float* vox_range_cam) {

  // Use two extrinsic matrices to compute relative rotations between current frame and first frame
  std::vector<float> ex_pose1 = extrinsic_poses[base_frame];
  std::vector<float> ex_pose2 = extrinsic_poses[curr_frame];

  float * ex_mat1 = &ex_pose1[0];
  float * ex_mat2 = &ex_pose2[0];

  float ex_mat1_inv[16] = {0};
  invert_matrix(ex_mat1, ex_mat1_inv);
  multiply_matrix(ex_mat1_inv, ex_mat2, camera_relative_pose);

  // Init cam view frustum
  float max_depth = 0.8;
  float cam_view_frustum[15] =
  { 0, -320 * max_depth / K[0], -320 * max_depth / K[0], 320 * max_depth / K[0],  320 * max_depth / K[0],
    0, -240 * max_depth / K[0],  240 * max_depth / K[0], 240 * max_depth / K[0], -240 * max_depth / K[0],
    0,               max_depth,               max_depth,              max_depth,              max_depth
  };

  // Rotate cam view frustum wrt Rt
  for (int i = 0; i < 5; i++) {
    float tmp_arr[3] = {0};
    tmp_arr[0] = camera_relative_pose[0 * 4 + 0] * cam_view_frustum[0 + i] + camera_relative_pose[0 * 4 + 1] * cam_view_frustum[5 + i] + camera_relative_pose[0 * 4 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[1] = camera_relative_pose[1 * 4 + 0] * cam_view_frustum[0 + i] + camera_relative_pose[1 * 4 + 1] * cam_view_frustum[5 + i] + camera_relative_pose[1 * 4 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[2] = camera_relative_pose[2 * 4 + 0] * cam_view_frustum[0 + i] + camera_relative_pose[2 * 4 + 1] * cam_view_frustum[5 + i] + camera_relative_pose[2 * 4 + 2] * cam_view_frustum[2 * 5 + i];
    cam_view_frustum[0 * 5 + i] = tmp_arr[0] + camera_relative_pose[3];
    cam_view_frustum[1 * 5 + i] = tmp_arr[1] + camera_relative_pose[7];
    cam_view_frustum[2 * 5 + i] = tmp_arr[2] + camera_relative_pose[11];
  }

  // Compute frustum endpoints
  float range2test[3][2] = {0};
  for (int i = 0; i < 3; i++) {
    range2test[i][0] = *std::min_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
    range2test[i][1] = *std::max_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
  }

  // Compute frustum bounds wrt volume
  for (int i = 0; i < 3; i++) {
    view_bounds[i * 2 + 0] = std::max(0.0f, std::floor((range2test[i][0] - vox_range_cam[i * 2 + 0]) / vox_unit));
    view_bounds[i * 2 + 1] = std::min((float)(vox_size[i]), std::ceil((range2test[i][1] - vox_range_cam[i * 2 + 0]) / vox_unit + 1));
  }
}

////////////////////////////////////////////////////////////////////////////////

void save_volume_to_ply(const std::string &file_name, int* vox_size, float* vox_tsdf, float* vox_weight) {
  float tsdf_threshold = 0.2f;
  float weight_threshold = 1.0f;
  // float radius = 5.0f;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
    if (std::abs(vox_tsdf[i]) < tsdf_threshold && vox_weight[i] >= weight_threshold)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(vox_tsdf[i]) < tsdf_threshold && vox_weight[i] >= weight_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (vox_size[0] * vox_size[1]));
      int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
      int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);
    }
  }
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

bool read_depth_data(const std::string &file_name, unsigned short * data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(), png::require_color_space< png::gray_pixel_16 >());
  int index = 0;
  for (int i = 0; i < 480; ++i) {
    for (int j = 0; j < 640; ++j) {
      unsigned short s = img.get_pixel(j, i);
      *(data + index) = s;
      ++index;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////

__global__
void integrate(float* tmp_K, unsigned short* tmp_depth_data, float* tmp_view_bounds, float* tmp_camera_relative_pose,
               float tmp_vox_unit, float tmp_vox_mu, int* tmp_vox_size, float* tmp_vox_range_cam, float* tmp_vox_tsdf, float* tmp_vox_weight) {

  int z = blockIdx.x;
  int y = threadIdx.x;

  if (z < (int)tmp_view_bounds[2 * 2 + 0] || z >= (int)tmp_view_bounds[2 * 2 + 1])
    return;
  if (y < (int)tmp_view_bounds[1 * 2 + 0] || y >= (int)tmp_view_bounds[1 * 2 + 1])
    return;
  for (int x = tmp_view_bounds[0 * 2 + 0]; x < tmp_view_bounds[0 * 2 + 1]; x++) {

    // grid to world coords
    float tmp_pos[3] = {0};
    tmp_pos[0] = (x + 1) * tmp_vox_unit + tmp_vox_range_cam[0 * 2 + 0];
    tmp_pos[1] = (y + 1) * tmp_vox_unit + tmp_vox_range_cam[1 * 2 + 0];
    tmp_pos[2] = (z + 1) * tmp_vox_unit + tmp_vox_range_cam[2 * 2 + 0];

    // transform
    float tmp_arr[3] = {0};
    tmp_arr[0] = tmp_pos[0] - tmp_camera_relative_pose[3];
    tmp_arr[1] = tmp_pos[1] - tmp_camera_relative_pose[7];
    tmp_arr[2] = tmp_pos[2] - tmp_camera_relative_pose[11];
    tmp_pos[0] = tmp_camera_relative_pose[0 * 4 + 0] * tmp_arr[0] + tmp_camera_relative_pose[1 * 4 + 0] * tmp_arr[1] + tmp_camera_relative_pose[2 * 4 + 0] * tmp_arr[2];
    tmp_pos[1] = tmp_camera_relative_pose[0 * 4 + 1] * tmp_arr[0] + tmp_camera_relative_pose[1 * 4 + 1] * tmp_arr[1] + tmp_camera_relative_pose[2 * 4 + 1] * tmp_arr[2];
    tmp_pos[2] = tmp_camera_relative_pose[0 * 4 + 2] * tmp_arr[0] + tmp_camera_relative_pose[1 * 4 + 2] * tmp_arr[1] + tmp_camera_relative_pose[2 * 4 + 2] * tmp_arr[2];
    if (tmp_pos[2] <= 0)
      continue;

    int px = roundf(tmp_K[0] * (tmp_pos[0] / tmp_pos[2]) + tmp_K[2]);
    int py = roundf(tmp_K[4] * (tmp_pos[1] / tmp_pos[2]) + tmp_K[5]);
    if (px < 1 || px > 640 || py < 1 || py > 480)
      continue;

    float p_depth = *(tmp_depth_data + (py - 1) * 640 + (px - 1)) / 1000.f;
    if (p_depth < 0.2 || p_depth > 0.8)
      continue;
    if (roundf(p_depth * 1000.0f) == 0)
      continue;

    float eta = (p_depth - tmp_pos[2]) * sqrtf(1 + powf((tmp_pos[0] / tmp_pos[2]), 2) + powf((tmp_pos[1] / tmp_pos[2]), 2));
    if (eta <= -tmp_vox_mu)
      continue;

    // Integrate
    int volumeIDX = z * tmp_vox_size[0] * tmp_vox_size[1] + y * tmp_vox_size[0] + x;
    float sdf = fmin(1.0f, eta / tmp_vox_mu);
    float w_old = tmp_vox_weight[volumeIDX];
    float w_new = w_old + 1.0f;
    tmp_vox_weight[volumeIDX] = w_new;
    tmp_vox_tsdf[volumeIDX] = (tmp_vox_tsdf[volumeIDX] * w_old + sdf) / w_new;
  }
}

////////////////////////////////////////////////////////////////////////////////

void vol2bin() {
  // Write data to binary file
  // std::string volume_filename = "volume.tsdf.bin";
  // std::ofstream out_file(volume_filename, std::ios::binary | std::ios::out);
  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   out_file.write((char*)&vox_tsdf[i], sizeof(float));
  // out_file.close();
}

////////////////////////////////////////////////////////////////////////////////

void FatalError(const int lineNumber = 0) {
  std::cerr << "FatalError";
  if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
  std::cerr << ". Program Terminated." << std::endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
    FatalError();
  }
}

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
//


////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Fusion: always keep a TSDF volume active in GPU

// TSDF volume in CPU memory
float vox_unit;
float vox_mu_grid;
float vox_mu;
int vox_size[3];
float vox_range_cam[6];
float * vox_tsdf;
float * vox_weight;

// TSDF volume in GPU memory
int * d_vox_size;
float * d_vox_tsdf;
float * d_vox_weight;

// Fusion params in GPU memory
float * d_K;
unsigned short * d_depth_data;
float * d_view_bounds;
float * d_camera_relative_pose;
float * d_vox_range_cam;

// Initialize existing TSDF volume in GPU memory
__global__
void reset_vox_GPU(int* tmp_vox_size, float* tmp_vox_tsdf, float* tmp_vox_weight) {
  int z = blockIdx.x;
  int y = threadIdx.x;
  for (int x = 0; x < tmp_vox_size[0]; x++) {
    tmp_vox_tsdf[z * tmp_vox_size[0] * tmp_vox_size[1] + y * tmp_vox_size[0] + x] = 1.0f;
    tmp_vox_weight[z * tmp_vox_size[0] * tmp_vox_size[1] + y * tmp_vox_size[0] + x] = 0;
  }
}

// Initialize TSDF volume and fusion params
void init_fusion_GPU() {

  // Init voxel volume params
  vox_unit = 0.005;
  vox_mu_grid = 5;
  vox_mu = vox_unit * vox_mu_grid;
  vox_size[0] = 512;
  vox_size[1] = 512;
  vox_size[2] = 512;
  vox_range_cam[0 * 2 + 0] = -(float)(vox_size[0]) * vox_unit / 2;
  vox_range_cam[0 * 2 + 1] = vox_range_cam[0 * 2 + 0] + (float)(vox_size[0]) * vox_unit;
  vox_range_cam[1 * 2 + 0] = -(float)(vox_size[1]) * vox_unit / 2;
  vox_range_cam[1 * 2 + 1] = vox_range_cam[1 * 2 + 0] + (float)(vox_size[1]) * vox_unit;
  vox_range_cam[2 * 2 + 0] = -50.0f * vox_unit;
  vox_range_cam[2 * 2 + 1] = vox_range_cam[2 * 2 + 0] + (float)(vox_size[2]) * vox_unit;
  vox_tsdf = new float[vox_size[0] * vox_size[1] * vox_size[2]];
  vox_weight = new float[vox_size[0] * vox_size[1] * vox_size[2]];
  memset(vox_weight, 0, sizeof(float) * vox_size[0] * vox_size[1] * vox_size[2]);
  memset(vox_tsdf, 0, sizeof(float) * vox_size[0] * vox_size[1] * vox_size[2]);
  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   vox_tsdf[i] = 1.0f;

  // Copy voxel volume to GPU
  cudaMalloc(&d_vox_size, 3 * sizeof(float));
  cudaMalloc(&d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float));
  cudaMalloc(&d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(d_vox_size, vox_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vox_tsdf, vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vox_weight, vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDA(__LINE__, cudaGetLastError());

  // Init volume in GPU
  int CUDA_NUM_BLOCKS = vox_size[2];
  int CUDA_NUM_THREADS = vox_size[1];
  reset_vox_GPU <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_vox_size, d_vox_tsdf, d_vox_weight);
  checkCUDA(__LINE__, cudaGetLastError());

  // Allocate GPU to hold fusion params
  cudaMalloc(&d_K, 9 * sizeof(float));
  cudaMalloc(&d_depth_data, 480 * 640 * sizeof(unsigned short));
  cudaMalloc(&d_view_bounds, 6 * sizeof(float));
  cudaMalloc(&d_camera_relative_pose, 16 * sizeof(float));
  cudaMalloc(&d_vox_range_cam, 6 * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());
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
    tmp_hypothesis_labels[hypothesis_idx] = 1;
  else
    tmp_hypothesis_labels[hypothesis_idx] = 2;

  // Save 2D patch of cube's 2D project to image
  tmp_hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx] = (char)roundf(cube_front_2D[2]);
  tmp_hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx] = (char)roundf(cube_front_2D[6]);
  tmp_hypothesis_crop_2D[2 * num_hypothesis + hypothesis_idx] = (char)roundf(cube_front_2D[1] - cube_front_2D[2]);
  tmp_hypothesis_crop_2D[3 * num_hypothesis + hypothesis_idx] = (char)roundf(cube_front_2D[4] - cube_front_2D[6]);


  // // Create hypothesis cubes that are valid (non-empty and with color)
  // float tsdf_threshold = 0.2f;
  // float cube_incr = 0.01f;
  // int cube_dim = 30;
  // int cube_incr_grid = (int) round(cube_incr / voxel_volume.unit);
  // std::vector<std::vector<int>> valid_cube_loc;
  // for (int x = grid_bounds[0] + cube_dim / 2; x < grid_bounds[1] - cube_dim / 2; x = x + cube_incr_grid)
  //   for (int y = grid_bounds[2] + cube_dim / 2; y < grid_bounds[3] - cube_dim / 2; y = y + cube_incr_grid)
  //     for (int z = grid_bounds[4] + cube_dim / 2; z < grid_bounds[5] - cube_dim / 2; z = z + cube_incr_grid) {

  //       // Get 3D cube
  //       int cube_occ = 0;
  //       // float * curr_cube = new float[30 * 30 * 30];
  //       for (int i = -15; i < 15; i++) {
  //         for (int j = -15; j < 15; j++) {
  //           for (int k = -15; k < 15; k++) {
  //             int volumeIDX = (z + k) * 512 * 512 + (y + j) * 512 + (x + i);
  //             // curr_cube[(k + 15) * 30 * 30 + (j + 15) * 30 + (i + 15)] = voxel_volume.tsdf[volumeIDX];
  //             if (voxel_volume.tsdf[volumeIDX] < tsdf_threshold)
  //               cube_occ++;
  //           }
  //         }
  //       }

  //       // Skip empty cubes
  //       if (cube_occ == 0)
  //         continue;

  //       // Convert cube location from grid to camera coordinates
  //       float x_cam = ((float)x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
  //       float y_cam = ((float)y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
  //       float z_cam = ((float)z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

  //       // If cube 2D projection is not in image bounds, cube is invalid
  //       float cube_rad = ((float) cube_dim) * voxel_volume.unit / 2;
  //       float cube_front[12] = {(x_cam + cube_rad), (x_cam + cube_rad), (x_cam - cube_rad), (x_cam - cube_rad),
  //                               (y_cam + cube_rad), (y_cam - cube_rad), (y_cam - cube_rad), (y_cam + cube_rad),
  //                               (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad)
  //                              };
  //       float cube_front_2D[8] = {};
  //       for (int i = 0; i < 4; i++) {
  //         cube_front_2D[0 * 4 + i] = cube_front[0 * 4 + i] * K[0] / cube_front[2 * 4 + i] + K[2];
  //         cube_front_2D[1 * 4 + i] = cube_front[1 * 4 + i] * K[4] / cube_front[2 * 4 + i] + K[5];
  //       }
  //       for (int i = 0; i < 12; i++)
  //         cube_front_2D[i] = std::round(cube_front_2D[i]);
  //       if (std::min(std::min(cube_front_2D[0], cube_front_2D[1]), std::min(cube_front_2D[2], cube_front_2D[3])) < 0 ||
  //           std::max(std::max(cube_front_2D[0], cube_front_2D[1]), std::max(cube_front_2D[2], cube_front_2D[3])) >= 640 ||
  //           std::min(std::min(cube_front_2D[4], cube_front_2D[5]), std::min(cube_front_2D[6], cube_front_2D[7])) < 0 ||
  //           std::max(std::max(cube_front_2D[4], cube_front_2D[5]), std::max(cube_front_2D[6], cube_front_2D[7])) >= 480)
  //         continue;

  //       // Get 2D patch of cube's 2D project to image
  //       cv::Rect curr_patch_ROI(std::round(cube_front_2D[2]), std::round(cube_front_2D[6]), std::round(cube_front_2D[1] - cube_front_2D[2]), std::round(cube_front_2D[4] - cube_front_2D[6]));
  //       // std::cout << std::round(cube_front_2D[2]) << " " << std::round(cube_front_2D[6]) << " " << std::round(cube_front_2D[1]-cube_front_2D[2]) << " " << std::round(cube_front_2D[4]-cube_front_2D[6]) << std::endl;
  //       cv::Mat curr_patch = curr_image(curr_patch_ROI);
  //       cv::resize(curr_patch, curr_patch, cv::Size(227, 227));

  //       // Check overlap of cube with object bbox
  //       float overlap = 0;
  //       std::vector<float> cube_bbox_grid;
  //       cube_bbox_grid.push_back((float)x - 15); cube_bbox_grid.push_back((float)x + 14);
  //       cube_bbox_grid.push_back((float)y - 15); cube_bbox_grid.push_back((float)y + 14);
  //       cube_bbox_grid.push_back((float)z - 15); cube_bbox_grid.push_back((float)z + 14);
  //       if (((cube_bbox_grid[0] > obj_bbox_grid[0] && cube_bbox_grid[0] < obj_bbox_grid[1]) || (cube_bbox_grid[1] > obj_bbox_grid[0] && cube_bbox_grid[1] < obj_bbox_grid[1])) &&
  //           ((cube_bbox_grid[2] > obj_bbox_grid[2] && cube_bbox_grid[2] < obj_bbox_grid[3]) || (cube_bbox_grid[3] > obj_bbox_grid[2] && cube_bbox_grid[3] < obj_bbox_grid[3])) &&
  //           ((cube_bbox_grid[4] > obj_bbox_grid[4] && cube_bbox_grid[4] < obj_bbox_grid[5]) || (cube_bbox_grid[5] > obj_bbox_grid[4] && cube_bbox_grid[5] < obj_bbox_grid[5]))) {
  //         float overlapX = std::min(std::abs(obj_bbox_grid[0] - cube_bbox_grid[1]), std::abs(obj_bbox_grid[1] - cube_bbox_grid[0]));
  //         float overlapY = std::min(std::abs(obj_bbox_grid[2] - cube_bbox_grid[3]), std::abs(obj_bbox_grid[3] - cube_bbox_grid[2]));
  //         float overlapZ = std::min(std::abs(obj_bbox_grid[4] - cube_bbox_grid[5]), std::abs(obj_bbox_grid[5] - cube_bbox_grid[4]));
  //         float overlap_intersect = overlapX * overlapY * overlapZ;
  //         float obj_bbox_area = (obj_bbox_grid[1] - obj_bbox_grid[0]) * (obj_bbox_grid[3] - obj_bbox_grid[2]) * (obj_bbox_grid[5] - obj_bbox_grid[4]);
  //         overlap = overlap_intersect / obj_bbox_area;
  //       }

  //       // Check distance between center of cube to center of object bbox
  //       float obj_dist = std::sqrt((obj_bbox_mid_cam[0] - x_cam) * (obj_bbox_mid_cam[0] - x_cam) +
  //                                  (obj_bbox_mid_cam[1] - y_cam) * (obj_bbox_mid_cam[1] - y_cam) +
  //                                  (obj_bbox_mid_cam[2] - z_cam) * (obj_bbox_mid_cam[2] - z_cam));

  //       int class_idx = 0;
  //       if (obj_dist < 0.02) {
  //         // std::cout << overlap << std::endl;
  //         // std::string scene_ply_name = "test.ply";
  //         // save_volume_to_ply_highlighted(scene_ply_name, cube_bbox_grid);
  //         class_idx = 1;

  //         // // Show 2D patch
  //         // std::cout << obj_dist << std::endl;
  //         // cv::namedWindow("Patch", CV_WINDOW_AUTOSIZE );
  //         // cv::imshow("Patch", curr_patch);
  //         // cv::waitKey(0);
  //         // std::cout << std::endl;
  //         // std::cout << std::endl;
  //       }

  //       // if (overlap > 0.3 && overlap < 0.7)
  //       //   continue;

  //       // for (int i = 0; i < 8; i++)
  //       //   std::cout << cube_front_2D[i] << " ";
  //       // std::cout << std::endl;
  //       // for (int i = 0; i < 6; i++)
  //       //   std::cout << cube_bbox_grid[i] << " ";
  //       // std::cout << std::endl;

  //       // Class (idx), 2D patch bbox (x1 x2 y1 y2), 3D tsdf bbox (x1 x2 y1 y2 z1 z2)
  //       std::string data_string = std::to_string(class_idx) + " " + std::to_string((int)cube_front_2D[2]) + " " + std::to_string((int)cube_front_2D[1]) + " " +
  //                                 std::to_string((int)cube_front_2D[5]) + " " + std::to_string((int)cube_front_2D[4]) + " " +
  //                                 std::to_string((int)cube_bbox_grid[0] - (int)grid_bounds[0]) + " " + std::to_string((int)cube_bbox_grid[1] - (int)grid_bounds[0]) + " " +
  //                                 std::to_string((int)cube_bbox_grid[2] - (int)grid_bounds[2]) + " " + std::to_string((int)cube_bbox_grid[3] - (int)grid_bounds[2]) + " " +
  //                                 std::to_string((int)cube_bbox_grid[4] - (int)grid_bounds[4]) + " " + std::to_string((int)cube_bbox_grid[5] - (int)grid_bounds[4]);
  //       // std::cout << data_string << std::endl;
  //       crop_data.push_back(data_string);

  //       // // Create name for data point
  //       // std::string hash_id = gen_rand_str(16);
  //       // std::string data_point_name = class_name + "." + hash_id;

  //       // // Save 2D patch to processed folder
  //       // cv::imwrite(processed_data_directory + "/" + data_point_name + ".color.png", curr_patch);

  //       // // Save 3D TSDF cube to processed folder
  //       // // std::string tsdf_filename = processed_data_directory + "/" + data_point_name + ".tsdf.bin";
  //       // std::ofstream tmp_out(processed_data_directory + "/" + data_point_name + ".tsdf.bin", std::ios::binary | std::ios::out);
  //       // for (int i = 0; i < 30*30*30; i++)
  //       //   tmp_out.write((char*)&curr_cube[i], sizeof(float));
  //       // tmp_out.close();

  //       // Clear memory
  //       // delete [] curr_cube;
  //     }
}

__global__
void check_valid_hypothesis_loc(char* tmp_is_grid_loc_valid, int* tmp_vox_size, float* tmp_vox_tsdf) {

  int z = blockIdx.x;
  int y = threadIdx.x;
  if (z < 15 || z >= (tmp_vox_size[2] - 15))
    return;
  if (y < 15 || y >= (tmp_vox_size[1] - 15))
    return;

  float tsdf_threshold = 0.2f;
  int cube_dim = 30;

  for (int x = 15; x < (tmp_vox_size[0] - 15); x++) {
    int loc_idx = blockIdx.x * tmp_vox_size[1] * tmp_vox_size[0] + threadIdx.x * tmp_vox_size[0] + x;

    int cube_occ = 0;
    for (int i = -15; i < 15; i++)
      for (int j = -15; j < 15; j++)
        for (int k = -15; k < 15; k++) {
          int vox_idx = (z + k) * tmp_vox_size[0] * tmp_vox_size[1] + (y + j) * tmp_vox_size[0] + (x + i);
          if (tmp_vox_tsdf[vox_idx] < tsdf_threshold)
            cube_occ++;
        }

    // Non-empty cubes are valid cubes
    if (cube_occ > cube_dim * cube_dim / 2)
      tmp_is_grid_loc_valid[loc_idx] = 1;
  }
}

int main(int argc, char **argv) {

  init_fusion_GPU();

  tic();
  std::string data_directory = "data/train";
  std::string object_name = "glue";
  std::string object_directory = data_directory + "/" + object_name;

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
  std::cout << curr_frame_name << std::endl;

  // Load intrinsics (3x3 matrix)
  std::string intrinsic_filename = curr_sequence_directory + "/intrinsics.K.txt";
  std::vector<float> K_vec = load_matrix_from_file(intrinsic_filename, 3, 3);
  float K[9];
  for (int i = 0; i < 9; i++)
    K[i] = K_vec[i];
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
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      std::cout << object_pose[i * 4 + j] << " ";
    std::cout << std::endl;
  }

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
  for (int i = 0; i < 3; i++)
    std::cout << object_pose_axis[i] << std::endl;

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
  cudaMemcpy(d_K, K, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_depth_data, depth_data, 480 * 640 * sizeof(unsigned short), cudaMemcpyHostToDevice);
  cudaMemcpy(d_view_bounds, view_bounds, 6 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_camera_relative_pose, camera_relative_pose, 16 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vox_range_cam, vox_range_cam, 6 * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDA(__LINE__, cudaGetLastError());

  // Integrate
  int CUDA_NUM_BLOCKS = vox_size[2];
  int CUDA_NUM_THREADS = vox_size[1];
  integrate <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
      vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
  checkCUDA(__LINE__, cudaGetLastError());

  // // Clear memory hold depth frame
  // free(depth_data);

  // // Reset volume in GPU
  // reset_vox_GPU <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_vox_size, d_vox_tsdf, d_vox_weight);
  // checkCUDA(__LINE__, cudaGetLastError());

  toc();

  // Copy data back to memory
  cudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_weight, d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost);
  checkCUDA(__LINE__, cudaGetLastError());

  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   std::cout << vox_tsdf[i] << std::endl;

  // Save curr volume to file
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
  std::cout << grid_bounds[0] << " " << grid_bounds[1] << std::endl;
  std::cout << grid_bounds[2] << " " << grid_bounds[3] << std::endl;
  std::cout << grid_bounds[4] << " " << grid_bounds[5] << std::endl;
  int grid_size[3] = {0};
  grid_size[0] = grid_bounds[1] - grid_bounds[0] + 1;
  grid_size[1] = grid_bounds[3] - grid_bounds[2] + 1;
  grid_size[2] = grid_bounds[5] - grid_bounds[4] + 1;

  // Create list of hypothesis cubes (store grid locations, and is valid or not (0 for invalid, 1 for positive, 2 for negative))
  int num_hypothesis = grid_size[0] * grid_size[1] * grid_size[2];
  std::cout << num_hypothesis << std::endl;
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

  int num_invalid = 0;
  int num_positive = 0;
  int num_negative = 0;
  for (int i = 0; i < num_hypothesis; i++) {
    if (((int)hypothesis_labels[i]) == 0)
      num_invalid++;
    if (((int)hypothesis_labels[i]) == 1)
      num_positive++;
    if (((int)hypothesis_labels[i]) == 2)
      num_negative++;
  }

  std::cout << num_invalid << std::endl;
  std::cout << num_positive << std::endl;
  std::cout << num_negative << std::endl;

  // gen_hypothesis_labels <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_rand_vox_points, d_vox_points_label,
  //   d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
  //     vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);
















  // tic();
  // // Initialize CUDA's random number generator
  // curandGenerator_t generator;
  // std::random_device rd;
  // if ( curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS || curandSetPseudoRandomGeneratorSeed(generator, rd()) != CURAND_STATUS_SUCCESS ) {
  //   std::cerr << "Cannot initialize Curand" << std::endl;
  //   FatalError(__LINE__);
  // }
  // curandState *d_state;
  // cudaMalloc(&d_state, sizeof(curandState));
  // checkCUDA(__LINE__, cudaGetLastError());

  // // Find a negative case
  // int * d_hypothesis_crop_2D_param;
  // float * d_hypothesis_crop_3D;
  // float * d_object_center_cam;

  // cudaMalloc(&d_hypothesis_crop_2D_param, 4 * sizeof(int));
  // cudaMalloc(&d_hypothesis_crop_3D, 30 * 30 * 30 * sizeof(float));
  // cudaMalloc(&d_object_center_cam, 3 * sizeof(float));
  // checkCUDA(__LINE__, cudaGetLastError());

  // cudaMemcpy(d_object_center_cam, object_center_cam, 3 * sizeof(float), cudaMemcpyHostToDevice);
  // checkCUDA(__LINE__, cudaGetLastError());

  // get_negative_hypothesis <<< 1, 1 >>>(false, d_state, d_hypothesis_crop_3D, d_hypothesis_crop_2D_param, d_object_center_cam, d_K, vox_unit, d_vox_size, d_vox_range_cam, d_vox_tsdf);
  // checkCUDA(__LINE__, cudaGetLastError());

  // int hypothesis_crop_2D_param[4] = {0};
  // int hypothesis_crop_3D[4] = {0};
  // cudaMemcpy(hypothesis_crop_2D_param, d_hypothesis_crop_2D_param, 4 * sizeof(int), cudaMemcpyDeviceToHost);
  // // cudaMemcpy(hypothesis_crop_3D, d_hypothesis_crop_3D, 30 * 30 * 30 * sizeof(float), cudaMemcpyDeviceToHost);
  // checkCUDA(__LINE__, cudaGetLastError());
  // for (int i = 0; i < 4; i++)
  //   std::cout << hypothesis_crop_2D_param[i] << std::endl;
  // toc();

















  // bool is_negative_patch_found = false;
  // while (!is_negative_patch_found) {

  //   // Pick random points in voxel volume (each with a binary label for positive/negative case)
  //   tic();
  //   const int num_rand_points = 512 * 100;
  //   int rand_vox_points[3 * num_rand_points] = {0};
  //   std::random_device rd;
  //   std::mt19937 mt(rd());
  //   std::uniform_real_distribution<double> dist_x(15.0f, (float)vox_size[0] - 15.0f - 0.0001);
  //   std::uniform_real_distribution<double> dist_y(15.0f, (float)vox_size[1] - 15.0f - 0.0001);
  //   std::uniform_real_distribution<double> dist_z(15.0f, (float)vox_size[2] - 15.0f - 0.0001);
  //   for (int i = 0; i < num_rand_points; i++) {
  //     rand_vox_points[0 * num_rand_points + i] = (int)floor(dist_x(mt));
  //     rand_vox_points[1 * num_rand_points + i] = (int)floor(dist_y(mt));
  //     rand_vox_points[2 * num_rand_points + i] = (int)floor(dist_z(mt));
  //   }
  //   toc();

  //   // For each point, label if it is valid (1) or invalid case (0) negative case, and save 2D crop
  //   int vox_points_label[num_rand_points] = {0};
  //   int vox_points_crop_2D[4 * num_rand_points] = {0};
  //   int * d_rand_vox_points;
  //   int * d_vox_points_label;
  //   int * d_vox_points_crop_2D;
  //   float * d_object_center_cam;
  //   cudaMalloc(&d_rand_vox_points, 3 * num_rand_points * sizeof(int));
  //   cudaMalloc(&d_vox_points_label, num_rand_points * sizeof(int));
  //   cudaMalloc(&d_vox_points_crop_2D, 4 * num_rand_points * sizeof(int));
  //   cudaMalloc(&d_object_center_cam, 3 * sizeof(float));
  //   checkCUDA(__LINE__, cudaGetLastError());
  //   cudaMemcpy(d_rand_vox_points, rand_vox_points, 3 * num_rand_points * sizeof(int), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_vox_points_label, vox_points_label, num_rand_points * sizeof(int), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_vox_points_crop_2D, vox_points_crop_2D, 4 * num_rand_points * sizeof(int), cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_object_center_cam, object_center_cam, 3 * sizeof(float), cudaMemcpyHostToDevice);
  //   checkCUDA(__LINE__, cudaGetLastError());
  //   CUDA_NUM_THREADS = 512;
  //   CUDA_NUM_BLOCKS = num_rand_points / CUDA_NUM_THREADS;
  //   // gen_hypothesis_labels(num_rand_points, rand_vox_points, vox_points_label, vox_points_crop_2D, object_center_cam, K, vox_unit, vox_size, vox_range_cam, vox_tsdf);
  //   gen_hypothesis_labels <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(num_rand_points, d_rand_vox_points, d_vox_points_label, d_vox_points_crop_2D, d_object_center_cam, d_K, vox_unit, d_vox_size, d_vox_range_cam, d_vox_tsdf);
  //   cudaMemcpy(vox_points_label, d_vox_points_label, num_rand_points * sizeof(int), cudaMemcpyDeviceToHost);
  //   cudaMemcpy(vox_points_crop_2D, d_vox_points_crop_2D, 4 * num_rand_points * sizeof(int), cudaMemcpyDeviceToHost);
  //   // gen_hypothesis_labels <<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_rand_vox_points, d_vox_points_label,
  //   //   d_K, d_depth_data, d_view_bounds, d_camera_relative_pose,
  //   //     vox_unit, vox_mu, d_vox_size, d_vox_range_cam, d_vox_tsdf, d_vox_weight);

  //   cv::namedWindow("Valid Negative Patch", CV_WINDOW_AUTOSIZE);
  //   // cv::rectangle(curr_image, cv::Point(crop_data[best_guess_IDX * 7 + 0], crop_data[best_guess_IDX * 7 + 2]), cv::Point(crop_data[best_guess_IDX * 7 + 1], crop_data[best_guess_IDX * 7 + 3]), cv::Scalar(0, 255, 0));

  //   for (int i = 0; i < num_rand_points; i++) {
  //     if (vox_points_label[i] == 1) {

  //       // Get 2D patch of cube's 2D project to image
  //       cv::Rect curr_patch_ROI(vox_points_crop_2D[0 * num_rand_points + i], vox_points_crop_2D[1 * num_rand_points + i], vox_points_crop_2D[2 * num_rand_points + i], vox_points_crop_2D[3 * num_rand_points + i]);
  //       // std::cout << std::round(cube_front_2D[2]) << " " << std::round(cube_front_2D[6]) << " " << std::round(cube_front_2D[1]-cube_front_2D[2]) << " " << std::round(cube_front_2D[4]-cube_front_2D[6]) << std::endl;
  //       cv::Mat curr_patch = curr_frame_color(curr_patch_ROI);
  //       cv::resize(curr_patch, curr_patch, cv::Size(227, 227));
  //       cv::imshow("Valid Negative Patch", curr_patch);
  //       cv::waitKey(0);
  //     }
  //   }
  // }

  // In GPU memory, crop cube from voxel volume

  // Copy cube to CPU memory














  // std::cout << closest_angle_bin << std::endl;






  // std::string data_directory = "data";
  // std::string sequence_directory = data_directory + "/glue.train.14";
  // std::string processed_data_directory = data_directory + "/train";
  // std::string object_name = "glue";

  // std::cout << sequence_directory << std::endl;

  // sys_command("mkdir -p " + processed_data_directory);

  // // Get file list of color images
  // std::vector<std::string> file_list_color;
  // std::string color_regex = ".color.png";
  // get_files_in_directory(sequence_directory, file_list_color, color_regex);
  // std::sort(file_list_color.begin(), file_list_color.end());

  // // Get file list of depth images
  // std::vector<std::string> file_list_depth;
  // std::string depth_regex = ".depth.png";
  // get_files_in_directory(sequence_directory, file_list_depth, depth_regex);
  // std::sort(file_list_depth.begin(), file_list_depth.end());

  // // Get file list of intrinsics
  // std::vector<std::string> file_list_intrinsics;
  // std::string intrinsics_regex = ".K.txt";
  // get_files_in_directory(sequence_directory, file_list_intrinsics, intrinsics_regex);
  // std::sort(file_list_intrinsics.begin(), file_list_intrinsics.end());

  // // Get file list of extrinsics
  // std::vector<std::string> file_list_extrinsics;
  // std::string extrinsics_regex = ".pose.txt";
  // get_files_in_directory(sequence_directory, file_list_extrinsics, extrinsics_regex);
  // std::sort(file_list_extrinsics.begin(), file_list_extrinsics.end());

  // // Load intrinsics (3x3 matrix)
  // std::string intrinsic_filename = sequence_directory + "/intrinsics.K.txt";
  // std::vector<float> K = load_matrix_from_file(intrinsic_filename, 3, 3);

  // // Load extrinsics (4x4 matrices)
  // std::vector<std::vector<float>> extrinsics;
  // for (std::string &curr_filename : file_list_extrinsics) {
  //   std::string curr_extrinsic_filename = sequence_directory + "/" + curr_filename;
  //   std::vector<float> curr_extrinsic = load_matrix_from_file(curr_extrinsic_filename, 4, 4);
  //   extrinsics.push_back(curr_extrinsic);
  // }

  // // std::cout << sequence_directory + "/" + file_list_color[0] << std::endl;
  // // std::string tmp_filename = sequence_directory + "/" + file_list_color[0];
  // // cv::Mat image;
  // // image = cv::imread( tmp_filename.c_str(), 1 );
  // // cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
  // // cv::imshow("Display Image", image);
  // // cv::waitKey(0);

  // // Init voxel volume params
  // init_voxel_volume();

  // // Set first frame of sequence as base coordinate frame
  // int base_frame = 0;

  // // Fuse frames
  // for (int curr_frame = 0; curr_frame < file_list_depth.size(); curr_frame++) {
  //   std::cerr << "Fusing frame " << curr_frame << "/" << file_list_depth.size() << "...";

  //   // Load image/depth/extrinsic data for current frame
  //   std::string curr_image_filename = sequence_directory + "/" + file_list_color[curr_frame];
  //   cv::Mat curr_image = cv::imread(curr_image_filename.c_str(), 1);
  //   unsigned short * depth_data = (unsigned short *) malloc(kImageRows * kImageCols * sizeof(unsigned short));
  //   std::string curr_depth_filename = sequence_directory + "/" + file_list_depth[curr_frame];
  //   read_depth_data(curr_depth_filename, depth_data);

  //   // Compute relative camera pose transform between current frame and base frame
  //   // Compute camera view frustum bounds within the voxel volume
  //   float camera_relative_pose[16] = {0};
  //   float view_bounds[6] = {0};
  //   // get_frustum_bounds(K, extrinsics, base_frame, curr_frame, camera_relative_pose, view_bounds);
  //   get_frustum_bounds(K, extrinsics, 0, 0, camera_relative_pose, view_bounds); // Note: set relative pose to identity for single frame fusion

  //   // Integrate
  //   integrate(K, depth_data, view_bounds, camera_relative_pose);

  //   // Read bounding box of object
  //   std::string obj_bbox_filename = sequence_directory + "/object.bbox.txt";
  //   std::vector<float> obj_bbox = load_matrix_from_file(obj_bbox_filename, 3, 2);

  //   // Convert bounding box of object from world coordinates to current camera coordinates to grid coordinates
  //   std::vector<float> curr_pose = extrinsics[curr_frame];
  //   std::vector<float> obj_bbox_cam = obj_bbox;
  //   std::vector<float> obj_bbox_grid = obj_bbox_cam;
  //   float * curr_pose_arr = &curr_pose[0];
  //   float curr_pose_inv[16] = {0};
  //   invert_matrix(curr_pose_arr, curr_pose_inv);
  //   for (int i = 0; i < 2; i++) {
  //     float tmp_arr[3] = {0};
  //     tmp_arr[0] = curr_pose_inv[0 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[0 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[0 * 4 + 2] * obj_bbox[2 * 2 + i];
  //     tmp_arr[1] = curr_pose_inv[1 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[1 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[1 * 4 + 2] * obj_bbox[2 * 2 + i];
  //     tmp_arr[2] = curr_pose_inv[2 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[2 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[2 * 4 + 2] * obj_bbox[2 * 2 + i];
  //     obj_bbox_cam[0 * 2 + i] = tmp_arr[0] + curr_pose_inv[3];
  //     obj_bbox_cam[1 * 2 + i] = tmp_arr[1] + curr_pose_inv[7];
  //     obj_bbox_cam[2 * 2 + i] = tmp_arr[2] + curr_pose_inv[11];
  //   }
  //   for (int i = 0; i < 3; i++) {
  //     if (obj_bbox_cam[i * 2 + 0] > obj_bbox_cam[i * 2 + 1]) {
  //       float tmp_swap = obj_bbox_cam[i * 2 + 0];
  //       obj_bbox_cam[i * 2 + 0] = obj_bbox_cam[i * 2 + 1];
  //       obj_bbox_cam[i * 2 + 1] = tmp_swap;
  //     }
  //   }
  //   for (int i = 0; i < 2; i++) {
  //     obj_bbox_grid[0 * 2 + i] = (obj_bbox_cam[0 * 2 + i] - voxel_volume.range[0][0]) / voxel_volume.unit - 1;
  //     obj_bbox_grid[1 * 2 + i] = (obj_bbox_cam[1 * 2 + i] - voxel_volume.range[1][0]) / voxel_volume.unit - 1;
  //     obj_bbox_grid[2 * 2 + i] = (obj_bbox_cam[2 * 2 + i] - voxel_volume.range[2][0]) / voxel_volume.unit - 1;
  //   }

  //   // Compute center of object's bounding box
  //   std::vector<float> obj_bbox_mid_cam;
  //   obj_bbox_mid_cam.push_back((obj_bbox_cam[0] + obj_bbox_cam[1]) / 2);
  //   obj_bbox_mid_cam.push_back((obj_bbox_cam[2] + obj_bbox_cam[3]) / 2);
  //   obj_bbox_mid_cam.push_back((obj_bbox_cam[4] + obj_bbox_cam[5]) / 2);

  //   // Get bounds of TSDF volume
  //   float grid_bounds[6] = {0};
  //   grid_bounds[0] = 512; grid_bounds[2] = 512; grid_bounds[4] = 1024;
  //   for (int i = 0; i < 512 * 512 * 1024; i++) {
  //     if (std::abs(voxel_volume.tsdf[i]) < 1.0f) {
  //       float z = (float) (floor(i / (512 * 512)));
  //       float y = (float) (floor((i - (z * 512 * 512)) / 512));
  //       float x = (float) (i - (z * 512 * 512) - (y * 512));
  //       grid_bounds[0] = std::min(x, grid_bounds[0]); grid_bounds[1] = std::max(x, grid_bounds[1]);
  //       grid_bounds[2] = std::min(y, grid_bounds[2]); grid_bounds[3] = std::max(y, grid_bounds[3]);
  //       grid_bounds[4] = std::min(z, grid_bounds[4]); grid_bounds[5] = std::max(z, grid_bounds[5]);
  //     }
  //   }

  //   // cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
  //   // cv::imshow("Display Image", curr_image);
  //   // cv::waitKey(0);

  //   // Data structure to store crop information of hypothesis cubes
  //   std::vector<std::string> crop_data;

  //   // Create hypothesis cubes that are valid (non-empty and with color)
  //   float tsdf_threshold = 0.2f;
  //   float cube_incr = 0.01f;
  //   int cube_dim = 30;
  //   int cube_incr_grid = (int) round(cube_incr / voxel_volume.unit);
  //   std::vector<std::vector<int>> valid_cube_loc;
  //   for (int x = grid_bounds[0] + cube_dim / 2; x < grid_bounds[1] - cube_dim / 2; x = x + cube_incr_grid)
  //     for (int y = grid_bounds[2] + cube_dim / 2; y < grid_bounds[3] - cube_dim / 2; y = y + cube_incr_grid)
  //       for (int z = grid_bounds[4] + cube_dim / 2; z < grid_bounds[5] - cube_dim / 2; z = z + cube_incr_grid) {

  //         // Get 3D cube
  //         int cube_occ = 0;
  //         // float * curr_cube = new float[30 * 30 * 30];
  //         for (int i = -15; i < 15; i++) {
  //           for (int j = -15; j < 15; j++) {
  //             for (int k = -15; k < 15; k++) {
  //               int volumeIDX = (z + k) * 512 * 512 + (y + j) * 512 + (x + i);
  //               // curr_cube[(k + 15) * 30 * 30 + (j + 15) * 30 + (i + 15)] = voxel_volume.tsdf[volumeIDX];
  //               if (voxel_volume.tsdf[volumeIDX] < tsdf_threshold)
  //                 cube_occ++;
  //             }
  //           }
  //         }

  //         // Skip empty cubes
  //         if (cube_occ == 0)
  //           continue;

  //         // Convert cube location from grid to camera coordinates
  //         float x_cam = ((float)x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
  //         float y_cam = ((float)y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
  //         float z_cam = ((float)z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

  //         // If cube 2D projection is not in image bounds, cube is invalid
  //         float cube_rad = ((float) cube_dim) * voxel_volume.unit / 2;
  //         float cube_front[12] = {(x_cam + cube_rad), (x_cam + cube_rad), (x_cam - cube_rad), (x_cam - cube_rad),
  //                                 (y_cam + cube_rad), (y_cam - cube_rad), (y_cam - cube_rad), (y_cam + cube_rad),
  //                                 (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad)
  //                                };
  //         float cube_front_2D[8] = {};
  //         for (int i = 0; i < 4; i++) {
  //           cube_front_2D[0 * 4 + i] = cube_front[0 * 4 + i] * K[0] / cube_front[2 * 4 + i] + K[2];
  //           cube_front_2D[1 * 4 + i] = cube_front[1 * 4 + i] * K[4] / cube_front[2 * 4 + i] + K[5];
  //         }
  //         for (int i = 0; i < 12; i++)
  //           cube_front_2D[i] = std::round(cube_front_2D[i]);
  //         if (std::min(std::min(cube_front_2D[0], cube_front_2D[1]), std::min(cube_front_2D[2], cube_front_2D[3])) < 0 ||
  //             std::max(std::max(cube_front_2D[0], cube_front_2D[1]), std::max(cube_front_2D[2], cube_front_2D[3])) >= 640 ||
  //             std::min(std::min(cube_front_2D[4], cube_front_2D[5]), std::min(cube_front_2D[6], cube_front_2D[7])) < 0 ||
  //             std::max(std::max(cube_front_2D[4], cube_front_2D[5]), std::max(cube_front_2D[6], cube_front_2D[7])) >= 480)
  //           continue;

  //         // Get 2D patch of cube's 2D project to image
  //         cv::Rect curr_patch_ROI(std::round(cube_front_2D[2]), std::round(cube_front_2D[6]), std::round(cube_front_2D[1] - cube_front_2D[2]), std::round(cube_front_2D[4] - cube_front_2D[6]));
  //         // std::cout << std::round(cube_front_2D[2]) << " " << std::round(cube_front_2D[6]) << " " << std::round(cube_front_2D[1]-cube_front_2D[2]) << " " << std::round(cube_front_2D[4]-cube_front_2D[6]) << std::endl;
  //         cv::Mat curr_patch = curr_image(curr_patch_ROI);
  //         cv::resize(curr_patch, curr_patch, cv::Size(227, 227));

  //         // Check overlap of cube with object bbox
  //         float overlap = 0;
  //         std::vector<float> cube_bbox_grid;
  //         cube_bbox_grid.push_back((float)x - 15); cube_bbox_grid.push_back((float)x + 14);
  //         cube_bbox_grid.push_back((float)y - 15); cube_bbox_grid.push_back((float)y + 14);
  //         cube_bbox_grid.push_back((float)z - 15); cube_bbox_grid.push_back((float)z + 14);
  //         if (((cube_bbox_grid[0] > obj_bbox_grid[0] && cube_bbox_grid[0] < obj_bbox_grid[1]) || (cube_bbox_grid[1] > obj_bbox_grid[0] && cube_bbox_grid[1] < obj_bbox_grid[1])) &&
  //             ((cube_bbox_grid[2] > obj_bbox_grid[2] && cube_bbox_grid[2] < obj_bbox_grid[3]) || (cube_bbox_grid[3] > obj_bbox_grid[2] && cube_bbox_grid[3] < obj_bbox_grid[3])) &&
  //             ((cube_bbox_grid[4] > obj_bbox_grid[4] && cube_bbox_grid[4] < obj_bbox_grid[5]) || (cube_bbox_grid[5] > obj_bbox_grid[4] && cube_bbox_grid[5] < obj_bbox_grid[5]))) {
  //           float overlapX = std::min(std::abs(obj_bbox_grid[0] - cube_bbox_grid[1]), std::abs(obj_bbox_grid[1] - cube_bbox_grid[0]));
  //           float overlapY = std::min(std::abs(obj_bbox_grid[2] - cube_bbox_grid[3]), std::abs(obj_bbox_grid[3] - cube_bbox_grid[2]));
  //           float overlapZ = std::min(std::abs(obj_bbox_grid[4] - cube_bbox_grid[5]), std::abs(obj_bbox_grid[5] - cube_bbox_grid[4]));
  //           float overlap_intersect = overlapX * overlapY * overlapZ;
  //           float obj_bbox_area = (obj_bbox_grid[1] - obj_bbox_grid[0]) * (obj_bbox_grid[3] - obj_bbox_grid[2]) * (obj_bbox_grid[5] - obj_bbox_grid[4]);
  //           overlap = overlap_intersect / obj_bbox_area;
  //         }

  //         // Check distance between center of cube to center of object bbox
  //         float obj_dist = std::sqrt((obj_bbox_mid_cam[0] - x_cam) * (obj_bbox_mid_cam[0] - x_cam) +
  //                                    (obj_bbox_mid_cam[1] - y_cam) * (obj_bbox_mid_cam[1] - y_cam) +
  //                                    (obj_bbox_mid_cam[2] - z_cam) * (obj_bbox_mid_cam[2] - z_cam));

  //         int class_idx = 0;
  //         if (obj_dist < 0.02) {
  //           // std::cout << overlap << std::endl;
  //           // std::string scene_ply_name = "test.ply";
  //           // save_volume_to_ply_highlighted(scene_ply_name, cube_bbox_grid);
  //           class_idx = 1;

  //           // // Show 2D patch
  //           // std::cout << obj_dist << std::endl;
  //           // cv::namedWindow("Patch", CV_WINDOW_AUTOSIZE );
  //           // cv::imshow("Patch", curr_patch);
  //           // cv::waitKey(0);
  //           // std::cout << std::endl;
  //           // std::cout << std::endl;
  //         }

  //         // if (overlap > 0.3 && overlap < 0.7)
  //         //   continue;

  //         // for (int i = 0; i < 8; i++)
  //         //   std::cout << cube_front_2D[i] << " ";
  //         // std::cout << std::endl;
  //         // for (int i = 0; i < 6; i++)
  //         //   std::cout << cube_bbox_grid[i] << " ";
  //         // std::cout << std::endl;

  //         // Class (idx), 2D patch bbox (x1 x2 y1 y2), 3D tsdf bbox (x1 x2 y1 y2 z1 z2)
  //         std::string data_string = std::to_string(class_idx) + " " + std::to_string((int)cube_front_2D[2]) + " " + std::to_string((int)cube_front_2D[1]) + " " +
  //                                   std::to_string((int)cube_front_2D[5]) + " " + std::to_string((int)cube_front_2D[4]) + " " +
  //                                   std::to_string((int)cube_bbox_grid[0] - (int)grid_bounds[0]) + " " + std::to_string((int)cube_bbox_grid[1] - (int)grid_bounds[0]) + " " +
  //                                   std::to_string((int)cube_bbox_grid[2] - (int)grid_bounds[2]) + " " + std::to_string((int)cube_bbox_grid[3] - (int)grid_bounds[2]) + " " +
  //                                   std::to_string((int)cube_bbox_grid[4] - (int)grid_bounds[4]) + " " + std::to_string((int)cube_bbox_grid[5] - (int)grid_bounds[4]);
  //         // std::cout << data_string << std::endl;
  //         crop_data.push_back(data_string);

  //         // // Create name for data point
  //         // std::string hash_id = gen_rand_str(16);
  //         // std::string data_point_name = class_name + "." + hash_id;

  //         // // Save 2D patch to processed folder
  //         // cv::imwrite(processed_data_directory + "/" + data_point_name + ".color.png", curr_patch);

  //         // // Save 3D TSDF cube to processed folder
  //         // // std::string tsdf_filename = processed_data_directory + "/" + data_point_name + ".tsdf.bin";
  //         // std::ofstream tmp_out(processed_data_directory + "/" + data_point_name + ".tsdf.bin", std::ios::binary | std::ios::out);
  //         // for (int i = 0; i < 30*30*30; i++)
  //         //   tmp_out.write((char*)&curr_cube[i], sizeof(float));
  //         // tmp_out.close();

  //         // Clear memory
  //         // delete [] curr_cube;
  //       }

  //   // Give current frame a hash id
  //   std::string curr_frame_hash = gen_rand_str(16);

  //   // Save image
  //   std::string curr_frame_color_filename = processed_data_directory + "/" + curr_frame_hash + ".color.png";
  //   cv::imwrite(curr_frame_color_filename, curr_image);

  //   // Save volume
  //   std::string curr_frame_tsdf_filename = processed_data_directory + "/" + curr_frame_hash + "." + std::to_string((int)grid_bounds[1] - (int)grid_bounds[0]) + "." + std::to_string((int)grid_bounds[3] - (int)grid_bounds[2]) + "." + std::to_string((int)grid_bounds[5] - (int)grid_bounds[4]) + ".tsdf.bin";
  //   std::ofstream tmp_out(curr_frame_tsdf_filename, std::ios::binary | std::ios::out);
  //   for (int z = grid_bounds[4]; z < grid_bounds[5]; z++)
  //     for (int y = grid_bounds[2]; y < grid_bounds[3]; y++)
  //       for (int x = grid_bounds[0]; x < grid_bounds[1]; x++) {
  //         int volumeIDX = z * 512 * 512 + y * 512 + x;
  //         tmp_out.write((char*)&voxel_volume.tsdf[volumeIDX], sizeof(float));
  //       }
  //   tmp_out.close();

  //   // Save crop information
  //   std::string curr_frame_crop_filename = processed_data_directory + "/" + curr_frame_hash + "." + std::to_string(crop_data.size()) + ".crop.txt";
  //   FILE *fp = fopen(curr_frame_crop_filename.c_str(), "w");
  //   for (int i = 0; i < crop_data.size(); i++)
  //     fprintf(fp, "%s\n", crop_data[i].c_str());
  //   fclose(fp);


  //   // Clear memory
  //   free(depth_data);
  //   std::cerr << " done!" << std::endl;

  //   // Save curr volume to file
  //   // std::string scene_ply_name = "test.ply";
  //   // save_volume_to_ply(scene_ply_name);
  //   // save_volume_to_ply_highlighted(scene_ply_name, obj_bbox_grid);

  //   // Init new volume
  //   memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
  //   for (int i = 0; i < 512 * 512 * 1024; i++)
  //     voxel_volume.tsdf[i] = 1.0f;

  // }
  return 0;
}

