#include "util/util.hpp"
#include <png++/png.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense/rs.hpp>
#include "display.hpp"

// Load a MxN matrix from a text file
std::vector<float> load_matrix_from_file(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}



const int kImageRows = 480;
const int kImageCols = 640;
const int kSampleFactor = 30;
const int kImageChannels = 3;

// struct _cam_k {
//   float fx;
//   float fy;
//   float cx;
//   float cy;
// } cam_K;

struct _voxel_volume {
  float unit;
  float mu_grid;
  float mu;
  float size_grid[3];
  float range[3][2];
  float* tsdf;
  float* weight;
} voxel_volume;

void init_voxel_volume() {
  voxel_volume.unit = 0.005;
  voxel_volume.mu_grid = 5;
  voxel_volume.mu = voxel_volume.unit * voxel_volume.mu_grid;

  voxel_volume.size_grid[0] = 512;
  voxel_volume.size_grid[1] = 512;
  voxel_volume.size_grid[2] = 1024;

  voxel_volume.range[0][0] = -voxel_volume.size_grid[0] * voxel_volume.unit / 2;
  voxel_volume.range[0][1] = voxel_volume.range[0][0] + voxel_volume.size_grid[0] * voxel_volume.unit;
  voxel_volume.range[1][0] = -voxel_volume.size_grid[1] * voxel_volume.unit / 2;
  voxel_volume.range[1][1] = voxel_volume.range[1][0] + voxel_volume.size_grid[1] * voxel_volume.unit;
  voxel_volume.range[2][0] = -50.0f * voxel_volume.unit;
  voxel_volume.range[2][1] = voxel_volume.range[2][0] + voxel_volume.size_grid[2] * voxel_volume.unit;

  // std::cout << voxel_volume.range[0][0] << std::endl;
  // std::cout << voxel_volume.range[1][0] << std::endl;
  // std::cout << voxel_volume.range[2][0] << std::endl;
  // std::cout << voxel_volume.range[0][1] << std::endl;
  // std::cout << voxel_volume.range[1][1] << std::endl;
  // std::cout << voxel_volume.range[2][1] << std::endl;

  voxel_volume.tsdf = new float[512 * 512 * 1024];
  voxel_volume.weight = new float[512 * 512 * 1024];
  memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
  for (int i = 0; i < 512 * 512 * 1024; i++)
    voxel_volume.tsdf[i] = 1.0f;

}

////////////////////////////////////////////////////////////////////////////////

void get_frustum_bounds(std::vector<float> &K, std::vector<std::vector<float>> &extrinsic_poses, int base_frame, int curr_frame, float* camera_relative_pose, float* view_bounds) {

  // if (curr_frame == 0) {

  //   // Relative rotation of first frame to first frame is identity matrix
  //   cam_R[0] = 1.0f;
  //   cam_R[1] = 0.0f;
  //   cam_R[2] = 0.0f;
  //   cam_R[3] = 0.0f;
  //   cam_R[4] = 1.0f;
  //   cam_R[5] = 0.0f;
  //   cam_R[6] = 0.0f;
  //   cam_R[7] = 0.0f;
  //   cam_R[8] = 1.0f;

  // } else {

  // Use two extrinsic matrices to compute relative rotations between current frame and first frame
  std::vector<float> ex_pose1 = extrinsic_poses[base_frame];
  std::vector<float> ex_pose2 = extrinsic_poses[curr_frame];


  float * ex_mat1 = &ex_pose1[0];
  float * ex_mat2 = &ex_pose2[0];

  // float ex_mat1[16] =
  // { ex_pose1.R[0 * 3 + 0], ex_pose1.R[0 * 3 + 1], ex_pose1.R[0 * 3 + 2], ex_pose1.t[0],
  //   ex_pose1.R[1 * 3 + 0], ex_pose1.R[1 * 3 + 1], ex_pose1.R[1 * 3 + 2], ex_pose1.t[1],
  //   ex_pose1.R[2 * 3 + 0], ex_pose1.R[2 * 3 + 1], ex_pose1.R[2 * 3 + 2], ex_pose1.t[2],
  //   0,                0,                0,            1
  // };

  // float ex_mat2[16] =
  // { ex_pose2.R[0 * 3 + 0], ex_pose2.R[0 * 3 + 1], ex_pose2.R[0 * 3 + 2], ex_pose2.t[0],
  //   ex_pose2.R[1 * 3 + 0], ex_pose2.R[1 * 3 + 1], ex_pose2.R[1 * 3 + 2], ex_pose2.t[1],
  //   ex_pose2.R[2 * 3 + 0], ex_pose2.R[2 * 3 + 1], ex_pose2.R[2 * 3 + 2], ex_pose2.t[2],
  //   0,                 0,                0,            1
  // };

  float ex_mat1_inv[16] = {0};
  invert_matrix(ex_mat1, ex_mat1_inv);
  // float camera_relative_pose[16] = {0};
  multiply_matrix(ex_mat1_inv, ex_mat2, camera_relative_pose);

  // cam_R[0] = ex_mat_rel[0];
  // cam_R[1] = ex_mat_rel[1];
  // cam_R[2] = ex_mat_rel[2];
  // cam_R[3] = ex_mat_rel[4];
  // cam_R[4] = ex_mat_rel[5];
  // cam_R[5] = ex_mat_rel[6];
  // cam_R[6] = ex_mat_rel[8];
  // cam_R[7] = ex_mat_rel[9];
  // cam_R[8] = ex_mat_rel[10];

  // cam_t[0] = ex_mat_rel[3];
  // cam_t[1] = ex_mat_rel[7];
  // cam_t[2] = ex_mat_rel[11];

  // }

  // std::cout << cam_R[0] << std::endl;
  // std::cout << cam_R[1] << std::endl;
  // std::cout << cam_R[2] << std::endl;
  // std::cout << cam_R[3] << std::endl;
  // std::cout << cam_R[4] << std::endl;
  // std::cout << cam_R[5] << std::endl;
  // std::cout << cam_R[6] << std::endl;
  // std::cout << cam_R[7] << std::endl;
  // std::cout << cam_R[8] << std::endl;

  // std::cout << cam_t[0] << std::endl;
  // std::cout << cam_t[1] << std::endl;
  // std::cout << cam_t[2] << std::endl;

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

  // std::cout << cam_view_frustum[0*5+0] << std::endl;
  // std::cout << cam_view_frustum[0*5+1] << std::endl;
  // std::cout << cam_view_frustum[0*5+2] << std::endl;
  // std::cout << cam_view_frustum[0*5+3] << std::endl;
  // std::cout << cam_view_frustum[0*5+4] << std::endl;

  // std::cout << cam_view_frustum[1*5+0] << std::endl;
  // std::cout << cam_view_frustum[1*5+1] << std::endl;
  // std::cout << cam_view_frustum[1*5+2] << std::endl;
  // std::cout << cam_view_frustum[1*5+3] << std::endl;
  // std::cout << cam_view_frustum[1*5+4] << std::endl;

  // std::cout << cam_view_frustum[2*5+0] << std::endl;
  // std::cout << cam_view_frustum[2*5+1] << std::endl;
  // std::cout << cam_view_frustum[2*5+2] << std::endl;
  // std::cout << cam_view_frustum[2*5+3] << std::endl;
  // std::cout << cam_view_frustum[2*5+4] << std::endl;

  // Compute frustum endpoints
  float range2test[3][2] = {0};
  for (int i = 0; i < 3; i++) {
    range2test[i][0] = *std::min_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
    range2test[i][1] = *std::max_element(&cam_view_frustum[i * 5], &cam_view_frustum[i * 5] + 5);
  }

  // std::cout << range2test[0][0] << std::endl;
  // std::cout << range2test[1][0] << std::endl;
  // std::cout << range2test[2][0] << std::endl;

  // std::cout << range2test[0][1] << std::endl;
  // std::cout << range2test[1][1] << std::endl;
  // std::cout << range2test[2][1] << std::endl;

  // Compute frustum bounds wrt volume
  for (int i = 0; i < 3; i++) {
    view_bounds[i * 2 + 0] = std::max(0.0f, std::floor((range2test[i][0] - voxel_volume.range[i][0]) / voxel_volume.unit));
    view_bounds[i * 2 + 1] = std::min(voxel_volume.size_grid[i], std::ceil((range2test[i][1] - voxel_volume.range[i][0]) / voxel_volume.unit + 1));
  }

  // std::cout << std::endl;
  // std::cout << view_bounds[0*2+0] << std::endl;
  // std::cout << view_bounds[0*2+1] << std::endl;
  // std::cout << view_bounds[1*2+0] << std::endl;
  // std::cout << view_bounds[1*2+1] << std::endl;
  // std::cout << view_bounds[2*2+0] << std::endl;
  // std::cout << view_bounds[2*2+1] << std::endl;

}

////////////////////////////////////////////////////////////////////////////////

void save_volume_to_ply(const std::string &file_name) {
  float tsdf_threshold = 0.2f;
  float weight_threshold = 1.0f;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] >= weight_threshold)
      num_points++;
  std::cout << num_points << std::endl;

  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < 512 * 512 * 1024; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] >= weight_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (512 * 512));
      int y = floor((i - (z * 512 * 512)) / 512);
      int x = i - (z * 512 * 512) - (y * 512);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      unsigned char r = (unsigned char) 180;
      unsigned char g = (unsigned char) 180;
      unsigned char b = (unsigned char) 180;
      fwrite(&r, sizeof(unsigned char), 1, fp);
      fwrite(&g, sizeof(unsigned char), 1, fp);
      fwrite(&b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void save_volume_to_ply_highlighted(const std::string &file_name, std::vector<float> &bbox) {
  float tsdf_threshold = 0.2f;
  float weight_threshold = 1.0f;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] >= weight_threshold)
      num_points++;
  std::cout << num_points << std::endl;

  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < 512 * 512 * 1024; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] >= weight_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (512 * 512));
      int y = floor((i - (z * 512 * 512)) / 512);
      int x = i - (z * 512 * 512) - (y * 512);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      unsigned char r = (unsigned char) 180;
      unsigned char g = (unsigned char) 180;
      unsigned char b = (unsigned char) 180;
      if (x > bbox[0] && x < bbox[1] &&
          y > bbox[2] && y < bbox[3] &&
          z > bbox[4] && z < bbox[5]) {
        r = (unsigned char) 255;
        g = (unsigned char) 0;
        b = (unsigned char) 0;
      }

      fwrite(&r, sizeof(unsigned char), 1, fp);
      fwrite(&g, sizeof(unsigned char), 1, fp);
      fwrite(&b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

bool read_depth_data(const std::string &file_name, unsigned short * data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(), png::require_color_space< png::gray_pixel_16 >());
  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      unsigned short s = img.get_pixel(j, i);
      *(data + index) = s;
      ++index;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////

void integrate(std::vector<float> &K, ushort* depth_data, float* view_bounds, float* camera_relative_pose) {

  for (int z = view_bounds[2 * 2 + 0]; z < view_bounds[2 * 2 + 1]; z++) {
    for (int y = view_bounds[1 * 2 + 0]; y < view_bounds[1 * 2 + 1]; y++) {
      for (int x = view_bounds[0 * 2 + 0]; x < view_bounds[0 * 2 + 1]; x++) {

        // grid to world coords
        float tmp_pos[3] = {0};
        tmp_pos[0] = (x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
        tmp_pos[1] = (y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
        tmp_pos[2] = (z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

        // std::cout << tmp_pos[0] << " " << tmp_pos[1] << " " << tmp_pos[2] << std::endl;

        // transform
        float tmp_arr[3] = {0};
        tmp_arr[0] = tmp_pos[0] - camera_relative_pose[3];
        tmp_arr[1] = tmp_pos[1] - camera_relative_pose[7];
        tmp_arr[2] = tmp_pos[2] - camera_relative_pose[11];
        tmp_pos[0] = camera_relative_pose[0 * 4 + 0] * tmp_arr[0] + camera_relative_pose[1 * 4 + 0] * tmp_arr[1] + camera_relative_pose[2 * 4 + 0] * tmp_arr[2];
        tmp_pos[1] = camera_relative_pose[0 * 4 + 1] * tmp_arr[0] + camera_relative_pose[1 * 4 + 1] * tmp_arr[1] + camera_relative_pose[2 * 4 + 1] * tmp_arr[2];
        tmp_pos[2] = camera_relative_pose[0 * 4 + 2] * tmp_arr[0] + camera_relative_pose[1 * 4 + 2] * tmp_arr[1] + camera_relative_pose[2 * 4 + 2] * tmp_arr[2];
        // std::cout << tmp_pos[0] << " " << tmp_pos[1] << " " << tmp_pos[2] << std::endl << std::endl;
        if (tmp_pos[2] <= 0)
          continue;

        int px = std::round(K[0] * (tmp_pos[0] / tmp_pos[2]) + K[2]);
        int py = std::round(K[4] * (tmp_pos[1] / tmp_pos[2]) + K[5]);
        if (px < 1 || px > 640 || py < 1 || py > 480)
          continue;

        float p_depth = *(depth_data + (py - 1) * kImageCols + (px - 1)) / 1000.f;
        if (p_depth < 0.2 || p_depth > 0.8)
          continue;
        if (std::round(p_depth * 1000.0f) == 0)
          continue;

        float eta = (p_depth - tmp_pos[2]) * sqrt(1 + pow((tmp_pos[0] / tmp_pos[2]), 2) + pow((tmp_pos[1] / tmp_pos[2]), 2));
        if (eta <= -voxel_volume.mu)
          continue;

        // Integrate
        int volumeIDX = z * 512 * 512 + y * 512 + x;
        float sdf = std::min(1.0f, eta / voxel_volume.mu);
        float w_old = voxel_volume.weight[volumeIDX];
        float w_new = w_old + 1.0f;
        voxel_volume.weight[volumeIDX] = w_new;
        voxel_volume.tsdf[volumeIDX] = (voxel_volume.tsdf[volumeIDX] * w_old + sdf) / w_new;

      }
    }
  }

}

void test() {

  // std::string target_color_filename = "data/test1/frame-000008.color.png";
  // std::string target_depth_filename = "data/test1/frame-000008.depth.png";
  // std::string target_intrinsics_filename = "data/test1/intrinsics.K.txt";
  std::string target_color_filename = "TMP.frame.color.png";
  std::string target_depth_filename = "TMP.frame.depth.png";
  std::string target_intrinsics_filename = "TMP.intrinsics.K.txt";
  // std::string target_color_filename = "data/644giw7d86ii83fu/frame-000270.color.png";
  // std::string target_depth_filename = "data/644giw7d86ii83fu/frame-000270.depth.png";
  // std::string target_intrinsics_filename = "data/644giw7d86ii83fu/intrinsics.K.txt";
  

  // Load intrinsics (3x3 matrix)
  std::vector<float> K = load_matrix_from_file(target_intrinsics_filename, 3, 3);

  // Init voxel volume params
  std::cerr << "Creating projective TSDF volume...";
  init_voxel_volume();

  // Load image/depth/extrinsic data for current frame
  cv::Mat curr_image = cv::imread(target_color_filename.c_str(), 1);
  unsigned short * depth_data = (unsigned short *) malloc(kImageRows * kImageCols * sizeof(unsigned short));
  read_depth_data(target_depth_filename, depth_data);

  // Compute relative camera pose transform between current frame and base frame
  // Compute camera view frustum bounds within the voxel volume
  float camera_relative_pose[16] = {0};
  float view_bounds[6] = {0};
  // get_frustum_bounds(K, extrinsics, base_frame, curr_frame, camera_relative_pose, view_bounds);
  std::vector<float> curr_extrinsic;
  for (int i = 0; i < 3; i++) {
    curr_extrinsic.push_back(1.0f); curr_extrinsic.push_back(0.0f); curr_extrinsic.push_back(0.0f); curr_extrinsic.push_back(0.0f); curr_extrinsic.push_back(0.0f);
  }
  curr_extrinsic.push_back(1.0f);
  std::vector<std::vector<float>> extrinsics; extrinsics.push_back(curr_extrinsic);
  get_frustum_bounds(K, extrinsics, 0, 0, camera_relative_pose, view_bounds); // Note: set relative pose to identity for single frame fusion

  // Integrate
  integrate(K, depth_data, view_bounds, camera_relative_pose);

  // Get bounds of TSDF volume
  float grid_bounds[6] = {0};
  grid_bounds[0] = 512; grid_bounds[2] = 512; grid_bounds[4] = 1024;
  for (int i = 0; i < 512 * 512 * 1024; i++) {
    if (std::abs(voxel_volume.tsdf[i]) < 1.0f) {
      float z = (float) (floor(i / (512 * 512)));
      float y = (float) (floor((i - (z * 512 * 512)) / 512));
      float x = (float) (i - (z * 512 * 512) - (y * 512));
      grid_bounds[0] = std::min(x, grid_bounds[0]); grid_bounds[1] = std::max(x, grid_bounds[1]);
      grid_bounds[2] = std::min(y, grid_bounds[2]); grid_bounds[3] = std::max(y, grid_bounds[3]);
      grid_bounds[4] = std::min(z, grid_bounds[4]); grid_bounds[5] = std::max(z, grid_bounds[5]);
    }
  }
  std::cerr << " done." << std::endl;

  // cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
  // cv::imshow("Display Image", curr_image);
  // cv::waitKey(0);

  // Init tensor files for marvin
  std::string data2D_tensor_filename = "TMPdata2D.tensor";
  std::string data3D_tensor_filename = "TMPdata3D.tensor";
  std::string label_tensor_filename = "TMPlabels.tensor";
  FILE *data2D_tensor_fp = fopen(data2D_tensor_filename.c_str(), "w");
  FILE *data3D_tensor_fp = fopen(data3D_tensor_filename.c_str(), "w");
  FILE *label_tensor_fp = fopen(label_tensor_filename.c_str(), "w");

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
  uint32_t data_size = (uint32_t)0;
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
  data_size = (uint32_t)0;
  fwrite((void*)&data_size, sizeof(uint32_t), 1, data3D_tensor_fp);
  data_chan = (uint32_t)1;
  fwrite((void*)&data_chan, sizeof(uint32_t), 1, data3D_tensor_fp);
  uint32_t volume_dim = (uint32_t)30;
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);
  fwrite((void*)&volume_dim, sizeof(uint32_t), 1, data3D_tensor_fp);

  // Write label header
  type_id = (uint8_t)1;
  fwrite((void*)&type_id, sizeof(uint8_t), 1, label_tensor_fp);
  size_of = (uint32_t)4;
  fwrite((void*)&size_of, sizeof(uint32_t), 1, label_tensor_fp);
  str_len = (uint32_t)6;
  fwrite((void*)&str_len, sizeof(uint32_t), 1, label_tensor_fp);
  fprintf(label_tensor_fp, "labels");
  data_dim = (uint32_t)5;
  fwrite((void*)&data_dim, sizeof(uint32_t), 1, label_tensor_fp);
  data_size = (uint32_t)0;
  fwrite((void*)&data_size, sizeof(uint32_t), 1, label_tensor_fp);
  data_chan = (uint32_t)1;
  fwrite((void*)&data_chan, sizeof(uint32_t), 1, label_tensor_fp);
  uint32_t label_dim = (uint32_t)1;
  fwrite((void*)&label_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&label_dim, sizeof(uint32_t), 1, label_tensor_fp);
  fwrite((void*)&label_dim, sizeof(uint32_t), 1, label_tensor_fp);

  // Create hypothesis cubes that are valid (non-empty and with color)
  std::cerr << "Generating hypothesis bounding boxes...";
  float tsdf_threshold = 0.2f;
  float cube_incr = 0.02f;
  int cube_dim = 30;
  int num_hypotheses = 0;
  std::vector<float> crop_data; // num_hypotheses x 6
  int cube_incr_grid = (int) round(cube_incr / voxel_volume.unit);
  std::vector<std::vector<int>> valid_cube_loc;
  for (int x = grid_bounds[0] + cube_dim / 2; x < grid_bounds[1] - cube_dim / 2; x = x + cube_incr_grid)
    for (int y = grid_bounds[2] + cube_dim / 2; y < grid_bounds[3] - cube_dim / 2; y = y + cube_incr_grid)
      for (int z = grid_bounds[4] + cube_dim / 2; z < grid_bounds[5] - cube_dim / 2; z = z + cube_incr_grid) {

        // Get 3D cube
        int cube_occ = 0;
        float * curr_cube = new float[30 * 30 * 30];
        for (int i = -15; i < 15; i++) {
          for (int j = -15; j < 15; j++) {
            for (int k = -15; k < 15; k++) {
              int volumeIDX = (z + k) * 512 * 512 + (y + j) * 512 + (x + i);
              curr_cube[(k + 15) * 30 * 30 + (j + 15) * 30 + (i + 15)] = voxel_volume.tsdf[volumeIDX];
              if (voxel_volume.tsdf[volumeIDX] < tsdf_threshold)
                cube_occ++;
            }
          }
        }

        // Skip empty cubes
        if (cube_occ == 0)
          continue;

        // Convert cube location from grid to camera coordinates
        float x_cam = ((float)x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
        float y_cam = ((float)y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
        float z_cam = ((float)z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

        // If cube 2D projection is not in image bounds, cube is invalid
        float cube_rad = ((float) cube_dim) * voxel_volume.unit / 2;
        float cube_front[12] = {(x_cam + cube_rad), (x_cam + cube_rad), (x_cam - cube_rad), (x_cam - cube_rad),
                                (y_cam + cube_rad), (y_cam - cube_rad), (y_cam - cube_rad), (y_cam + cube_rad),
                                (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad), (z_cam - cube_rad)
                               };
        float cube_front_2D[8] = {};
        for (int i = 0; i < 4; i++) {
          cube_front_2D[0 * 4 + i] = cube_front[0 * 4 + i] * K[0] / cube_front[2 * 4 + i] + K[2];
          cube_front_2D[1 * 4 + i] = cube_front[1 * 4 + i] * K[4] / cube_front[2 * 4 + i] + K[5];
        }
        for (int i = 0; i < 12; i++)
          cube_front_2D[i] = std::round(cube_front_2D[i]);
        if (std::min(std::min(cube_front_2D[0], cube_front_2D[1]), std::min(cube_front_2D[2], cube_front_2D[3])) < 0 ||
            std::max(std::max(cube_front_2D[0], cube_front_2D[1]), std::max(cube_front_2D[2], cube_front_2D[3])) >= 640 ||
            std::min(std::min(cube_front_2D[4], cube_front_2D[5]), std::min(cube_front_2D[6], cube_front_2D[7])) < 0 ||
            std::max(std::max(cube_front_2D[4], cube_front_2D[5]), std::max(cube_front_2D[6], cube_front_2D[7])) >= 480)
          continue;

        // Get 2D patch of cube's 2D project to image
        cv::Rect curr_patch_ROI(std::round(cube_front_2D[2]), std::round(cube_front_2D[6]), std::round(cube_front_2D[1] - cube_front_2D[2]), std::round(cube_front_2D[4] - cube_front_2D[6]));
        // std::cout << std::round(cube_front_2D[2]) << " " << std::round(cube_front_2D[6]) << " " << std::round(cube_front_2D[1]-cube_front_2D[2]) << " " << std::round(cube_front_2D[4]-cube_front_2D[6]) << std::endl;
        cv::Mat curr_patch = curr_image(curr_patch_ROI);
        cv::resize(curr_patch, curr_patch, cv::Size(227, 227));
        // cv::namedWindow("Patch", CV_WINDOW_AUTOSIZE );
        // cv::imshow("Patch", curr_patch);
        // cv::waitKey(0);

        // Save 2D crop data
        crop_data.push_back(cube_front_2D[2]);
        crop_data.push_back(cube_front_2D[1]);
        crop_data.push_back(cube_front_2D[6]);
        crop_data.push_back(cube_front_2D[4]);

        // Save 3D crop data
        // std::vector<float> cube_bbox_grid;
        // cube_bbox_grid.push_back((float)x - 15); cube_bbox_grid.push_back((float)x + 14);
        // cube_bbox_grid.push_back((float)y - 15); cube_bbox_grid.push_back((float)y + 14);
        // cube_bbox_grid.push_back((float)z - 15); cube_bbox_grid.push_back((float)z + 14);
        // for (int i = 0; i < 6; i++)
        //   crop_data.push_back(cube_bbox_grid[i]);
        crop_data.push_back(x_cam);
        crop_data.push_back(y_cam);
        crop_data.push_back(z_cam);

        // Write 2D image patch to data tensor file (bgr and subtract mean)
        float * patch_data = new float[3 * 227 * 227];
        for (int tmp_row = 0; tmp_row < 227; tmp_row++)
          for (int tmp_col = 0; tmp_col < 227; tmp_col++) {
            patch_data[0 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[0]) - 102.9801f; // B
            patch_data[1 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[1]) - 115.9465f; // G
            patch_data[2 * 227 * 227 + tmp_row * 227 + tmp_col] = ((float) curr_patch.at<cv::Vec3b>(tmp_row,tmp_col)[2]) - 122.7717f; // R
          }
        fwrite(patch_data, sizeof(float), 3 * 227 * 227, data2D_tensor_fp);

        // Write 3D tsdf volume to data tensor file
        fwrite(curr_cube, sizeof(float), 30 * 30 * 30, data3D_tensor_fp);

        // Write dummy label to label tensor file
        float tmp_label = 0;
        fwrite(&tmp_label, sizeof(float), 1, label_tensor_fp);

        num_hypotheses = num_hypotheses + 1;

        // // Create name for data point
        // std::string hash_id = gen_rand_str(16);
        // std::string data_point_name = class_name + "." + hash_id;

        // // Save 2D patch to processed folder
        // cv::imwrite(processed_data_directory + "/" + data_point_name + ".color.png", curr_patch);

        // // Save 3D TSDF cube to processed folder
        // // std::string tsdf_filename = processed_data_directory + "/" + data_point_name + ".tsdf.bin";
        // std::ofstream tmp_out(processed_data_directory + "/" + data_point_name + ".tsdf.bin", std::ios::binary | std::ios::out);
        // for (int i = 0; i < 30*30*30; i++)
        //   tmp_out.write((char*)&curr_cube[i], sizeof(float));
        // tmp_out.close();

        // Clear memory
        delete [] patch_data;
        delete [] curr_cube;
      }

  std::cerr << " done." << std::endl;

  // Rewrite number of hypothesis in tensor headers
  std::cerr << "Found " << num_hypotheses << " hypothesis bounding boxes." << std::endl;
  std::cerr << "Saving hypotheses to tensors on disk for Marvin...";
  data_size = (uint32_t)num_hypotheses;
  fseek(data2D_tensor_fp, 17, SEEK_SET);
  fwrite((void*)&data_size, sizeof(uint32_t), 1, data2D_tensor_fp);
  fseek(data3D_tensor_fp, 17, SEEK_SET);
  fwrite((void*)&data_size, sizeof(uint32_t), 1, data3D_tensor_fp);
  fseek(label_tensor_fp, 19, SEEK_SET);
  fwrite((void*)&data_size, sizeof(uint32_t), 1, label_tensor_fp);

  fclose(data2D_tensor_fp);
  fclose(data3D_tensor_fp);
  fclose(label_tensor_fp);

  std::cerr << " done." << std::endl;

  // Clear memory
  free(depth_data);

  // Save curr volume to file
  // std::string scene_ply_name = "test.ply";
  // save_volume_to_ply(scene_ply_name);
  // save_volume_to_ply_highlighted(scene_ply_name, obj_bbox_grid);

  // // Run marvin 
  // std::cerr << "Running Marvin for deep learning...";
  std::string class_score_tensor_filename = "TMP.class_score_response.tensor";
  sys_command("rm " + class_score_tensor_filename);
  sys_command("cd tools/marvin; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cudnn/v4rc/lib64; ./marvin test model.json PeriodNet.marvin class_score ../../" + class_score_tensor_filename);
  // sys_command("rm " + data2D_tensor_filename);
  // sys_command("rm " + data3D_tensor_filename);
  // sys_command("rm " + label_tensor_filename);
  // std::cerr << " done." << std::endl;

  // Parse class scores
  std::ifstream inFile(class_score_tensor_filename, std::ios::binary | std::ios::in);
  int header_bytes = (1 + 4 + 4) + (4) + (4 + 4 + 4 + 4 + 4);
  inFile.seekg(size_t(header_bytes));
  float * class_score_raw = new float[num_hypotheses * 2];
  inFile.read((char*)class_score_raw, num_hypotheses * 2 * sizeof(float));
  inFile.close();
  sys_command("rm " + class_score_tensor_filename);

  float highest_class_score = 0;
  float best_guess_IDX = 0;
  for (int i = 0; i < num_hypotheses; i++) {
    if (class_score_raw[i * 2 + 1] > 0.5f) {
      cv::rectangle(curr_image, cv::Point(crop_data[i * 7 + 0], crop_data[i * 7 + 2]), cv::Point(crop_data[i * 7 + 1], crop_data[i * 7 + 3]), cv::Scalar(255, 0, 0));
      if (class_score_raw[i * 2 + 1] > highest_class_score) {
        highest_class_score = class_score_raw[i * 2 + 1];
        best_guess_IDX = i;
      }
    }
  }

  std::cout << "Finished running Marvin." << std::endl;

  if (highest_class_score == 0) {
    std::cout << "No object detected!" << std::endl;
    std::string results_filename = "TMP.results.txt";
    FILE *fp = fopen(results_filename.c_str(), "w");
    fprintf(fp, "0 0 0 0");
    fclose(fp); 
    return;
  }

  // Display detection results
  cv::rectangle(curr_image, cv::Point(crop_data[best_guess_IDX * 7 + 0], crop_data[best_guess_IDX * 7 + 2]), cv::Point(crop_data[best_guess_IDX * 7 + 1], crop_data[best_guess_IDX * 7 + 3]), cv::Scalar(0, 255, 0));
  // std::cout << crop_data[best_guess_IDX * 10 + 0] << " " << crop_data[best_guess_IDX * 10 + 1] << " " << crop_data[best_guess_IDX * 10 + 2] << " " << crop_data[best_guess_IDX * 10 + 3] << std::endl;
  cv::namedWindow("Object Detection", CV_WINDOW_AUTOSIZE );
  cv::imshow("Object Detection", curr_image);

  std::cout << "Detection confidence: " << highest_class_score << std::endl;
  // Convert cube location from grid to camera coordinates
  // std::vector<float> best_guess_center_grid;
  // best_guess_center_grid.push_back(((float)crop_data[best_guess_IDX * 7 + 4]) + ((float)crop_data[best_guess_IDX * 7 + 5])/2);
  // best_guess_center_grid.push_back(((float)crop_data[best_guess_IDX * 7 + 6]) + ((float)crop_data[best_guess_IDX * 7 + 7])/2);
  // best_guess_center_grid.push_back(((float)crop_data[best_guess_IDX * 7 + 8]) + ((float)crop_data[best_guess_IDX * 7 + 9])/2);
  // std::vector<float> best_guess_center_cam;
  // best_guess_center_cam.push_back((best_guess_center_grid[0] + 1) * voxel_volume.unit + voxel_volume.range[0][0]);
  // best_guess_center_cam.push_back((best_guess_center_grid[1] + 1) * voxel_volume.unit + voxel_volume.range[1][0]);
  // best_guess_center_cam.push_back((best_guess_center_grid[2] + 1) * voxel_volume.unit + voxel_volume.range[2][0]);
  std::cout << "Camera coordinates of detected object: " << crop_data[best_guess_IDX * 7 + 4] << " " << crop_data[best_guess_IDX * 7 + 5] << " " << crop_data[best_guess_IDX * 7 + 6] << std::endl;

  std::string results_filename = "TMP.results.txt";
  FILE *fp = fopen(results_filename.c_str(), "w");
  fprintf(fp, "%.17g %.17g %.17g %.17g", highest_class_score, (float)crop_data[best_guess_IDX * 7 + 4], (float)crop_data[best_guess_IDX * 7 + 5], (float)crop_data[best_guess_IDX * 7 + 6]);
  fclose(fp); 
  
  // cv::waitKey(0);
  // cv::destroyWindow("Object Detection");
  // sys_command("rm TMP.frame.color.png");
  // sys_command("rm TMP.frame.depth.png");
  // sys_command("rm TMP.intrinsics.K.txt");
  // cv::waitKey(0);
}
