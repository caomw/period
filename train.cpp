#include "util/util.hpp"
#include <png++/png.hpp>
#include <opencv2/opencv.hpp>

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

int main(int argc, char **argv) {

  std::string data_directory = "data";
  std::string sequence_directory = data_directory + "/train1";
  std::string processed_data_directory = data_directory + "/train";
  std::string object_name = "glue";

  sys_command("mkdir -p " + processed_data_directory);

  // Get file list of color images
  std::vector<std::string> file_list_color;
  std::string color_regex = ".color.png";
  get_files_in_directory(sequence_directory, file_list_color, color_regex);
  std::sort(file_list_color.begin(), file_list_color.end());

  // Get file list of depth images
  std::vector<std::string> file_list_depth;
  std::string depth_regex = ".depth.png";
  get_files_in_directory(sequence_directory, file_list_depth, depth_regex);
  std::sort(file_list_depth.begin(), file_list_depth.end());

  // Get file list of intrinsics
  std::vector<std::string> file_list_intrinsics;
  std::string intrinsics_regex = ".K.txt";
  get_files_in_directory(sequence_directory, file_list_intrinsics, intrinsics_regex);
  std::sort(file_list_intrinsics.begin(), file_list_intrinsics.end());

  // Get file list of extrinsics
  std::vector<std::string> file_list_extrinsics;
  std::string extrinsics_regex = ".pose.txt";
  get_files_in_directory(sequence_directory, file_list_extrinsics, extrinsics_regex);
  std::sort(file_list_extrinsics.begin(), file_list_extrinsics.end());

  // Load intrinsics (3x3 matrix)
  std::string intrinsic_filename = sequence_directory + "/intrinsics.K.txt";
  std::vector<float> K = load_matrix_from_file(intrinsic_filename, 3, 3);

  // Load extrinsics (4x4 matrices)
  std::vector<std::vector<float>> extrinsics;
  for (std::string &curr_filename : file_list_extrinsics) {
    std::string curr_extrinsic_filename = sequence_directory + "/" + curr_filename;
    std::vector<float> curr_extrinsic = load_matrix_from_file(curr_extrinsic_filename, 4, 4);
    extrinsics.push_back(curr_extrinsic);
  }

  // std::cout << sequence_directory + "/" + file_list_color[0] << std::endl;
  // std::string tmp_filename = sequence_directory + "/" + file_list_color[0];
  // cv::Mat image;
  // image = cv::imread( tmp_filename.c_str(), 1 );
  // cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
  // cv::imshow("Display Image", image);
  // cv::waitKey(0);

  // Init voxel volume params
  init_voxel_volume();

  // Set first frame of sequence as base coordinate frame
  int base_frame = 0;

  // Fuse frames
  for (int curr_frame = 0; curr_frame < file_list_depth.size(); curr_frame++) {
    std::cerr << "Fusing frame " << curr_frame << "...";

    // Load image/depth/extrinsic data for current frame
    std::string curr_image_filename = sequence_directory + "/" + file_list_color[curr_frame];
    cv::Mat curr_image = cv::imread(curr_image_filename.c_str(), 1);
    unsigned short * depth_data = (unsigned short *) malloc(kImageRows * kImageCols * sizeof(unsigned short));
    std::string curr_depth_filename = sequence_directory + "/" + file_list_depth[curr_frame];
    read_depth_data(curr_depth_filename, depth_data);

    // Compute relative camera pose transform between current frame and base frame
    // Compute camera view frustum bounds within the voxel volume
    float camera_relative_pose[16] = {0};
    float view_bounds[6] = {0};
    // get_frustum_bounds(K, extrinsics, base_frame, curr_frame, camera_relative_pose, view_bounds);
    get_frustum_bounds(K, extrinsics, 0, 0, camera_relative_pose, view_bounds); // Note: set relative pose to identity for single frame fusion

    // Integrate
    integrate(K, depth_data, view_bounds, camera_relative_pose);

    // Read bounding box of object
    std::string obj_bbox_filename = sequence_directory + "/glue.bbox.txt";
    std::vector<float> obj_bbox = load_matrix_from_file(obj_bbox_filename, 3, 2);

    // Convert bounding box of object from world coordinates to current camera coordinates to grid coordinates
    std::vector<float> curr_pose = extrinsics[curr_frame];
    std::vector<float> obj_bbox_cam = obj_bbox;
    std::vector<float> obj_bbox_grid = obj_bbox_cam;
    float * curr_pose_arr = &curr_pose[0];
    float curr_pose_inv[16] = {0};
    invert_matrix(curr_pose_arr, curr_pose_inv);
    for (int i = 0; i < 2; i++) {
      float tmp_arr[3] = {0};
      tmp_arr[0] = curr_pose_inv[0 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[0 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[0 * 4 + 2] * obj_bbox[2 * 2 + i];
      tmp_arr[1] = curr_pose_inv[1 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[1 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[1 * 4 + 2] * obj_bbox[2 * 2 + i];
      tmp_arr[2] = curr_pose_inv[2 * 4 + 0] * obj_bbox[0 * 2 + i] + curr_pose_inv[2 * 4 + 1] * obj_bbox[1 * 2 + i] + curr_pose_inv[2 * 4 + 2] * obj_bbox[2 * 2 + i];
      obj_bbox_cam[0 * 2 + i] = tmp_arr[0] + curr_pose_inv[3];
      obj_bbox_cam[1 * 2 + i] = tmp_arr[1] + curr_pose_inv[7];
      obj_bbox_cam[2 * 2 + i] = tmp_arr[2] + curr_pose_inv[11];
    }
    for (int i = 0; i < 3; i++) {
      if (obj_bbox_cam[i * 2 + 0] > obj_bbox_cam[i * 2 + 1]) {
        float tmp_swap = obj_bbox_cam[i * 2 + 0];
        obj_bbox_cam[i * 2 + 0] = obj_bbox_cam[i * 2 + 1];
        obj_bbox_cam[i * 2 + 1] = tmp_swap;
      }
    }
    for (int i = 0; i < 2; i++) {
      obj_bbox_grid[0 * 2 + i] = (obj_bbox_cam[0 * 2 + i] - voxel_volume.range[0][0]) / voxel_volume.unit - 1;
      obj_bbox_grid[1 * 2 + i] = (obj_bbox_cam[1 * 2 + i] - voxel_volume.range[1][0]) / voxel_volume.unit - 1;
      obj_bbox_grid[2 * 2 + i] = (obj_bbox_cam[2 * 2 + i] - voxel_volume.range[2][0]) / voxel_volume.unit - 1;
    }

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

    // cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", curr_image);
    // cv::waitKey(0);

    // Data structure to store crop information of hypothesis cubes
    std::vector<std::string> crop_data;

    // Create hypothesis cubes that are valid (non-empty and with color)
    float tsdf_threshold = 0.2f;
    float cube_incr = 0.01f;
    int cube_dim = 30;
    int cube_incr_grid = (int) round(cube_incr / voxel_volume.unit);
    std::vector<std::vector<int>> valid_cube_loc;
    for (int x = grid_bounds[0] + cube_dim / 2; x < grid_bounds[1] - cube_dim / 2; x = x + cube_incr_grid)
      for (int y = grid_bounds[2] + cube_dim / 2; y < grid_bounds[3] - cube_dim / 2; y = y + cube_incr_grid)
        for (int z = grid_bounds[4] + cube_dim / 2; z < grid_bounds[5] - cube_dim / 2; z = z + cube_incr_grid) {

          // Get 3D cube
          int cube_occ = 0;
          // float * curr_cube = new float[30 * 30 * 30];
          for (int i = -15; i < 15; i++) {
            for (int j = -15; j < 15; j++) {
              for (int k = -15; k < 15; k++) {
                int volumeIDX = (z + k) * 512 * 512 + (y + j) * 512 + (x + i);
                // curr_cube[(k + 15) * 30 * 30 + (j + 15) * 30 + (i + 15)] = voxel_volume.tsdf[volumeIDX];
                if (voxel_volume.tsdf[volumeIDX] < tsdf_threshold)
                  cube_occ++;
              }
            }
          }

          // Skip empty cubes
          if (cube_occ == 0)
            continue;

          // Convert cube location from grid to camera coordinates
          float x_cam = (x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
          float y_cam = (y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
          float z_cam = (z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

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

          // Check overlap of cube with object bbox
          float overlap = 0;
          std::vector<float> cube_bbox_grid;
          cube_bbox_grid.push_back((float)x - 15); cube_bbox_grid.push_back((float)x + 14);
          cube_bbox_grid.push_back((float)y - 15); cube_bbox_grid.push_back((float)y + 14);
          cube_bbox_grid.push_back((float)z - 15); cube_bbox_grid.push_back((float)z + 14);
          if (((cube_bbox_grid[0] > obj_bbox_grid[0] && cube_bbox_grid[0] < obj_bbox_grid[1]) || (cube_bbox_grid[1] > obj_bbox_grid[0] && cube_bbox_grid[1] < obj_bbox_grid[1])) && 
              ((cube_bbox_grid[2] > obj_bbox_grid[2] && cube_bbox_grid[2] < obj_bbox_grid[3]) || (cube_bbox_grid[3] > obj_bbox_grid[2] && cube_bbox_grid[3] < obj_bbox_grid[3])) &&
              ((cube_bbox_grid[4] > obj_bbox_grid[4] && cube_bbox_grid[4] < obj_bbox_grid[5]) || (cube_bbox_grid[5] > obj_bbox_grid[4] && cube_bbox_grid[5] < obj_bbox_grid[5]))) {
            float overlapX = std::min(std::abs(obj_bbox_grid[0]-cube_bbox_grid[1]),std::abs(obj_bbox_grid[1]-cube_bbox_grid[0]));
            float overlapY = std::min(std::abs(obj_bbox_grid[2]-cube_bbox_grid[3]),std::abs(obj_bbox_grid[3]-cube_bbox_grid[2]));
            float overlapZ = std::min(std::abs(obj_bbox_grid[4]-cube_bbox_grid[5]),std::abs(obj_bbox_grid[5]-cube_bbox_grid[4]));
            float overlap_intersect = overlapX*overlapY*overlapZ;
            float obj_bbox_area = (obj_bbox_grid[1]-obj_bbox_grid[0])*(obj_bbox_grid[3]-obj_bbox_grid[2])*(obj_bbox_grid[5]-obj_bbox_grid[4]);
            overlap = overlap_intersect/obj_bbox_area;
          }

          int class_idx = 0;
          if (overlap > 0.7) {
            // std::cout << overlap << std::endl;
            // std::string scene_ply_name = "test.ply";
            // save_volume_to_ply_highlighted(scene_ply_name, cube_bbox_grid);
            class_idx = 1;

            // // Show 2D patch
            // cv::namedWindow("Patch", CV_WINDOW_AUTOSIZE );
            // cv::imshow("Patch", curr_patch);
            // cv::waitKey(0);
            // std::cout << std::endl;
            // std::cout << std::endl;
          }

          if (overlap > 0.3 && overlap < 0.7)
            continue;

          // for (int i = 0; i < 8; i++) 
          //   std::cout << cube_front_2D[i] << " ";
          // std::cout << std::endl;
          // for (int i = 0; i < 6; i++) 
          //   std::cout << cube_bbox_grid[i] << " ";
          // std::cout << std::endl;

          // Class (idx), 2D patch bbox (x1 x2 y1 y2), 3D tsdf bbox (x1 x2 y1 y2 z1 z2)
          std::string data_string = std::to_string(class_idx) + " " + std::to_string((int)cube_front_2D[2]) + " " + std::to_string((int)cube_front_2D[1]) + " " + 
                                                                      std::to_string((int)cube_front_2D[5]) + " " + std::to_string((int)cube_front_2D[4]) + " " + 
                                                                      std::to_string((int)cube_bbox_grid[0]-(int)grid_bounds[0]) + " " + std::to_string((int)cube_bbox_grid[1]-(int)grid_bounds[0]) + " " + 
                                                                      std::to_string((int)cube_bbox_grid[2]-(int)grid_bounds[2]) + " " + std::to_string((int)cube_bbox_grid[3]-(int)grid_bounds[2]) + " " + 
                                                                      std::to_string((int)cube_bbox_grid[4]-(int)grid_bounds[4]) + " " + std::to_string((int)cube_bbox_grid[5]-(int)grid_bounds[4]);
          // std::cout << data_string << std::endl;
          crop_data.push_back(data_string);

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
          // delete [] curr_cube;
        }

    // Give current frame a hash id
    std::string curr_frame_hash = gen_rand_str(16);

    // Save image
    std::string curr_frame_color_filename = processed_data_directory + "/" + curr_frame_hash + ".color.png";
    cv::imwrite(curr_frame_color_filename, curr_image);

    // Save volume
    std::string curr_frame_tsdf_filename = processed_data_directory + "/" + curr_frame_hash + "." + std::to_string((int)grid_bounds[1]-(int)grid_bounds[0]) + "." + std::to_string((int)grid_bounds[3]-(int)grid_bounds[2]) + "." + std::to_string((int)grid_bounds[5]-(int)grid_bounds[4]) + ".tsdf.bin";
    std::ofstream tmp_out(curr_frame_tsdf_filename, std::ios::binary | std::ios::out);
    for (int z = grid_bounds[4]; z < grid_bounds[5]; z++)
      for (int y = grid_bounds[2]; y < grid_bounds[3]; y++)
        for (int x = grid_bounds[0]; x < grid_bounds[1]; x++) {
          int volumeIDX = z * 512 * 512 + y * 512 + x;
          tmp_out.write((char*)&voxel_volume.tsdf[volumeIDX], sizeof(float));
        }
    tmp_out.close();

    // Save crop information
    std::string curr_frame_crop_filename = processed_data_directory + "/" + curr_frame_hash + "." + std::to_string(crop_data.size()) + ".crop.txt";
    FILE *fp = fopen(curr_frame_crop_filename.c_str(), "w");
    for (int i = 0; i < crop_data.size(); i++)
      fprintf(fp, "%s\n", crop_data[i].c_str());
    fclose(fp);


    // Clear memory
    free(depth_data);
    std::cerr << " done!" << std::endl;

    // Save curr volume to file
    // std::string scene_ply_name = "test.ply";
    // save_volume_to_ply(scene_ply_name);
    // save_volume_to_ply_highlighted(scene_ply_name, obj_bbox_grid);

    // Init new volume
    memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
    for (int i = 0; i < 512 * 512 * 1024; i++)
      voxel_volume.tsdf[i] = 1.0f;

  }
  return 0;
}

