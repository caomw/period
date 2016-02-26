#include "util/util.hpp"
#include <png++/png.hpp>

////////////////////////////////////////////////////////////////////////////////

const int kImageRows = 480;
const int kImageCols = 640;
const int kSampleFactor = 30;
const int kImageChannels = 3;
const int kFileNameLength = 24;
const int kTimeStampPos = 8;
const int kTimeStampLength = 12;

typedef unsigned char uchar;
typedef unsigned short ushort;

struct stat file_info;

struct _cam_k {
  float fx;
  float fy;
  float cx;
  float cy;
} cam_K;

//float cam_view_frustum[3][5];

typedef struct _extrinsic {
  float R[9];
  float t[3];
} extrinsic;

typedef struct _keypoint {
  float x;
  float y;
  float z;
  float response;
} keypoint;

typedef struct _normal {
  float x;
  float y;
  float z;
} normal;

struct _voxel_volume {
  float unit;
  float mu_grid;
  float mu;
  float size_grid[3];
  float range[3][2];
  float* tsdf;
  float* weight;
  std::vector<keypoint> keypoints;
} voxel_volume;

bool get_depth_data_sevenscenes(const std::string &file_name, ushort *data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(), png::require_color_space< png::gray_pixel_16 >());

  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      ushort s = img.get_pixel(j, i);
      *(data + index) = s;
      ++index;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool get_image_data_sun3d(const std::string &file_name, uchar *data) {
  unsigned char *raw_image = NULL;

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];

  FILE *infile = fopen(file_name.c_str(), "rb");
  unsigned long location = 0;

  if (!infile) {
    printf("Error opening jpeg file %s\n!", file_name.c_str());
    return -1;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  raw_image = (unsigned char*) malloc(
                cinfo.output_width * cinfo.output_height * cinfo.num_components);
  row_pointer[0] = (unsigned char *) malloc(
                     cinfo.output_width * cinfo.num_components);

  while (cinfo.output_scanline < cinfo.image_height) {
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
    for (uint i = 0; i < cinfo.image_width * cinfo.num_components; i++)
      raw_image[location++] = row_pointer[0][i];
  }

  int index = 0;
  for (uint i = 0; i < cinfo.image_height; ++i) {
    for (uint j = 0; j < cinfo.image_width; ++j) {
      for (int k = 0; k < kImageChannels; ++k) {
        *(data + index) = raw_image[(i * cinfo.image_width * 3) + (j * 3) + k];
        ++index;
      }
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(row_pointer[0]);
  fclose(infile);

  return true;
}

////////////////////////////////////////////////////////////////////////////////

void get_extrinsic_data(const std::string &file_name, int size,
                        std::vector<extrinsic> *poses) {
  FILE *fp = fopen(file_name.c_str(), "r");

  for (int i = 0; i < size; ++i) {
    extrinsic m;
    for (int d = 0; d < 3; ++d) {
      int iret;
      iret = fscanf(fp, "%f", &m.R[d * 3 + 0]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 1]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 2]);
      iret = fscanf(fp, "%f", &m.t[d]);
    }
    poses->push_back(m);
  }

  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void get_sevenscenes_data(std::string local_dir, std::vector<std::string> &image_list, std::vector<std::string> &depth_list, std::vector<extrinsic> &extrinsic_poses) {

  // Get all filenames
  std::vector<std::string> local_filenames;
  get_local_filenames(local_dir, &local_filenames);

  // Sort filenames as iamge/depth/extrinsic
  std::vector<std::string> extrinsic_list;
  for (int i = 0; i < local_filenames.size(); i++) {
    if (local_filenames[i].find(".color.png") != std::string::npos)
      image_list.push_back(local_filenames[i]);
    if (local_filenames[i].find(".depth.png") != std::string::npos)
      depth_list.push_back(local_filenames[i]);
    if (local_filenames[i].find(".pose.txt") != std::string::npos)
      extrinsic_list.push_back(local_filenames[i]);
  }

  // Get extrinsic matrices
  for (int i = 0; i < extrinsic_list.size(); i++) {
    FILE *fp = fopen(extrinsic_list[i].c_str(), "r");
    int iret;
    extrinsic m;
    for (int d = 0; d < 3; ++d) {
      iret = fscanf(fp, "%f", &m.R[d * 3 + 0]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 1]);
      iret = fscanf(fp, "%f", &m.R[d * 3 + 2]);
      iret = fscanf(fp, "%f", &m.t[d]);
    }
    extrinsic_poses.push_back(m);
    fclose(fp);

  }
}

////////////////////////////////////////////////////////////////////////////////

void init_voxel_volume() {
  voxel_volume.unit = 0.01;
  voxel_volume.mu_grid = 5;
  voxel_volume.mu = voxel_volume.unit * voxel_volume.mu_grid;

  voxel_volume.size_grid[0] = 512;
  voxel_volume.size_grid[1] = 512;
  voxel_volume.size_grid[2] = 1024;

  voxel_volume.range[0][0] = -voxel_volume.size_grid[0] * voxel_volume.unit / 2;
  voxel_volume.range[0][1] = voxel_volume.range[0][0] + voxel_volume.size_grid[0] * voxel_volume.unit;
  voxel_volume.range[1][0] = -voxel_volume.size_grid[1] * voxel_volume.unit / 2;
  voxel_volume.range[1][1] = voxel_volume.range[1][0] + voxel_volume.size_grid[1] * voxel_volume.unit;
  voxel_volume.range[2][0] = -0.5;
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

  voxel_volume.keypoints.clear();
}

////////////////////////////////////////////////////////////////////////////////

void save_volume_to_ply(const std::string &file_name) {
  float tsdf_threshold = 0.2;
  float weight_threshold = 1;
  float radius = 5;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold)
      num_points++;

  // std::cout << "keypoint size during check: " << voxel_volume.keypoints.size() << std::endl;
  int keypoint_count = 0;
  // Create header for ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points + (int)voxel_volume.keypoints.size() * 20);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < 512 * 512 * 1024; i++) {

    // if (voxel_volume.weight[i] > weight_threshold) {
    //   int z = floor(i/(512*512));
    //   int y = floor((i - (z*512*512))/512);
    //   int x = i - (z*512*512) - (y*512);
    //   float float_x = (float) x;
    //   float float_y = (float) y;
    //   float float_z = (float) z;
    //   fwrite(&float_x, sizeof(float), 1, fp);
    //   fwrite(&float_y, sizeof(float), 1, fp);
    //   fwrite(&float_z, sizeof(float), 1, fp);
    //   uchar r = (char)0;
    //   uchar g = (char)0;
    //   uchar b = (char)255;
    //   fwrite(&r, sizeof(uchar), 1, fp);
    //   fwrite(&g, sizeof(uchar), 1, fp);
    //   fwrite(&b, sizeof(uchar), 1, fp);
    // }

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold) {

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

      // If voxel is keypoint, color it red, otherwise color it gray
      uchar r = (uchar)180;
      uchar g = (uchar)180;
      uchar b = (uchar)180;
      bool is_keypoint = false;
      // bool is_valid_keypoint = false;
      for (int k = 0; k < voxel_volume.keypoints.size(); k++) {
        if (voxel_volume.keypoints[k].x == x && voxel_volume.keypoints[k].y == y && voxel_volume.keypoints[k].z == z) {
          r = (uchar)0;
          g = (uchar)0;
          b = (uchar)255;
          // float empty_space = 0;
          float occupancy = 0;
          for (int kk = -15; kk <= 15; kk++) {
            for (int jj = -15; jj <= 15; jj++) {
              for (int ii = -15; ii <= 15; ii++) {
                if (voxel_volume.weight[(z + kk) * 512 * 512 + (y + jj) * 512 + (x + ii)] >= 1) {
                  occupancy++;
                }
              }
            }
          }
          //float occupancy = 1 - empty_space/(31*31*31);
          occupancy = occupancy / (31 * 31 * 31);
          if (occupancy > 0.5) {
            r = (uchar)255;
            g = (uchar)0;
            b = (uchar)0;
            keypoint_count++;
            // is_valid_keypoint = true;
          }
          is_keypoint = true;
          break;
        }
      }

      fwrite(&r, sizeof(uchar), 1, fp);
      fwrite(&g, sizeof(uchar), 1, fp);
      fwrite(&b, sizeof(uchar), 1, fp);

      // Draw 5x5x5 box around keypoint
      if (is_keypoint) {
        for (int kk = -1; kk <= 1; kk++) {
          for (int jj = -1; jj <= 1; jj++) {
            for (int ii = -1; ii <= 1; ii++) {
              int num_edges = 0;
              if (kk == -1 || kk == 1)
                num_edges++;
              if (jj == -1 || jj == 1)
                num_edges++;
              if (ii == -1 || ii == 1)
                num_edges++;
              if (num_edges >= 2) {
                float float_x = (float) x + ii;
                float float_y = (float) y + jj;
                float float_z = (float) z + kk;
                fwrite(&float_x, sizeof(float), 1, fp);
                fwrite(&float_y, sizeof(float), 1, fp);
                fwrite(&float_z, sizeof(float), 1, fp);
                fwrite(&r, sizeof(uchar), 1, fp);
                fwrite(&g, sizeof(uchar), 1, fp);
                fwrite(&b, sizeof(uchar), 1, fp);
              }
            }
          }
        }
      }

    }
  }
  // std::cout << keypoint_count << std::endl;
  fclose(fp);
}

void save_volume_to_tsdf(const std::string &file_name) {
  float tsdf_threshold = 0.2;
  float weight_threshold = 1;

  std::ofstream outFile(file_name, std::ios::binary | std::ios::out);

  // Find tight bounds of volume (bind to where weight > 1 and std::abs tsdf < 1)
  int start_idx = -1;
  for (int i = 0; i < 512 * 512 * 1024; i++)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold && start_idx == -1) {
      start_idx = i;
      break;
    }

  int end_idx = -1;
  for (int i = 512 * 512 * 1024 - 1; i >= 0; i--)
    if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold && end_idx == -1) {
      end_idx = i;
      break;
    }

  // std::cout << start_idx << std::endl;
  // std::cout << end_idx << std::endl;

  int num_elements = end_idx - start_idx + 1;
  outFile.write((char*)&start_idx, sizeof(int));
  outFile.write((char*)&num_elements, sizeof(int));

  for (int i = start_idx; i <= end_idx; i++)
    outFile.write((char*)&voxel_volume.tsdf[i], sizeof(float));

  outFile.close();

  // TEMPORARY  
  std::string tmp_name = file_name + "f";
  std::ofstream tmpOut(tmp_name, std::ios::binary | std::ios::out);

  for (int i = 0; i < 512*512*1024; i++)
    tmpOut.write((char*)&voxel_volume.tsdf[i], sizeof(float));

  tmpOut.close();
  // if (std::abs(voxel_volume.tsdf[i]) < tsdf_threshold && voxel_volume.weight[i] > weight_threshold)
  //   num_points++;

}

////////////////////////////////////////////////////////////////////////////////

void integrate(ushort* depth_data, float* range_grid, float* cam_R, float* cam_t) {

  for (int z = range_grid[2 * 2 + 0]; z < range_grid[2 * 2 + 1]; z++) {
    for (int y = range_grid[1 * 2 + 0]; y < range_grid[1 * 2 + 1]; y++) {
      for (int x = range_grid[0 * 2 + 0]; x < range_grid[0 * 2 + 1]; x++) {

        // grid to world coords
        float tmp_pos[3] = {0};
        tmp_pos[0] = (x + 1) * voxel_volume.unit + voxel_volume.range[0][0];
        tmp_pos[1] = (y + 1) * voxel_volume.unit + voxel_volume.range[1][0];
        tmp_pos[2] = (z + 1) * voxel_volume.unit + voxel_volume.range[2][0];

        // transform
        float tmp_arr[3] = {0};
        tmp_arr[0] = tmp_pos[0] - cam_t[0];
        tmp_arr[1] = tmp_pos[1] - cam_t[1];
        tmp_arr[2] = tmp_pos[2] - cam_t[2];
        tmp_pos[0] = cam_R[0 * 3 + 0] * tmp_arr[0] + cam_R[1 * 3 + 0] * tmp_arr[1] + cam_R[2 * 3 + 0] * tmp_arr[2];
        tmp_pos[1] = cam_R[0 * 3 + 1] * tmp_arr[0] + cam_R[1 * 3 + 1] * tmp_arr[1] + cam_R[2 * 3 + 1] * tmp_arr[2];
        tmp_pos[2] = cam_R[0 * 3 + 2] * tmp_arr[0] + cam_R[1 * 3 + 2] * tmp_arr[1] + cam_R[2 * 3 + 2] * tmp_arr[2];
        if (tmp_pos[2] <= 0)
          continue;

        int px = std::round(cam_K.fx * (tmp_pos[0] / tmp_pos[2]) + cam_K.cx);
        int py = std::round(cam_K.fy * (tmp_pos[1] / tmp_pos[2]) + cam_K.cy);
        if (px < 1 || px > 640 || py < 1 || py > 480)
          continue;

        float p_depth = *(depth_data + (py - 1) * kImageCols + (px - 1)) / 1000.f;
        if (p_depth > 4.5f)
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

////////////////////////////////////////////////////////////////////////////////

// Save intrinsic matrix to global variable cam_K
void get_intrinsic_matrix(const std::string &local_camera) {
  int iret;
  float ff;
  FILE *fp = fopen(local_camera.c_str(), "r");
  iret = fscanf(fp, "%f", &cam_K.fx);
  iret = fscanf(fp, "%f", &ff);
  iret = fscanf(fp, "%f", &cam_K.cx);
  iret = fscanf(fp, "%f", &ff);
  iret = fscanf(fp, "%f", &cam_K.fy);
  iret = fscanf(fp, "%f", &cam_K.cy);
  fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////

void compute_frustum_bounds(std::vector<extrinsic> &extrinsic_poses, int base_frame, int curr_frame, float* cam_R, float* cam_t, float* range_grid) {

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
  extrinsic ex_pose1 = extrinsic_poses[base_frame];
  extrinsic ex_pose2 = extrinsic_poses[curr_frame];

  float ex_mat1[16] =
  { ex_pose1.R[0 * 3 + 0], ex_pose1.R[0 * 3 + 1], ex_pose1.R[0 * 3 + 2], ex_pose1.t[0],
    ex_pose1.R[1 * 3 + 0], ex_pose1.R[1 * 3 + 1], ex_pose1.R[1 * 3 + 2], ex_pose1.t[1],
    ex_pose1.R[2 * 3 + 0], ex_pose1.R[2 * 3 + 1], ex_pose1.R[2 * 3 + 2], ex_pose1.t[2],
    0,                0,                0,            1
  };

  float ex_mat2[16] =
  { ex_pose2.R[0 * 3 + 0], ex_pose2.R[0 * 3 + 1], ex_pose2.R[0 * 3 + 2], ex_pose2.t[0],
    ex_pose2.R[1 * 3 + 0], ex_pose2.R[1 * 3 + 1], ex_pose2.R[1 * 3 + 2], ex_pose2.t[1],
    ex_pose2.R[2 * 3 + 0], ex_pose2.R[2 * 3 + 1], ex_pose2.R[2 * 3 + 2], ex_pose2.t[2],
    0,                 0,                0,            1
  };

  float ex_mat1_inv[16] = {0};
  invMatrix(ex_mat1, ex_mat1_inv);
  float ex_mat_rel[16] = {0};
  mulMatrix(ex_mat1_inv, ex_mat2, ex_mat_rel);

  cam_R[0] = ex_mat_rel[0];
  cam_R[1] = ex_mat_rel[1];
  cam_R[2] = ex_mat_rel[2];
  cam_R[3] = ex_mat_rel[4];
  cam_R[4] = ex_mat_rel[5];
  cam_R[5] = ex_mat_rel[6];
  cam_R[6] = ex_mat_rel[8];
  cam_R[7] = ex_mat_rel[9];
  cam_R[8] = ex_mat_rel[10];

  cam_t[0] = ex_mat_rel[3];
  cam_t[1] = ex_mat_rel[7];
  cam_t[2] = ex_mat_rel[11];

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
  float cam_view_frustum[15] =
  { 0, -320 * 8 / cam_K.fx, -320 * 8 / cam_K.fx, 320 * 8 / cam_K.fx,  320 * 8 / cam_K.fx,
    0, -240 * 8 / cam_K.fx,  240 * 8 / cam_K.fx, 240 * 8 / cam_K.fx, -240 * 8 / cam_K.fx,
    0,               8,               8,              8,              8
  };

  // Rotate cam view frustum wrt Rt
  for (int i = 0; i < 5; i++) {
    float tmp_arr[3] = {0};
    tmp_arr[0] = cam_R[0 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[0 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[0 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[1] = cam_R[1 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[1 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[1 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    tmp_arr[2] = cam_R[2 * 3 + 0] * cam_view_frustum[0 + i] + cam_R[2 * 3 + 1] * cam_view_frustum[5 + i] + cam_R[2 * 3 + 2] * cam_view_frustum[2 * 5 + i];
    cam_view_frustum[0 * 5 + i] = tmp_arr[0] + cam_t[0];
    cam_view_frustum[1 * 5 + i] = tmp_arr[1] + cam_t[1];
    cam_view_frustum[2 * 5 + i] = tmp_arr[2] + cam_t[2];
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
    range_grid[i * 2 + 0] = std::max(0.0f, std::floor((range2test[i][0] - voxel_volume.range[i][0]) / voxel_volume.unit));
    range_grid[i * 2 + 1] = std::min(voxel_volume.size_grid[i], std::ceil((range2test[i][1] - voxel_volume.range[i][0]) / voxel_volume.unit + 1));
  }

  // std::cout << range_grid[0*2+0] << std::endl;
  // std::cout << range_grid[0*2+1] << std::endl;
  // std::cout << range_grid[1*2+0] << std::endl;
  // std::cout << range_grid[1*2+1] << std::endl;
  // std::cout << range_grid[2*2+0] << std::endl;
  // std::cout << range_grid[2*2+1] << std::endl;

}

void generate_data(const std::string &sequence_prefix, const std::string &local_dir) {

  std::vector<std::string> image_list;
  std::vector<std::string> depth_list;
  std::vector<extrinsic> extrinsic_poses;

/////////////////////////////////////////////////////////////////////

  // // Retrieve local directory
  // std::string sequence_name = "hotel_umd/maryland_hotel3/";
  // passwd* pw = getpwuid(getuid());
  // std::string home_dir(pw->pw_dir);
  // std::string local_dir = home_dir + "/marvin/kinfu/data/sun3d/" + sequence_name;

  // // Query RGB-D sequence from SUN3D
  // get_sun3d_data(sequence_name, local_dir, image_list, depth_list, extrinsic_poses);

/////////////////////////////////////////////////////////////////////

  // Retrieve local directory
  // std::string sequence_name = "fire/seq-03/";
  // std::string sequence_prefix = "/data/04/andyz/kinfu/train/fire_seq03/";
  // sys_command("mkdir -p /data/04/andyz/kinfu/train");
  sys_command("mkdir -p " + sequence_prefix);
  // passwd* pw = getpwuid(getuid());
  // std::string home_dir(pw->pw_dir);
  std::cout << std::endl << local_dir << std::endl;
  // std::string local_dir = home_dir + "/marvin/kinfu/data/sevenscenes/" + sequence_name;
  // std::string local_dir = "/data/04/andyz/kinfu/data/sevenscenes/fire/seq-03/";

  // Query RGB-D sequence from
  get_sevenscenes_data(local_dir, image_list, depth_list, extrinsic_poses);

/////////////////////////////////////////////////////////////////////

  // Count total number of frames
  int total_files = 0;
  image_list.size() < depth_list.size() ? total_files = image_list.size() : total_files = depth_list.size();

  // Init voxel volume params
  init_voxel_volume();

  // Init intrinsic matrix K
  std::string local_camera = local_dir + "intrinsics.txt";
  if (std::ifstream(local_camera))
    get_intrinsic_matrix(local_camera);
  else {
    std::cout << "Intrinsics not found. Loading default matrix." << std::endl;
    cam_K.fx = 585.0f; cam_K.fy = 585.0f; cam_K.cx = 320.0f; cam_K.cy = 240.0f;
  }

  // Set first frame of sequence as base coordinate frame
  int base_frame = 0;
  int first_frame = 0;
  std::vector<int> base_frames;
  base_frames.push_back(base_frame);

  // Fuse frames
  for (int curr_frame = 0; curr_frame < total_files; curr_frame++) {
    std::cout << "Fusing frame " << curr_frame << "...";

    // Load image/depth/extrinsic data for frame curr_frame
    // uchar *image_data = (uchar *) malloc(kImageRows * kImageCols * kImageChannels * sizeof(uchar));
    ushort *depth_data = (ushort *) malloc(kImageRows * kImageCols * sizeof(ushort));
    // get_image_data_sun3d(image_list[curr_frame], image_data);
    // get_depth_data_sun3d(depth_list[curr_frame], depth_data);
    // std::cout << depth_list[curr_frame] << std::endl;
    get_depth_data_sevenscenes(depth_list[curr_frame], depth_data);

    // if (curr_frame == 0) {
    //   int bestValue = 0;
    //   for (int i = 0; i < kImageRows * kImageCols ; i++)
    //     if ((int)depth_data[i] > bestValue)
    //       bestValue = (int)depth_data[i];
    //   std::cout << bestValue << std::endl;
    // }

    // Compute camera Rt between current frame and base frame (saved in cam_R and cam_t)
    // then return view frustum bounds for volume (saved in range_grid)
    float cam_R[9] = {0};
    float cam_t[3] = {0};
    float range_grid[6] = {0};
    compute_frustum_bounds(extrinsic_poses, first_frame, curr_frame, cam_R, cam_t, range_grid);

    // Integrate
    time_t tstart, tend;
    tstart = time(0);
    integrate(depth_data, range_grid, cam_R, cam_t);
    tend = time(0);

    // Clear memory
    // free(image_data);
    free(depth_data);

    // Print time
    std::cout << " done!" << std::endl;
    // std::cout << "Integrating took " << difftime(tend, tstart) << " second(s)." << std::endl;

    // Compute intersection between view frustum and current volume
    float view_occupancy = (range_grid[1] - range_grid[0]) * (range_grid[3] - range_grid[2]) * (range_grid[5] - range_grid[4]) / (512 * 512 * 1024);
    // std::cout << "Intersection: " << 100 * view_occupancy << "%" << std::endl;

    if ((curr_frame - first_frame >= 30) || curr_frame == total_files - 1) {

      // Find keypoints
      find_keypoints(range_grid);

      // Save curr volume to file
      std::string volume_name = sequence_prefix + "scene" +  std::to_string(first_frame) + "_" + std::to_string(curr_frame);

      std::string scene_ply_name = volume_name + ".ply";
      save_volume_to_ply(scene_ply_name);

      std::string scene_tsdf_name = volume_name + ".tsdf";
      save_volume_to_tsdf(scene_tsdf_name);

      std::string scene_keypoints_name = volume_name + "_pts.txt";
      save_volume_keypoints(scene_keypoints_name);

      std::string scene_extrinsics_name = volume_name + "_ext.txt";
      save_volume_to_world_matrix(scene_extrinsics_name, extrinsic_poses, first_frame);


      // // Check for loop closure and set a base frame
      base_frame = curr_frame;
      // float base_frame_intersection = 0;
      // for (int i = 0; i < base_frames.size(); i++) {
      //   float cam_R[9] = {0};
      //   float cam_t[3] = {0};
      //   float range_grid[6] = {0};
      //   compute_frustum_bounds(extrinsic_poses, base_frames[i], curr_frame, cam_R, cam_t, range_grid);
      //   float view_occupancy = (range_grid[1]-range_grid[0])*(range_grid[3]-range_grid[2])*(range_grid[5]-range_grid[4])/(512*512*1024);
      //   if (view_occupancy > std::max(0.7f, base_frame_intersection)) {
      //     base_frame = base_frames[i];
      //     base_frame_intersection = view_occupancy;
      //   }
      // }
      // base_frames.push_back(base_frame);

      // Init new volume
      first_frame = curr_frame;
      if (!(curr_frame == total_files - 1))
        curr_frame = curr_frame - 1;
      std::cout << "Creating new volume." << std::endl;
      memset(voxel_volume.weight, 0, sizeof(float) * 512 * 512 * 1024);
      voxel_volume.keypoints.clear();
      for (int i = 0; i < 512 * 512 * 1024; i++)
        voxel_volume.tsdf[i] = 1.0f;

    }

    // std::cout << std::endl;
  }

}


void tsdf2ply(const std::string &filename, float* scene_tsdf, float tsdf_threshold, float* ext_mat, int x_dim, int y_dim, int z_dim, bool use_ext, keypoint pc_color) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < x_dim * y_dim * z_dim; i++)
    if (std::abs(scene_tsdf[i]) < tsdf_threshold)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
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
  for (int z = 0; z < z_dim; z++) {
    for (int y = 0; y < y_dim; y++) {
      for (int x = 0; x < x_dim; x++) {
        if (std::abs(scene_tsdf[z * y_dim * x_dim + y * x_dim + x]) < tsdf_threshold) {

          // grid to world coords (7scene)
          // float sx = ((float)x + 1) * 0.01 - 512 * 0.01 / 2;
          // float sy = ((float)y + 1) * 0.01 - 512 * 0.01 / 2;
          // float sz = ((float)z + 1) * 0.01 - 0.5;

          // (synth)
          // float sx = (float)x * 0.01;
          // float sy = (float)y * 0.01;
          // float sz = (float)z * 0.01;

          if (use_ext) {
            float sx = (float)x;
            float sy = (float)y;
            float sz = (float)z;
            float fx = ext_mat[0] * sx + ext_mat[1] * sy + ext_mat[2] * sz;
            float fy = ext_mat[4] * sx + ext_mat[5] * sy + ext_mat[6] * sz;
            float fz = ext_mat[8] * sx + ext_mat[9] * sy + ext_mat[10] * sz;
            fx = fx + ext_mat[3];
            fy = fy + ext_mat[7];
            fz = fz + ext_mat[11];
            fwrite(&fx, sizeof(float), 1, fp);
            fwrite(&fy, sizeof(float), 1, fp);
            fwrite(&fz, sizeof(float), 1, fp);

            uchar r = (uchar) pc_color.x;
            uchar g = (uchar) pc_color.y;
            uchar b = (uchar) pc_color.z;
            fwrite(&r, sizeof(uchar), 1, fp);
            fwrite(&g, sizeof(uchar), 1, fp);
            fwrite(&b, sizeof(uchar), 1, fp);
          } else {
            float sx = (float)x;
            float sy = (float)y;
            float sz = (float)z;
            fwrite(&sx, sizeof(float), 1, fp);
            fwrite(&sy, sizeof(float), 1, fp);
            fwrite(&sz, sizeof(float), 1, fp);

            uchar r = (uchar) pc_color.x;
            uchar g = (uchar) pc_color.y;
            uchar b = (uchar) pc_color.z;
            fwrite(&r, sizeof(uchar), 1, fp);
            fwrite(&g, sizeof(uchar), 1, fp);
            fwrite(&b, sizeof(uchar), 1, fp);
          }


          // transform
          // float fx = ext_mat[0] * sx + ext_mat[1] * sy + ext_mat[2] * sz;
          // float fy = ext_mat[4] * sx + ext_mat[5] * sy + ext_mat[6] * sz;
          // float fz = ext_mat[8] * sx + ext_mat[9] * sy + ext_mat[10] * sz;
          // fx = fx + ext_mat[3];
          // fy = fy + ext_mat[7];
          // fz = fz + ext_mat[11];
          // fwrite(&fx, sizeof(float), 1, fp);
          // fwrite(&fy, sizeof(float), 1, fp);
          // fwrite(&fz, sizeof(float), 1, fp);
        }
      }
    }
  }
  fclose(fp);
}
