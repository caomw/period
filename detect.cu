#include "util/util.hpp"
#include "kinfu.hpp"
#include <opencv2/opencv.hpp>






////////////////////////////////////////////////////////////////////////////////
// Import Marvin

#define DATATYPE 1 // Marvin datatype
#include "marvin.hpp"

////////////////////////////////////////////////////////////////////////////////
// Global variables for Marvin
std::string model_idx = "4";
marvin::Net main_net("models/model" + model_idx + ".test.json");

// Init marvin net
void init_marvin() {
  main_net.Malloc(marvin::Testing);
  std::vector<std::string> models = marvin::getStringVector("models/PeriodNet.5." + model_idx + "_snapshot_25000.marvin");
  for (int m=0;m<models.size();++m)   
    main_net.loadWeights(models[m]);
//     // marvin::Net net("tools/marvin/model" + model_idx + ".test.json");
//     main_net.Malloc(marvin::Testing);
//     std::vector<std::string> models = marvin::getStringVector("tools/marvin/PeriodNet.1." + model_idx + ".60000.marvin");
//     for (int m=0;m<models.size();++m)   
//       main_net.loadWeights(models[m]);
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
    object_pose_display_3D[0 * 6 + i * 2 + 0] = object_pose[0 * 4 + 3] - object_pose[0 * 4 + i] * 0.07f;
    object_pose_display_3D[0 * 6 + i * 2 + 1] = object_pose[0 * 4 + 3] + object_pose[0 * 4 + i] * 0.07f;
    object_pose_display_3D[1 * 6 + i * 2 + 0] = object_pose[1 * 4 + 3] - object_pose[1 * 4 + i] * 0.07f;
    object_pose_display_3D[1 * 6 + i * 2 + 1] = object_pose[1 * 4 + 3] + object_pose[1 * 4 + i] * 0.07f;
    object_pose_display_3D[2 * 6 + i * 2 + 0] = object_pose[2 * 4 + 3] - object_pose[2 * 4 + i] * 0.07f;
    object_pose_display_3D[2 * 6 + i * 2 + 1] = object_pose[2 * 4 + 3] + object_pose[2 * 4 + i] * 0.07f;
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
  cv::line(display_frame, cv::Point(object_pose_display_2D[0], object_pose_display_2D[6]), cv::Point(object_pose_display_2D[1], object_pose_display_2D[7]), cv::Scalar(0, 0, 255), 2);
  cv::line(display_frame, cv::Point(object_pose_display_2D[2], object_pose_display_2D[8]), cv::Point(object_pose_display_2D[3], object_pose_display_2D[9]), cv::Scalar(0, 255, 0), 2);
  cv::line(display_frame, cv::Point(object_pose_display_2D[4], object_pose_display_2D[10]), cv::Point(object_pose_display_2D[5], object_pose_display_2D[11]), cv::Scalar(255, 0, 0), 2);
  cv::circle(display_frame, cv::Point(object_center_display_2D[0], object_center_display_2D[1]), 4, cv::Scalar(0, 255, 255), -1);
  cv::circle(display_frame, cv::Point(object_pose_display_2D[0], object_pose_display_2D[6]), 4, cv::Scalar(0, 0, 255), -1);
  cv::circle(display_frame, cv::Point(object_pose_display_2D[2], object_pose_display_2D[8]), 4, cv::Scalar(0, 255, 0), -1);
  cv::circle(display_frame, cv::Point(object_pose_display_2D[4], object_pose_display_2D[10]), 4, cv::Scalar(255, 0, 0), -1);
  // cv::namedWindow("Object Pose", CV_WINDOW_AUTOSIZE);
  // cv::imshow("Object Pose", display_frame);
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

void detect(const std::string &sequence_directory, const std::string &frame_prefix) {

  std::cout << "CPU: Loading RGB-D frame and camera info." << std::endl;

  // Load intrinsics (3x3 matrix)
  std::string intrinsic_filename = sequence_directory + "/intrinsics.K.txt";
  std::vector<float> K_vec = load_matrix_from_file(intrinsic_filename, 3, 3);
  float * K = &K_vec[0];
  // for (int i = 0; i < 9; i++)
  //   std::cout << K[i] << std::endl;

  // Load RGB-D frame
  std::string curr_frame_color_filename = sequence_directory + "/" + frame_prefix + ".color.png";
  cv::Mat curr_frame_color = cv::imread(curr_frame_color_filename.c_str(), 1);
  std::string curr_frame_depth_filename = sequence_directory + "/" + frame_prefix + ".depth.png";
  cv::Mat curr_frame_depth = cv::imread(curr_frame_depth_filename.c_str(), CV_LOAD_IMAGE_UNCHANGED);

  // Load image/depth/extrinsic data for current frame
  unsigned short * depth_data = (unsigned short *) malloc(480 * 640 * sizeof(unsigned short));
  for (int i = 0; i < 480 * 640; i++) {
    depth_data[i] = (((unsigned short) curr_frame_depth.data[i * 2 + 1]) << 8) + ((unsigned short) curr_frame_depth.data[i * 2 + 0]);
    // std::cout << depth_data[i] << std::endl;
  }

  std::cout << "GPU: Fusing depth into TSDF volume." << std::endl;
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

  // Save curr volume to pointcloud file
  std::string scene_ply_name = "volume.pointcloud.ply";
  save_volume_to_ply(scene_ply_name, vox_size, vox_tsdf);

  // // Save curr volume to raw file
  // std::string volume_name = "volume.tsdf.bin";
  // std::ofstream outFile(volume_name, std::ios::binary | std::ios::out);
  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   outFile.write((char*)&vox_tsdf[i], sizeof(float));
  // outFile.close();

  std::cout << "GPU: Exhaustively generating sliding windows for object detection." << std::endl;

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
        // std::cout << hypothesis_idx << std::endl;
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

  // ROS_INFO("Found %d hypothesis bounding boxes.", num_valid_hypotheses);
  // ROS_INFO("Saving hypotheses to tensors on disk for Marvin.");
  std::cout << "GPU: Found " << num_valid_hypotheses << " hypothesis bounding boxes." << std::endl;
  std::cout << "CPU: Passing hypothesis bounding boxes to Marvin." << std::endl;

  buffer_data2D.clear();
  buffer_data3D.clear();
  buffer_data2D.resize(num_valid_hypotheses);
  buffer_data3D.resize(num_valid_hypotheses);

  // Write hypothesis cubes and patches to tensor file
  int valid_hypothesis_counter = 0;
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
      buffer_data2D[valid_hypothesis_counter].resize(3 * 227 * 227);
      for (int i = 0; i < 3 * 227 * 227; i++)
        buffer_data2D[valid_hypothesis_counter][i] = patch_data[i];

      // Write 3D tsdf volume to data tensor file
      buffer_data3D[valid_hypothesis_counter].resize(30 * 30 * 30);
      for (int i = 0; i < 30 * 30 * 30; i++)
        buffer_data3D[valid_hypothesis_counter][i] = curr_cube[i];

      valid_hypothesis_counter++;

      // Clear memory
      delete [] patch_data;
      delete [] curr_cube;
    }
  }

  // Clear memory
  free(depth_data);
  cudaFree(d_hypothesis_locations);
  cudaFree(d_hypothesis_labels);
  cudaFree(d_hypothesis_crop_2D);
  toc();

  // Temporarily clear TSDF volume on GPU
  // cudaFree(d_vox_tsdf);
  // cudaFree(d_vox_weight);

  // Run marvin
  std::cerr << "GPU: Running Marvin for 2D/3D deep learning." << std::endl;
  // std::string class_score_tensor_filename = "TMP.class.response.tensor";
  // std::string quaternion_score_tensor_filename = "TMP.quaternion.response.tensor";
  // std::string translation_score_tensor_filename = "TMP.quaternion.response.tensor";
  // std::string axis_score_tensor_filename = "TMP.axis_score_response.tensor";
  // std::string angle_score_tensor_filename = "TMP.angle_score_response.tensor";
  if (true) {
    int itersPerSave = 0;
    main_net.test(marvin::getStringVector("class_score,quat_pred,trans_pred"), marvin::getStringVector(""), itersPerSave);
    // sys_command("cd src/apc_vision/tools/marvin; export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cudnn/v4rc/lib64; ./marvin test model.json PeriodNet.marvin class_score ../../../../" + class_score_tensor_filename);
  }
  std::cout << "CPU: Extracting object pose information from Marvin results." << std::endl;

  // Parse Marvin scores
  float * class_score_raw = &buffer_scores_class[0];
  float * quaternion_score_raw = &buffer_scores_quaternion[0];
  float * translation_score_raw = &buffer_scores_translation[0];

  // List objects
  std::vector<std::string> object_names;
  // object_names.push_back("book");
  // object_names.push_back("duck");
  object_names.push_back("expo");
  // object_names.push_back("frog");
  object_names.push_back("glue");
  // object_names.push_back("plugs");
  // object_names.push_back("spark");

  float * highest_class_scores = new float[object_names.size()];
  for (int i = 0; i < object_names.size(); i++)
    highest_class_scores[i] = 0;
  int * best_guess_IDX = new int[object_names.size()];
  float * best_guess_quaternion = new float[object_names.size() * 4];
  float * best_guess_translation = new float[object_names.size() * 3];
  int valid_hypothesis_idx = 0;
  for (int hypothesis_idx = 0; hypothesis_idx < num_hypothesis; hypothesis_idx++) {
    if ((int)(hypothesis_labels[hypothesis_idx]) == 1) {
      // std::cout << hypothesis_idx << " " << num_hypothesis << " " << valid_hypothesis_idx << std::endl;

      for (int j = 0; j < (object_names.size()+1); j++)
        std::cout << class_score_raw[valid_hypothesis_idx * (object_names.size()+1) + j] << " ";
      std::cout << std::endl;

      // Loop through each object type
      for (int i = 0; i < object_names.size(); i++) {

        // for (int j = 0; j < 4; j++)
        //   std::cout << std::to_string(i) << ": " << quaternion_score_raw[valid_hypothesis_idx * object_names.size() * 4 + i * 4 + j] << " ";
        // std::cout << std::endl;

        for (int j = 0; j < 3; j++)
          std::cout << std::to_string(i) << ": " << translation_score_raw[valid_hypothesis_idx * object_names.size() * 3 + i * 3 + j] << " ";
        std::cout << std::endl;

        float curr_class_score_raw = class_score_raw[valid_hypothesis_idx * (object_names.size() + 1) + i + 1];
        if (curr_class_score_raw > 0.5f) {

          // Draw top scoring boxes
          std::cout << valid_hypothesis_idx << std::endl;
          int crop_x1 = hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx];
          int crop_y1 = hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx];
          int crop_x2 = hypothesis_crop_2D[0 * num_hypothesis + hypothesis_idx] + hypothesis_crop_2D[2 * num_hypothesis + hypothesis_idx];
          int crop_y2 = hypothesis_crop_2D[1 * num_hypothesis + hypothesis_idx] + hypothesis_crop_2D[3 * num_hypothesis + hypothesis_idx];
          cv::rectangle(curr_frame_color, cv::Point(crop_x1, crop_y1), cv::Point(crop_x2, crop_y2), cv::Scalar(255, 0, 0));
          float object_location[3] = {0};
          object_location[0] = (hypothesis_locations[0 * num_hypothesis + hypothesis_idx] + 1) * vox_unit + vox_range_cam[0 * 2 + 0];
          object_location[1] = (hypothesis_locations[1 * num_hypothesis + hypothesis_idx] + 1) * vox_unit + vox_range_cam[1 * 2 + 0];
          object_location[2] = (hypothesis_locations[2 * num_hypothesis + hypothesis_idx] + 1) * vox_unit + vox_range_cam[2 * 2 + 0];
          float object_location_2D[2] = {0};
          object_location_2D[0] = (object_location[0] * K[0]) / (object_location[2]) + K[2];
          object_location_2D[1] = (object_location[1] * K[4]) / (object_location[2]) + K[5];
          cv::circle(curr_frame_color, cv::Point(object_location_2D[0], object_location_2D[1]), 4, cv::Scalar(255, 0, 0), -1);
          object_location[0] = object_location[0] + translation_score_raw[valid_hypothesis_idx * object_names.size() * 3 + i * 3 + 0]*(0.03f/std::sqrt(2.0f));
          object_location[1] = object_location[1] + translation_score_raw[valid_hypothesis_idx * object_names.size() * 3 + i * 3 + 1]*(0.03f/std::sqrt(2.0f));
          object_location[2] = object_location[2] + translation_score_raw[valid_hypothesis_idx * object_names.size() * 3 + i * 3 + 2]*(0.03f/std::sqrt(2.0f));
          object_location_2D[0] = (object_location[0] * K[0]) / (object_location[2]) + K[2];
          object_location_2D[1] = (object_location[1] * K[4]) / (object_location[2]) + K[5];
          cv::circle(curr_frame_color, cv::Point(object_location_2D[0], object_location_2D[1]), 4, cv::Scalar(0, 255, 0), -1);

          if (false) {
            float curr_quaternion[4] = {0};
            for (int j = 0; j < 4; j++)
              curr_quaternion[j] = quaternion_score_raw[valid_hypothesis_idx * object_names.size() * 4 + i * 4 + j];

            // Normalize quaternion 
            const float curr_quaternion_norm = 1.0f / sqrt(curr_quaternion[0] * curr_quaternion[0] + 
                                                                 curr_quaternion[1] * curr_quaternion[1] +
                                                                 curr_quaternion[2] * curr_quaternion[2] +
                                                                 curr_quaternion[3] * curr_quaternion[3]);
            curr_quaternion[0] *= curr_quaternion_norm;
            curr_quaternion[1] *= curr_quaternion_norm;
            curr_quaternion[2] *= curr_quaternion_norm;
            curr_quaternion[3] *= curr_quaternion_norm;
            std::cout << "Quaternion (normalized): " << curr_quaternion[0] << " " << curr_quaternion[1] << " " << curr_quaternion[2] << " " << curr_quaternion[3] << std::endl;

            // Convert quaternion to pose
            float object_pose[16];
            object_pose[0 * 4 + 0] = 1.0f - 2.0f * curr_quaternion[2] * curr_quaternion[2] - 2.0f * curr_quaternion[3] * curr_quaternion[3];
            object_pose[0 * 4 + 1] = 2.0f * curr_quaternion[1] * curr_quaternion[2] - 2.0f * curr_quaternion[3] * curr_quaternion[0];
            object_pose[0 * 4 + 2] = 2.0f * curr_quaternion[1] * curr_quaternion[3] + 2.0f * curr_quaternion[2] * curr_quaternion[0];
            object_pose[0 * 4 + 3] = object_location[0];
            object_pose[1 * 4 + 0] = 2.0f * curr_quaternion[1] * curr_quaternion[2] + 2.0f * curr_quaternion[3] * curr_quaternion[0];
            object_pose[1 * 4 + 1] = 1.0f - 2.0f * curr_quaternion[1] * curr_quaternion[1] - 2.0f * curr_quaternion[3] * curr_quaternion[3];
            object_pose[1 * 4 + 2] = 2.0f * curr_quaternion[2] * curr_quaternion[3] - 2.0f * curr_quaternion[1] * curr_quaternion[0];
            object_pose[1 * 4 + 3] = object_location[1];
            object_pose[2 * 4 + 0] = 2.0f * curr_quaternion[1] * curr_quaternion[3] - 2.0f * curr_quaternion[2] * curr_quaternion[0];
            object_pose[2 * 4 + 1] = 2.0f * curr_quaternion[2] * curr_quaternion[3] + 2.0f * curr_quaternion[1] * curr_quaternion[0];
            object_pose[2 * 4 + 2] = 1.0f - 2.0f * curr_quaternion[1] * curr_quaternion[1] - 2.0f * curr_quaternion[2] * curr_quaternion[2];
            object_pose[2 * 4 + 3] = object_location[2];
            object_pose[3 * 4 + 0] = 0.0f;
            object_pose[3 * 4 + 1] = 0.0f;
            object_pose[3 * 4 + 2] = 0.0f;
            object_pose[3 * 4 + 3] = 1.0f;

            // Display object pose
            show_object_pose(K, object_pose, curr_frame_color);
          }

          if (curr_class_score_raw > highest_class_scores[i]) {
            highest_class_scores[i] = curr_class_score_raw;
            best_guess_IDX[i] = hypothesis_idx;

            // Get best guess quaternion
            for (int j = 0; j < 4; j++)
              best_guess_quaternion[i * 4 + j] = quaternion_score_raw[valid_hypothesis_idx * object_names.size() * 4 + i * 4 + j];

            // Get best guess translation
            for (int j = 0; j < 3; j++)
              best_guess_translation[i * 3 + j] = translation_score_raw[valid_hypothesis_idx * object_names.size() * 3 + i * 3 + j];

          }
        }
      }
      valid_hypothesis_idx++;
    }
  }

      // std::cout << "got here" << std::endl;
  // // If no objects are detected
  // if (highest_class_scores[0] == 0 && highest_class_scores[1] == 0) {
  //   std::cout << "No objects detected!" << std::endl;
  //   object_names.clear();
  // }

  for (int object_idx = 0; object_idx < object_names.size(); object_idx++) {

    if (highest_class_scores[object_idx] == 0)
      continue;

    // Display detection results
    int crop_x1 = hypothesis_crop_2D[0 * num_hypothesis + best_guess_IDX[object_idx]];
    int crop_y1 = hypothesis_crop_2D[1 * num_hypothesis + best_guess_IDX[object_idx]];
    int crop_x2 = hypothesis_crop_2D[0 * num_hypothesis + best_guess_IDX[object_idx]] + hypothesis_crop_2D[2 * num_hypothesis + best_guess_IDX[object_idx]];
    int crop_y2 = hypothesis_crop_2D[1 * num_hypothesis + best_guess_IDX[object_idx]] + hypothesis_crop_2D[3 * num_hypothesis + best_guess_IDX[object_idx]];
    cv::rectangle(curr_frame_color, cv::Point(crop_x1, crop_y1), cv::Point(crop_x2, crop_y2), cv::Scalar(0, 255, 0), 2);
    // cv::circle(curr_frame_color, cv::Point((crop_x1 + crop_x2) / 2, (crop_y1 + crop_y2) / 2), 5, cv::Scalar(0, 255, 0), -1);

    std::cout << "Quaternion (raw): " << best_guess_quaternion[object_idx * 4 + 0] << " " << best_guess_quaternion[object_idx * 4 + 1] << " " << best_guess_quaternion[object_idx * 4 + 2] << " " << best_guess_quaternion[object_idx * 4 + 3] << std::endl;

    // Retrieve object location
    float object_location[3] = {0};
    object_location[0] = (hypothesis_locations[0 * num_hypothesis + best_guess_IDX[object_idx]] + 1) * vox_unit + vox_range_cam[0 * 2 + 0];
    object_location[1] = (hypothesis_locations[1 * num_hypothesis + best_guess_IDX[object_idx]] + 1) * vox_unit + vox_range_cam[1 * 2 + 0];
    object_location[2] = (hypothesis_locations[2 * num_hypothesis + best_guess_IDX[object_idx]] + 1) * vox_unit + vox_range_cam[2 * 2 + 0];

    // Apply translation prediction to object location
    std::cout << "Translation (raw): " << best_guess_translation[object_idx * 3 + 0] << " " << best_guess_translation[object_idx * 3 + 1] << " " << best_guess_translation[object_idx * 3 + 2] << std::endl;
    object_location[0] = object_location[0] + best_guess_translation[object_idx * 3 + 0]*(0.03f/std::sqrt(2.0f));
    object_location[1] = object_location[1] + best_guess_translation[object_idx * 3 + 1]*(0.03f/std::sqrt(2.0f));
    object_location[2] = object_location[2] + best_guess_translation[object_idx * 3 + 2]*(0.03f/std::sqrt(2.0f));

    // Normalize quaternion 
    const float best_guess_quaternion_norm = 1.0f / sqrt(best_guess_quaternion[object_idx * 4 + 0] * best_guess_quaternion[object_idx * 4 + 0] + best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 1] + best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 2] + best_guess_quaternion[object_idx * 4 + 3] * best_guess_quaternion[object_idx * 4 + 3]);
    best_guess_quaternion[object_idx * 4 + 0] *= best_guess_quaternion_norm;
    best_guess_quaternion[object_idx * 4 + 1] *= best_guess_quaternion_norm;
    best_guess_quaternion[object_idx * 4 + 2] *= best_guess_quaternion_norm;
    best_guess_quaternion[object_idx * 4 + 3] *= best_guess_quaternion_norm;
    std::cout << "Quaternion (normalized): " << best_guess_quaternion[object_idx * 4 + 0] << " " << best_guess_quaternion[object_idx * 4 + 1] << " " << best_guess_quaternion[object_idx * 4 + 2] << " " << best_guess_quaternion[object_idx * 4 + 3] << std::endl;

    // Convert quaternion to pose
    float object_pose[16];
    object_pose[0 * 4 + 0] = 1.0f - 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 2] - 2.0f * best_guess_quaternion[object_idx * 4 + 3] * best_guess_quaternion[object_idx * 4 + 3];
    object_pose[0 * 4 + 1] = 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 2] - 2.0f * best_guess_quaternion[object_idx * 4 + 3] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[0 * 4 + 2] = 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 3] + 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[0 * 4 + 3] = object_location[0];
    object_pose[1 * 4 + 0] = 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 2] + 2.0f * best_guess_quaternion[object_idx * 4 + 3] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[1 * 4 + 1] = 1.0f - 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 1] - 2.0f * best_guess_quaternion[object_idx * 4 + 3] * best_guess_quaternion[object_idx * 4 + 3];
    object_pose[1 * 4 + 2] = 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 3] - 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[1 * 4 + 3] = object_location[1];
    object_pose[2 * 4 + 0] = 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 3] - 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[2 * 4 + 1] = 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 3] + 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 0];
    object_pose[2 * 4 + 2] = 1.0f - 2.0f * best_guess_quaternion[object_idx * 4 + 1] * best_guess_quaternion[object_idx * 4 + 1] - 2.0f * best_guess_quaternion[object_idx * 4 + 2] * best_guess_quaternion[object_idx * 4 + 2];
    object_pose[2 * 4 + 3] = object_location[2];
    object_pose[3 * 4 + 0] = 0.0f;
    object_pose[3 * 4 + 1] = 0.0f;
    object_pose[3 * 4 + 2] = 0.0f;
    object_pose[3 * 4 + 3] = 1.0f;

    // Display object pose
    show_object_pose(K, object_pose, curr_frame_color);

    // Show object class label
    cv::putText(curr_frame_color, "Class: " + object_names[object_idx], cv::Point(crop_x1 + 5, crop_y2 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);

    // // Compare against ground truth
    // if (true) {
    //   // Read ground truth object pose from file
    //   std::string gt_object_pose_filename = sequence_directory + "/object.pose.txt";
    //   std::vector<float> gt_object_pose_raw = load_matrix_from_file(gt_object_pose_filename, 4, 4);
    //   float * gt_object_pose_arr = &gt_object_pose_raw[0];

    //   // Compute ground truth object pose w.r.t. current camera pose
    //   std::string gt_cam_pose_filename = sequence_directory + "/" + frame_prefix + ".pose.txt";
    //   std::vector<float> gt_cam_pose_raw = load_matrix_from_file(gt_cam_pose_filename, 4, 4);
    //   float * gt_cam_pose_arr = &gt_cam_pose_raw[0];
    //   float gt_cam_pose_inv[16] = {0};
    //   invert_matrix(gt_cam_pose_arr, gt_cam_pose_inv);
    //   float gt_object_pose[16] = {0};
    //   multiply_matrix(gt_cam_pose_inv, gt_object_pose_arr, gt_object_pose);

    //   // Compute center of ground truth object in 3D camera coordinates
    //   float gt_object_center_cam[3] = {0};
    //   for (int i = 0; i < 3; i++)
    //     gt_object_center_cam[i] = gt_object_pose[i * 4 + 3];

    //   // Convert ground truth pose from rotation matrix to quaternion
    //   float trace = gt_object_pose[0 * 4 + 0] + gt_object_pose[1 * 4 + 1] + gt_object_pose[2 * 4 + 2]; // I removed + 1.0f; see discussion with Ethan
    //   float gt_object_pose_quaternion[4];
    //   if ( trace > 0 ) { // I changed M_EPSILON to 0
    //     float s = 0.5f / sqrtf(trace + 1.0f);
    //     gt_object_pose_quaternion[0] = 0.25f / s;
    //     gt_object_pose_quaternion[1] = ( gt_object_pose[2 * 4 + 1] - gt_object_pose[1 * 4 + 2] ) * s;
    //     gt_object_pose_quaternion[2] = ( gt_object_pose[0 * 4 + 2] - gt_object_pose[2 * 4 + 0] ) * s;
    //     gt_object_pose_quaternion[3] = ( gt_object_pose[1 * 4 + 0] - gt_object_pose[0 * 4 + 1] ) * s;
    //   } else {
    //     if ( gt_object_pose[0 * 4 + 0] > gt_object_pose[1 * 4 + 1] && gt_object_pose[0 * 4 + 0] > gt_object_pose[2 * 4 + 2] ) {
    //       float s = 2.0f * sqrtf( 1.0f + gt_object_pose[0 * 4 + 0] - gt_object_pose[1 * 4 + 1] - gt_object_pose[2 * 4 + 2]);
    //       gt_object_pose_quaternion[0] = (gt_object_pose[2 * 4 + 1] - gt_object_pose[1 * 4 + 2] ) / s;
    //       gt_object_pose_quaternion[1] = 0.25f * s;
    //       gt_object_pose_quaternion[2] = (gt_object_pose[0 * 4 + 1] + gt_object_pose[1 * 4 + 0] ) / s;
    //       gt_object_pose_quaternion[3] = (gt_object_pose[0 * 4 + 2] + gt_object_pose[2 * 4 + 0] ) / s;
    //     } else if (gt_object_pose[1 * 4 + 1] > gt_object_pose[2 * 4 + 2]) {
    //       float s = 2.0f * sqrtf( 1.0f + gt_object_pose[1 * 4 + 1] - gt_object_pose[0 * 4 + 0] - gt_object_pose[2 * 4 + 2]);
    //       gt_object_pose_quaternion[0] = (gt_object_pose[0 * 4 + 2] - gt_object_pose[2 * 4 + 0] ) / s;
    //       gt_object_pose_quaternion[1] = (gt_object_pose[0 * 4 + 1] + gt_object_pose[1 * 4 + 0] ) / s;
    //       gt_object_pose_quaternion[2] = 0.25f * s;
    //       gt_object_pose_quaternion[3] = (gt_object_pose[1 * 4 + 2] + gt_object_pose[2 * 4 + 1] ) / s;
    //     } else {
    //       float s = 2.0f * sqrtf( 1.0f + gt_object_pose[2 * 4 + 2] - gt_object_pose[0 * 4 + 0] - gt_object_pose[1 * 4 + 1] );
    //       gt_object_pose_quaternion[0] = (gt_object_pose[1 * 4 + 0] - gt_object_pose[0 * 4 + 1] ) / s;
    //       gt_object_pose_quaternion[1] = (gt_object_pose[0 * 4 + 2] + gt_object_pose[2 * 4 + 0] ) / s;
    //       gt_object_pose_quaternion[2] = (gt_object_pose[1 * 4 + 2] + gt_object_pose[2 * 4 + 1] ) / s;
    //       gt_object_pose_quaternion[3] = 0.25f * s;
    //     }
    //   }

    //   float obj_dist = sqrtf((gt_object_center_cam[0] - object_location[0]) * (gt_object_center_cam[0] - object_location[0]) +
    //                          (gt_object_center_cam[1] - object_location[1]) * (gt_object_center_cam[1] - object_location[1]) +
    //                          (gt_object_center_cam[2] - object_location[2]) * (gt_object_center_cam[2] - object_location[2]));

    //   if (obj_dist < 0.02f)
    //     cv::putText(curr_frame_color, "Detection: < 2cm conf: " + std::to_string(highest_class_scores[object_idx]), cv::Point(crop_x1 + 5, crop_y2 - 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
    //   else
    //     cv::putText(curr_frame_color, "Detection: > 2cm conf: " + std::to_string(highest_class_scores[object_idx]), cv::Point(crop_x1 + 5, crop_y2 - 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);

    //   cv::putText(curr_frame_color, "PR: " + std::to_string(best_guess_quaternion[object_idx * 4 + 0]) + " " + std::to_string(best_guess_quaternion[object_idx * 4 + 1]) + " " + std::to_string(best_guess_quaternion[object_idx * 4 + 2]) + " " + std::to_string(best_guess_quaternion[object_idx * 4 + 3]), cv::Point(crop_x1 + 5, crop_y2 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
    //   cv::putText(curr_frame_color, "GT: " + std::to_string(gt_object_pose_quaternion[0]) + " " + std::to_string(gt_object_pose_quaternion[1]) + " " + std::to_string(gt_object_pose_quaternion[2]) + " " + std::to_string(gt_object_pose_quaternion[3]), cv::Point(crop_x1 + 5, crop_y2 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);

    // }
  }

  cv::namedWindow("Object Poses", CV_WINDOW_AUTOSIZE);
  cv::imshow("Object Poses", curr_frame_color);
  cv::waitKey(100);

  // Save display result
  std::string results_directory = sequence_directory + "/results." + model_idx;
  sys_command("mkdir -p " + results_directory);
  cv::imwrite(results_directory + "/" + frame_prefix + ".result.png", curr_frame_color);

  // Re-alloc and re-init TSDF on GPU
  num_blocks = vox_size[2];
  num_threads = vox_size[1];
  reset_vox_whole_GPU<<<num_blocks,num_threads>>>(d_vox_size, d_vox_tsdf, d_vox_weight);
  // memset(vox_weight, 0, sizeof(float) * vox_size[0] * vox_size[1] * vox_size[2]);
  // for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
  //   vox_tsdf[i] = 1.0f;
  // kCheckCUDA(__LINE__, cudaMalloc(&d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float)));
  // kCheckCUDA(__LINE__, cudaMalloc(&d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float)));
  // kCheckCUDA(__LINE__, cudaMemcpy(d_vox_tsdf, vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice));
  // kCheckCUDA(__LINE__, cudaMemcpy(d_vox_weight, vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice));

  // Clear other excess memory
  // delete [] class_score_raw;
  // delete [] axis_score_raw;
  // delete [] angle_score_raw;
  delete [] hypothesis_locations;
  delete [] hypothesis_labels;
  delete [] hypothesis_crop_2D;
}

int main(int argc, char **argv) {



  init_fusion_GPU();

  init_marvin();
  

  // tic();
  // detect("data/train/expo/000000","frame-000000");
  // toc();

  // // List RGB-D sequences
  // std::string object_directory = "data/train/expo";
  // std::vector<std::string> sequence_names;
  // get_files_in_directory(object_directory, sequence_names, "");
  // int rand_sequence_idx = (int)floor(gen_random_float(0, (float)sequence_names.size()));

  // for (int sequence_idx = 0; sequence_idx < sequence_names.size(); sequence_idx++) {
  //   std::string curr_sequence_name = sequence_names[sequence_idx];
  //   std::string curr_sequence_directory = object_directory + "/" + curr_sequence_name;

  //   // List RGB-D frames
  //   std::vector<std::string> frame_names;
  //   get_files_in_directory(curr_sequence_directory, frame_names, ".color.png");
  //   for (int frame_idx = 0; frame_idx < frame_names.size(); frame_idx++) {
  //     std::string curr_frame_name = frame_names[frame_idx];
  //     curr_frame_name = curr_frame_name.substr(0, curr_frame_name.length() - 10);
  //     tic();
  //     detect(curr_sequence_directory,curr_frame_name);
  //     toc();
  //   }
  // }


  std::string curr_sequence_directory = "data/train/expo/000004";
  // List RGB-D frames
  std::vector<std::string> frame_names;
  get_files_in_directory(curr_sequence_directory, frame_names, ".color.png");
  for (int frame_idx = 0; frame_idx < frame_names.size(); frame_idx++) {
    std::string curr_frame_name = frame_names[frame_idx];
    curr_frame_name = curr_frame_name.substr(0, curr_frame_name.length() - 10);
    tic();
    detect(curr_sequence_directory,curr_frame_name);
    toc();
  }








  return 0;
}



