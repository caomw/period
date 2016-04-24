#include "util/util.hpp"
#include <librealsense/rs.hpp>
#include "display.hpp"
#include <png++/png.hpp>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <thread>

texture_buffer buffers[6];

#pragma pack(push, 1)
struct rgb_pixel
{
    uint8_t r, g, b;
};
#pragma pack(pop)

int main(int argc, char * argv[]) try {
    rs::log_to_console(rs::log_severity::warn);
    //rs::log_to_file(rs::log_severity::debug, "librealsense.log");

    rs::context ctx;
    if (ctx.get_device_count() == 0) throw std::runtime_error("No device detected. Is it plugged in?");
    rs::device & dev = *ctx.get_device(0);

    dev.enable_stream(rs::stream::depth, rs::preset::best_quality);
    dev.enable_stream(rs::stream::color, rs::preset::best_quality);
    try { dev.enable_stream(rs::stream::infrared2, rs::preset::best_quality); } catch (...) {}
    dev.start();

    // Open a GLFW window
    glfwInit();
    std::ostringstream ss; ss << "RGB-D frame capture (" << dev.get_name() << ")";
    GLFWwindow * win = glfwCreateWindow(dev.is_stream_enabled(rs::stream::infrared2) ? 1920 : 1280, 960, ss.str().c_str(), 0, 0);
    glfwMakeContextCurrent(win);

    // Create a data folder to save frames
    std::string hash_id = gen_rand_str(16);
    std::string saveto_directory = "data/raw/" + hash_id + "/";
    sys_command("mkdir -p " + saveto_directory);

    // Determine depth value corresponding to one meter
    const uint16_t one_meter = static_cast<uint16_t>(1.0f / dev.get_depth_scale());

    // Save intrinsics of depth camera
    rs::intrinsics depth_intrinsics = dev.get_stream_intrinsics(rs::stream::depth);
    std::string depth_intrinsics_filename = saveto_directory + "camera.intrinsics.depth.txt";
    FILE *fp = fopen(depth_intrinsics_filename.c_str(), "w");
    fprintf(fp, "%.17g %.17g %.17g\n", depth_intrinsics.fx, 0.0f, depth_intrinsics.ppx);
    fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, depth_intrinsics.fy, depth_intrinsics.ppy);
    fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, 0.0f, 1.0f);
    fclose(fp); 

    // Save intrinsics of color camera
    rs::intrinsics color_intrinsics = dev.get_stream_intrinsics(rs::stream::color);
    std::string color_intrinsics_filename = saveto_directory + "camera.intrinsics.color.txt";
    fp = fopen(color_intrinsics_filename.c_str(), "w");
    fprintf(fp, "%.17g %.17g %.17g\n", color_intrinsics.fx, 0.0f, color_intrinsics.ppx);
    fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, color_intrinsics.fy, color_intrinsics.ppy);
    fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, 0.0f, 1.0f);
    fclose(fp); 

    // Save extrinsics from depth camera to color camera
    rs::extrinsics depth2color_extrinsics = dev.get_extrinsics(rs::stream::depth,rs::stream::color);
    std::string depth2color_extrinsics_filename = saveto_directory + "camera.extrinsics.depth2color.txt";
    fp = fopen(depth2color_extrinsics_filename.c_str(), "w");
    fprintf(fp, "%.17g %.17g %.17g %.17g\n", depth2color_extrinsics.rotation[0], depth2color_extrinsics.rotation[3], depth2color_extrinsics.rotation[6], depth2color_extrinsics.translation[0]);
    fprintf(fp, "%.17g %.17g %.17g %.17g\n", depth2color_extrinsics.rotation[1], depth2color_extrinsics.rotation[4], depth2color_extrinsics.rotation[7], depth2color_extrinsics.translation[1]);
    fprintf(fp, "%.17g %.17g %.17g %.17g\n", depth2color_extrinsics.rotation[2], depth2color_extrinsics.rotation[5], depth2color_extrinsics.rotation[8], depth2color_extrinsics.translation[2]);
    fclose(fp); 

    // Frame increment
    int frame_idx = 0;
    int key_press_state = 0;
    while (!glfwWindowShouldClose(win)) {

        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS && key_press_state == 0) {
            key_press_state++;
        }

        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_RELEASE && key_press_state == 1) {
          key_press_state = 0;
          break;
        }


        // Wait for new images
        glfwPollEvents();
        dev.wait_for_frames();

        // Debug frame timestamps
        // std::cout << dev.get_frame_timestamp(rs::stream::depth_aligned_to_color) << " " << dev.get_frame_timestamp(rs::stream::color) << std::endl;

        // Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
        const uint16_t * depth_frame = reinterpret_cast<const uint16_t *>(dev.get_frame_data(rs::stream::depth_aligned_to_color));
        const uint16_t * depth_frame_raw = reinterpret_cast<const uint16_t *>(dev.get_frame_data(rs::stream::depth));

        // Retrieve color data, which was previously configured as a 640 x 480 x 3 image of 8-bit color values
        const uint8_t * color_frame = reinterpret_cast<const uint8_t *>(dev.get_frame_data(rs::stream::color));

        // Get frame filename
        std::ostringstream frame_name_ss;
        frame_name_ss << std::setw(6) << std::setfill('0') << frame_idx;
        frame_idx++;

        // Save depth frame to disk (depth in millimeters, 16-bit, PNG)
        int depth_frame_width = dev.get_stream_width(rs::stream::depth_aligned_to_color);
        int depth_frame_height = dev.get_stream_height(rs::stream::depth_aligned_to_color);
        png::image<png::gray_pixel_16> depth_image(depth_frame_width, depth_frame_height);
        for (size_t y = 0; y < depth_frame_height; y++) {
            for (size_t x = 0; x < depth_frame_width; x++) {
                float depth_meters = ((float)depth_frame[y * depth_frame_width + x]) / ((float)one_meter);
                // std::cout << depth_meters << std::endl;
                depth_image[y][x] = png::gray_pixel_16((uint16_t)round(depth_meters * 1000));
            }
        }
        std::string depth_image_name = saveto_directory + "frame-" + frame_name_ss.str() + ".depth.png";
        depth_image.write(depth_image_name);

        int depth_frame_raw_width = dev.get_stream_width(rs::stream::depth);
        int depth_frame_raw_height = dev.get_stream_height(rs::stream::depth);
        png::image<png::gray_pixel_16> depth_image_raw(depth_frame_raw_width, depth_frame_raw_height);
        for (size_t y = 0; y < depth_frame_raw_height; y++) {
            for (size_t x = 0; x < depth_frame_raw_width; x++) {
                float depth_meters = ((float)depth_frame_raw[y * depth_frame_raw_width + x]) / ((float)one_meter);
                // std::cout << depth_meters << std::endl;
                depth_image_raw[y][x] = png::gray_pixel_16((uint16_t)round(depth_meters * 1000));
            }
        }
        std::string depth_image_raw_name = saveto_directory + "frame-" + frame_name_ss.str() + ".depth.raw.png";
        depth_image_raw.write(depth_image_raw_name);

        // Save color frame to disk (RGB, 24-bit, PNG)
        int color_frame_width = dev.get_stream_width(rs::stream::color);
        int color_frame_height = dev.get_stream_height(rs::stream::color);
        png::image<png::rgb_pixel> color_image(color_frame_width, color_frame_height);
        for (size_t y = 0; y < color_frame_height; y++) {
            for (size_t x = 0; x < color_frame_width; x++) {
                uint8_t r = color_frame[y * color_frame_width * 3 + x * 3 + 0];
                uint8_t g = color_frame[y * color_frame_width * 3 + x * 3 + 1];
                uint8_t b = color_frame[y * color_frame_width * 3 + x * 3 + 2];
                color_image[y][x] = png::rgb_pixel(r, g, b);
            }
        }
        std::string color_image_name = saveto_directory + "frame-" + frame_name_ss.str() + ".color.png";
        color_image.write(color_image_name);

        // Clear the framebuffer
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the images
        glPushMatrix();
        glfwGetWindowSize(win, &w, &h);
        glOrtho(0, w, h, 0, -1, +1);
        int s = w / (dev.is_stream_enabled(rs::stream::infrared2) ? 3 : 2);
        buffers[0].show(dev, rs::stream::color, 0, 0, s, h - h / 2);
        buffers[1].show(dev, rs::stream::color_aligned_to_depth, s, 0, s, h - h / 2);
        buffers[2].show(dev, rs::stream::depth_aligned_to_color, 0, h / 2, s, h - h / 2);
        buffers[3].show(dev, rs::stream::depth, s, h / 2, s, h - h / 2);
        if (dev.is_stream_enabled(rs::stream::infrared2)) {
            buffers[4].show(dev, rs::stream::infrared2_aligned_to_depth, 2 * s, 0, s, h - h / 2);
            buffers[5].show(dev, rs::stream::depth_aligned_to_infrared2, 2 * s, h / 2, s, h - h / 2);
        }
        glPopMatrix();
        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}
catch (const rs::error & e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}