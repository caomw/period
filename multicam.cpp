#include "util/util.hpp"
#include <librealsense/rs.hpp>
#include "display.hpp"
#include <png++/png.hpp>

std::vector<texture_buffer> buffers;

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
    
    // Enumerate all devices
    std::vector<rs::device *> devices;
    for (int i=0; i<ctx.get_device_count(); ++i) {
        devices.push_back(ctx.get_device(i));
    }
    // rs::device & dev = *ctx.get_device(0);

    // Configure and start our devices
    for (auto dev : devices) {
        std::cout << "Starting " << dev->get_name() << "... ";
        dev->enable_stream(rs::stream::depth, rs::preset::best_quality);
        dev->enable_stream(rs::stream::color, rs::preset::best_quality);
        dev->start();
        std::cout << "done." << std::endl;
    }

    // Depth and color
    buffers.resize(ctx.get_device_count() * 2);

    // Open a GLFW window
    glfwInit();
    std::ostringstream ss; ss << "Multi-camera RGB-D Capture";
    GLFWwindow * win = glfwCreateWindow(1280, 960, ss.str().c_str(), 0, 0);
    glfwMakeContextCurrent(win);

    int windowWidth, windowHeight;
    glfwGetWindowSize(win, &windowWidth, &windowHeight);

    // Does not account for correct aspect ratios
    auto perTextureWidth = windowWidth / devices.size();
    auto perTextureHeight = 480;

    // Determine depth value corresponding to one meter
    const uint16_t one_meter = static_cast<uint16_t>(1.0f / devices[0]->get_depth_scale());

    // Create a data folder to save frames
    std::string saveto_directory = "data/seq/";
    sys_command("mkdir -p " + saveto_directory);

    // Save intrinsics of each depth camera
    for (int i = 0; i < devices.size(); i++) {
        std::cout << "Saving depth intrinsics... ";
        rs::intrinsics depth_K = devices[i]->get_stream_intrinsics(rs::stream::depth_aligned_to_color);
        std::string intrinsics_filename = saveto_directory + "intrinsics." + std::to_string(i) + ".K.txt";
        FILE *fp = fopen(intrinsics_filename.c_str(), "w");
        fprintf(fp, "%.17g %.17g %.17g\n", depth_K.fx, 0.0f, depth_K.ppx);
        fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, depth_K.fy, depth_K.ppy);
        fprintf(fp, "%.17g %.17g %.17g\n", 0.0f, 0.0f, 1.0f);
        fclose(fp); 
        std::cout << "done." << std::endl;
    }

    //Instructions
    std::cout << "Press the spacebar key to save RGB-D frames to disk." << std::endl;

    int frame_idx = 0;
    int key_press_state = 0;
    while (!glfwWindowShouldClose(win)) {

        // Wait for new images
        glfwPollEvents();
        // dev.wait_for_frames();

        // Clear the framebuffer
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw the images
        glfwGetWindowSize(win, &w, &h);
        glPushMatrix();
        glOrtho(0, w, h, 0, -1, +1);
        glPixelZoom(1, -1);
        int i=0, x=0;
        for(auto dev : devices) {
            dev->poll_for_frames();
            // const auto c = dev->get_stream_intrinsics(rs::stream::color), d = dev->get_stream_intrinsics(rs::stream::depth);
            buffers[i++].show(*dev, rs::stream::color, x, 0, perTextureWidth, perTextureHeight);
            buffers[i++].show(*dev, rs::stream::depth_aligned_to_color, x, perTextureHeight, perTextureWidth, perTextureHeight);
            x += perTextureWidth;
        }

        // int state = glfwGetKey(win, GLFW_KEY_SPACE);
        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS && key_press_state == 0) {
            key_press_state++;
        }

        if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_RELEASE && key_press_state == 1) {
            std::cout << "Saving RGB-D frames... ";

            // Stop streaming on all devices
            for(auto dev : devices) {
                if(dev->is_streaming()) dev->stop();
            }

            for(int i = 0; i < devices.size(); i++) {
                devices[i]->start();

                // // Capture multiple depth frames and median them
                // int num_frames_per_capture = 30;
                // std::vector<const uint16_t *> depth_inter_frame;
                // std::vector<const uint8_t *> color_inter_frame;
                // for (int j = 0; j < num_frames_per_capture; j++) {
                //     devices[i]->wait_for_frames();
                
                //     // Debug frame timestamps
                //     // std::cout << devices[i]->get_frame_timestamp(rs::stream::depth_aligned_to_color) << " " << devices[i]->get_frame_timestamp(rs::stream::color) << std::endl;

                //     // Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
                //     const uint16_t * depth_frame = reinterpret_cast<const uint16_t *>(devices[i]->get_frame_data(rs::stream::depth_aligned_to_color));
                //     depth_inter_frame.push_back(depth_frame);

                //     // Retrieve color data, which was previously configured as a 640 x 480 x 3 image of 8-bit color values
                //     const uint8_t * color_frame = reinterpret_cast<const uint8_t *>(devices[i]->get_frame_data(rs::stream::color));
                //     color_inter_frame.push_back(color_frame);
                // }
                // uint16_t * depth_frame = new uint16_t[480*640];
                // for (int j = 0; j < 480*640; j++) {
                //     double depth_value = (double)depth_inter_frame[0][j] + (double)depth_inter_frame[1][j] + (double)depth_inter_frame[2][j];
                //     depth_frame[j] = (uint16_t)(round(depth_value/3.0));
                // }

                // const uint8_t * color_frame = color_inter_frame[0];

                devices[i]->wait_for_frames();

                // Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
                const uint16_t * depth_frame = reinterpret_cast<const uint16_t *>(devices[i]->get_frame_data(rs::stream::depth_aligned_to_color));

                // Retrieve color data, which was previously configured as a 640 x 480 x 3 image of 8-bit color values
                const uint8_t * color_frame = reinterpret_cast<const uint8_t *>(devices[i]->get_frame_data(rs::stream::color));

                // Get frame filename
                std::ostringstream frame_name_ss;
                frame_name_ss << std::setw(6) << std::setfill('0') << frame_idx;

                // Save depth frame to disk (depth in millimeters, 16-bit, PNG)
                int depth_frame_width = devices[i]->get_stream_width(rs::stream::depth_aligned_to_color);
                int depth_frame_height = devices[i]->get_stream_height(rs::stream::depth_aligned_to_color);
                png::image<png::gray_pixel_16> depth_image(depth_frame_width, depth_frame_height);
                for (size_t y = 0; y < depth_frame_height; y++) {
                    for (size_t x = 0; x < depth_frame_width; x++) {
                        float depth_meters = ((float)depth_frame[y * depth_frame_width + x]) / ((float)one_meter);
                        // std::cout << depth_meters << std::endl;
                        depth_image[y][x] = png::gray_pixel_16((uint16_t)round(depth_meters * 1000));
                    }
                }
                std::string depth_image_name = saveto_directory + "frame-" + frame_name_ss.str() + "." + std::to_string(i) + ".depth.png";
                depth_image.write(depth_image_name);

                // Save color frame to disk (RGB, 24-bit, PNG)
                int color_frame_width = devices[i]->get_stream_width(rs::stream::color);
                int color_frame_height = devices[i]->get_stream_height(rs::stream::color);
                png::image<png::rgb_pixel> color_image(color_frame_width, color_frame_height);
                for (size_t y = 0; y < color_frame_height; y++) {
                    for (size_t x = 0; x < color_frame_width; x++) {
                        uint8_t r = color_frame[y * color_frame_width * 3 + x * 3 + 0];
                        uint8_t g = color_frame[y * color_frame_width * 3 + x * 3 + 1];
                        uint8_t b = color_frame[y * color_frame_width * 3 + x * 3 + 2];
                        color_image[y][x] = png::rgb_pixel(r, g, b);
                    }
                }
                std::string color_image_name = saveto_directory + "frame-" + frame_name_ss.str() + "." + std::to_string(i) + ".color.png";
                color_image.write(color_image_name);

                devices[i]->stop();
            }

            // Resume streaming on all devices
            for(auto dev : devices) {
                if(!dev->is_streaming()) dev->start();
            }
            frame_idx++;
            key_press_state = 0;
            std::cout << "done." << std::endl;
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