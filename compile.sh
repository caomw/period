g++ capture.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` -o capture
