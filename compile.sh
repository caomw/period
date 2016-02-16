# g++ kinfu.cpp -std=c++11 -Iinclude -Llib -lm -lpng `libpng-config --cflags` -o kinfu
g++ multicam.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` -o multicam
g++ capture.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` -o capture

