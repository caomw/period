export PATH=$PATH:/usr/local/cuda/bin
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/usr/local/cudnn/v4rc/lib64

# nvcc kinfu.cu -std=c++11 -Iinclude -Llib -lm -lpng `libpng-config --cflags` -o kinfu
# g++ multicam.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` -o multicam
# g++ capture.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` -o capture
# nvcc prep.cu -std=c++11 -Iinclude -Llib -lm -lpng -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR -lcurand `libpng-config --cflags` `pkg-config --cflags opencv` `pkg-config --libs opencv` -o prep
nvcc detect.cu -std=c++11 -O0 -Iinclude -Itools/marvin -I/usr/local/cuda/include -I/usr/local/cudnn/v4rc/include -Llib -lm -lpng -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR -lcurand -lcudart -lcublas -lcudnn `libpng-config --cflags` `pkg-config --cflags opencv` `pkg-config --libs opencv` -o detect
# nvcc train.cu -std=c++11 -Iinclude -Llib -lm -lpng -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR -lcurand `libpng-config --cflags` `pkg-config --cflags opencv` `pkg-config --libs opencv` -o train
# g++ test.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` `pkg-config --cflags opencv` `pkg-config --libs opencv` -o test
# g++ standalone.cpp -std=c++11 -Iinclude -Llib -lrealsense -lm -lpng -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR `pkg-config --cflags --libs glfw3 gl glu` `libpng-config --cflags` `pkg-config --cflags opencv` `pkg-config --libs opencv` -o standalone
