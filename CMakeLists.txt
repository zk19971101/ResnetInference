cmake_minimum_required(VERSION 3.10)
project(TensorRTExample)

find_package(CUDA REQUIRED)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TensorRT 的安装路径，根据实际情况修改
set(TENSORRT_ROOT /home/liquid/software/TensorRT-10.6.0.26)  # 示例路径，请根据实际情况修改

# 指定cmake寻找opencv库路径
set(OpenCV_DIR /home/liquid/lib/opencv4.10.0_install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
# 引入头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 指定编译模式为debug
set(CMAKE_BUILD_TYPE Debug)


# 查找 TensorRT 库
find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)
find_library(CUDA_LIBRARY cudart HINTS /usr/local/cuda-12.2/lib64)  # 根据 CUDA 安装路径修改

# 查找 TensorRT 头文件路径
include_directories(${TENSORRT_ROOT}/include)
include_directories(/usr/local/cuda-12.2/include)  # 根据 CUDA 安装路径修改

# 添加可执行文件
cuda_add_executable(out main.cu)
# 链接 TensorRT 和 CUDA 库
target_link_libraries(out ${NVINFER_LIBRARY} ${NVINFER_PLUGIN_LIBRARY} ${CUDA_LIBRARY} ${OpenCV_LIBS})