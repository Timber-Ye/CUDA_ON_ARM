cmake_minimum_required(VERSION 3.24)
project(CUDA_ON_ARM CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

#find_package(CUDA REQUIRED)

add_subdirectory(hello_cuda)

add_subdirectory(vector_add)

add_subdirectory(matmul)

add_subdirectory(homework)

add_subdirectory(stream_and_libraries)

add_subdirectory(atomic)

add_subdirectory(exam)

add_subdirectory(expand)
