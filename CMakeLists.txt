cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA_ARCHITECTURES before the project command
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(CudaLearning LANGUAGES CXX CUDA)

add_subdirectory(src)

enable_testing()

add_subdirectory(tests)