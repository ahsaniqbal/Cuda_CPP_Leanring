cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA_ARCHITECTURES before the project command
set(CMAKE_CUDA_ARCHITECTURES OFF)

project(CudaLearning LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp)
list(FILTER CURRENT_HEADERS EXCLUDE REGEX "CMakeFiles")

file(GLOB_RECURSE CURRENT_SOURCES  *.cpp)
list(FILTER CURRENT_SOURCES EXCLUDE REGEX "CMakeFiles")

file(GLOB_RECURSE CURRENT_CUDA_SOURCES  *.cu)
list(FILTER CURRENT_CUDA_SOURCES EXCLUDE REGEX "CMakeFiles")

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES})
source_group("CUDA" FILES ${CURRENT_CUDA_SOURCES})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

add_executable(CudaLearning ${CURRENT_HEADERS} ${CURRENT_SOURCES} ${CURRENT_CUDA_SOURCES})

# Set CUDA_ARCHITECTURES property after the target is created
set_property(TARGET CudaLearning PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(CudaLearning ${OpenCV_LIBS} ${CUDA_LIBRARIES})