cmake_minimum_required(VERSION 3.22)
project(GEMM LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD  14)
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else ()
    message(FATAL_ERROR "Cannot find CUDA")
endif ()



include_directories(include)
cuda_include_directories(include)

aux_source_directory(./src SRC_FILES)

add_executable(gemm main.cu ${SRC_FILES})
target_include_directories(gemm PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gemm PRIVATE ${CUDA_LIBRARIES})