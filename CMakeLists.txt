cmake_minimum_required(VERSION 3.22)
project(cuda LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
set(Torch_DIR /home/guopeiwen/libtorch/share/cmake/Torch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD  17)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

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


add_executable(cuda_test main.cu ${SRC_FILES})
target_include_directories(cuda_test PRIVATE ${CUDA_INCLUDE_DIRS} include)
target_link_libraries(cuda_test PRIVATE ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})