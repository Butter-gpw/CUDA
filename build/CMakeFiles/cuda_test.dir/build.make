# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/guopeiwen/code/cppProjects/CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guopeiwen/code/cppProjects/CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_test.dir/flags.make

CMakeFiles/cuda_test.dir/main.cu.o: CMakeFiles/cuda_test.dir/flags.make
CMakeFiles/cuda_test.dir/main.cu.o: ../main.cu
CMakeFiles/cuda_test.dir/main.cu.o: CMakeFiles/cuda_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cuda_test.dir/main.cu.o"
	/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_test.dir/main.cu.o -MF CMakeFiles/cuda_test.dir/main.cu.o.d -x cu -c /home/guopeiwen/code/cppProjects/CUDA/main.cu -o CMakeFiles/cuda_test.dir/main.cu.o

CMakeFiles/cuda_test.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_test.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_test.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_test.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o: CMakeFiles/cuda_test.dir/flags.make
CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o: ../reduce/block_all_reduce.cu
CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o: CMakeFiles/cuda_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o"
	/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o -MF CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o.d -x cu -c /home/guopeiwen/code/cppProjects/CUDA/reduce/block_all_reduce.cu -o CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o

CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda_test.dir/src/elementwise.cu.o: CMakeFiles/cuda_test.dir/flags.make
CMakeFiles/cuda_test.dir/src/elementwise.cu.o: ../src/elementwise.cu
CMakeFiles/cuda_test.dir/src/elementwise.cu.o: CMakeFiles/cuda_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/cuda_test.dir/src/elementwise.cu.o"
	/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_test.dir/src/elementwise.cu.o -MF CMakeFiles/cuda_test.dir/src/elementwise.cu.o.d -x cu -c /home/guopeiwen/code/cppProjects/CUDA/src/elementwise.cu -o CMakeFiles/cuda_test.dir/src/elementwise.cu.o

CMakeFiles/cuda_test.dir/src/elementwise.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_test.dir/src/elementwise.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_test.dir/src/elementwise.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_test.dir/src/elementwise.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda_test.dir/src/gemm.cu.o: CMakeFiles/cuda_test.dir/flags.make
CMakeFiles/cuda_test.dir/src/gemm.cu.o: ../src/gemm.cu
CMakeFiles/cuda_test.dir/src/gemm.cu.o: CMakeFiles/cuda_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/cuda_test.dir/src/gemm.cu.o"
	/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_test.dir/src/gemm.cu.o -MF CMakeFiles/cuda_test.dir/src/gemm.cu.o.d -x cu -c /home/guopeiwen/code/cppProjects/CUDA/src/gemm.cu -o CMakeFiles/cuda_test.dir/src/gemm.cu.o

CMakeFiles/cuda_test.dir/src/gemm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_test.dir/src/gemm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_test.dir/src/gemm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_test.dir/src/gemm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda_test.dir/src/reduce.cu.o: CMakeFiles/cuda_test.dir/flags.make
CMakeFiles/cuda_test.dir/src/reduce.cu.o: ../src/reduce.cu
CMakeFiles/cuda_test.dir/src/reduce.cu.o: CMakeFiles/cuda_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cuda_test.dir/src/reduce.cu.o"
	/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda_test.dir/src/reduce.cu.o -MF CMakeFiles/cuda_test.dir/src/reduce.cu.o.d -x cu -c /home/guopeiwen/code/cppProjects/CUDA/src/reduce.cu -o CMakeFiles/cuda_test.dir/src/reduce.cu.o

CMakeFiles/cuda_test.dir/src/reduce.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_test.dir/src/reduce.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_test.dir/src/reduce.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_test.dir/src/reduce.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuda_test
cuda_test_OBJECTS = \
"CMakeFiles/cuda_test.dir/main.cu.o" \
"CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o" \
"CMakeFiles/cuda_test.dir/src/elementwise.cu.o" \
"CMakeFiles/cuda_test.dir/src/gemm.cu.o" \
"CMakeFiles/cuda_test.dir/src/reduce.cu.o"

# External object files for target cuda_test
cuda_test_EXTERNAL_OBJECTS =

cuda_test: CMakeFiles/cuda_test.dir/main.cu.o
cuda_test: CMakeFiles/cuda_test.dir/reduce/block_all_reduce.cu.o
cuda_test: CMakeFiles/cuda_test.dir/src/elementwise.cu.o
cuda_test: CMakeFiles/cuda_test.dir/src/gemm.cu.o
cuda_test: CMakeFiles/cuda_test.dir/src/reduce.cu.o
cuda_test: CMakeFiles/cuda_test.dir/build.make
cuda_test: /usr/local/cuda-12.4/lib64/libcudart.so
cuda_test: /home/guopeiwen/libtorch/lib/libtorch.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10.so
cuda_test: /home/guopeiwen/libtorch/lib/libkineto.a
cuda_test: /usr/lib/x86_64-linux-gnu/libcuda.so
cuda_test: /usr/local/cuda-12.4/lib64/libnvrtc.so
cuda_test: /usr/local/cuda-12.4/lib64/libnvToolsExt.so
cuda_test: /usr/local/cuda-12.4/lib64/libcudart.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10_cuda.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10.so
cuda_test: /home/guopeiwen/libtorch/lib/libkineto.a
cuda_test: /usr/lib/x86_64-linux-gnu/libcuda.so
cuda_test: /usr/local/cuda-12.4/lib64/libnvrtc.so
cuda_test: /usr/local/cuda-12.4/lib64/libnvToolsExt.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10_cuda.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10_cuda.so
cuda_test: /home/guopeiwen/libtorch/lib/libc10.so
cuda_test: /usr/local/cuda-12.4/lib64/libcudart.so
cuda_test: /usr/local/cuda-12.4/lib64/libnvToolsExt.so
cuda_test: CMakeFiles/cuda_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CUDA executable cuda_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_test.dir/build: cuda_test
.PHONY : CMakeFiles/cuda_test.dir/build

CMakeFiles/cuda_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_test.dir/clean

CMakeFiles/cuda_test.dir/depend:
	cd /home/guopeiwen/code/cppProjects/CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guopeiwen/code/cppProjects/CUDA /home/guopeiwen/code/cppProjects/CUDA /home/guopeiwen/code/cppProjects/CUDA/build /home/guopeiwen/code/cppProjects/CUDA/build /home/guopeiwen/code/cppProjects/CUDA/build/CMakeFiles/cuda_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_test.dir/depend

