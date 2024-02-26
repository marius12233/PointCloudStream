# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cem2brg/CUDA-PointPillars

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cem2brg/CUDA-PointPillars/build

# Include any dependencies generated for this target.
include CMakeFiles/pointpillar_core.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pointpillar_core.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pointpillar_core.dir/flags.make

CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o: CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o.depend
CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o: CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o.Release.cmake
CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o: ../src/common/tensor.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o"
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common && /usr/bin/cmake -E make_directory /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common/.
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common/./pointpillar_core_generated_tensor.cu.o -D generated_cubin_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common/./pointpillar_core_generated_tensor.cu.o.cubin.txt -P /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o.Release.cmake

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o.depend
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o.Release.cmake
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o: ../src/pointpillar/lidar-backbone.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o"
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -E make_directory /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/.
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-backbone.cu.o -D generated_cubin_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-backbone.cu.o.cubin.txt -P /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o.Release.cmake

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o.depend
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o.Release.cmake
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o: ../src/pointpillar/lidar-postprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building NVCC (Device) object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o"
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -E make_directory /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/.
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-postprocess.cu.o -D generated_cubin_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-postprocess.cu.o.cubin.txt -P /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o.Release.cmake

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o.depend
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o.Release.cmake
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o: ../src/pointpillar/lidar-voxelization.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building NVCC (Device) object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o"
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -E make_directory /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/.
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-voxelization.cu.o -D generated_cubin_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_lidar-voxelization.cu.o.cubin.txt -P /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o.Release.cmake

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o.depend
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o.Release.cmake
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o: ../src/pointpillar/pillarscatter-kernel.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building NVCC (Device) object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o"
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -E make_directory /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/.
	cd /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_pillarscatter-kernel.cu.o -D generated_cubin_file:STRING=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/./pointpillar_core_generated_pillarscatter-kernel.cu.o.cubin.txt -P /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o.Release.cmake

CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o: CMakeFiles/pointpillar_core.dir/flags.make
CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o: ../src/common/tensorrt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o"
	/usr/lib/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o -c /home/cem2brg/CUDA-PointPillars/src/common/tensorrt.cpp

CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cem2brg/CUDA-PointPillars/src/common/tensorrt.cpp > CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.i

CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cem2brg/CUDA-PointPillars/src/common/tensorrt.cpp -o CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.s

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o: CMakeFiles/pointpillar_core.dir/flags.make
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o: ../src/pointpillar/pointpillar-scatter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o"
	/usr/lib/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o -c /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar-scatter.cpp

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar-scatter.cpp > CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.i

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar-scatter.cpp -o CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.s

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o: CMakeFiles/pointpillar_core.dir/flags.make
CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o: ../src/pointpillar/pointpillar.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o"
	/usr/lib/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o -c /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar.cpp

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar.cpp > CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.i

CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cem2brg/CUDA-PointPillars/src/pointpillar/pointpillar.cpp -o CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.s

# Object files for target pointpillar_core
pointpillar_core_OBJECTS = \
"CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o" \
"CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o" \
"CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o"

# External object files for target pointpillar_core
pointpillar_core_EXTERNAL_OBJECTS = \
"/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o" \
"/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o" \
"/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o" \
"/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o" \
"/home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o"

libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/common/tensorrt.cpp.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar-scatter.cpp.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar.cpp.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/build.make
libpointpillar_core.so: /usr/local/cuda-12.0/lib64/libcudart_static.a
libpointpillar_core.so: /usr/lib/x86_64-linux-gnu/librt.so
libpointpillar_core.so: CMakeFiles/pointpillar_core.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cem2brg/CUDA-PointPillars/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library libpointpillar_core.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pointpillar_core.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pointpillar_core.dir/build: libpointpillar_core.so

.PHONY : CMakeFiles/pointpillar_core.dir/build

CMakeFiles/pointpillar_core.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pointpillar_core.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pointpillar_core.dir/clean

CMakeFiles/pointpillar_core.dir/depend: CMakeFiles/pointpillar_core.dir/src/common/pointpillar_core_generated_tensor.cu.o
CMakeFiles/pointpillar_core.dir/depend: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-backbone.cu.o
CMakeFiles/pointpillar_core.dir/depend: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-postprocess.cu.o
CMakeFiles/pointpillar_core.dir/depend: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_lidar-voxelization.cu.o
CMakeFiles/pointpillar_core.dir/depend: CMakeFiles/pointpillar_core.dir/src/pointpillar/pointpillar_core_generated_pillarscatter-kernel.cu.o
	cd /home/cem2brg/CUDA-PointPillars/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cem2brg/CUDA-PointPillars /home/cem2brg/CUDA-PointPillars /home/cem2brg/CUDA-PointPillars/build /home/cem2brg/CUDA-PointPillars/build /home/cem2brg/CUDA-PointPillars/build/CMakeFiles/pointpillar_core.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pointpillar_core.dir/depend

