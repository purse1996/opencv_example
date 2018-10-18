# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/shuaibin/opencv_workspace/examples/backProjection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shuaibin/opencv_workspace/examples/backProjection

# Include any dependencies generated for this target.
include CMakeFiles/backProjection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/backProjection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/backProjection.dir/flags.make

CMakeFiles/backProjection.dir/backProjection.cpp.o: CMakeFiles/backProjection.dir/flags.make
CMakeFiles/backProjection.dir/backProjection.cpp.o: backProjection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shuaibin/opencv_workspace/examples/backProjection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/backProjection.dir/backProjection.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/backProjection.dir/backProjection.cpp.o -c /home/shuaibin/opencv_workspace/examples/backProjection/backProjection.cpp

CMakeFiles/backProjection.dir/backProjection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/backProjection.dir/backProjection.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shuaibin/opencv_workspace/examples/backProjection/backProjection.cpp > CMakeFiles/backProjection.dir/backProjection.cpp.i

CMakeFiles/backProjection.dir/backProjection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/backProjection.dir/backProjection.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shuaibin/opencv_workspace/examples/backProjection/backProjection.cpp -o CMakeFiles/backProjection.dir/backProjection.cpp.s

CMakeFiles/backProjection.dir/backProjection.cpp.o.requires:

.PHONY : CMakeFiles/backProjection.dir/backProjection.cpp.o.requires

CMakeFiles/backProjection.dir/backProjection.cpp.o.provides: CMakeFiles/backProjection.dir/backProjection.cpp.o.requires
	$(MAKE) -f CMakeFiles/backProjection.dir/build.make CMakeFiles/backProjection.dir/backProjection.cpp.o.provides.build
.PHONY : CMakeFiles/backProjection.dir/backProjection.cpp.o.provides

CMakeFiles/backProjection.dir/backProjection.cpp.o.provides.build: CMakeFiles/backProjection.dir/backProjection.cpp.o


# Object files for target backProjection
backProjection_OBJECTS = \
"CMakeFiles/backProjection.dir/backProjection.cpp.o"

# External object files for target backProjection
backProjection_EXTERNAL_OBJECTS =

backProjection: CMakeFiles/backProjection.dir/backProjection.cpp.o
backProjection: CMakeFiles/backProjection.dir/build.make
backProjection: /usr/local/lib/libopencv_stitching.so.3.4.1
backProjection: /usr/local/lib/libopencv_superres.so.3.4.1
backProjection: /usr/local/lib/libopencv_videostab.so.3.4.1
backProjection: /usr/local/lib/libopencv_ml.so.3.4.1
backProjection: /usr/local/lib/libopencv_calib3d.so.3.4.1
backProjection: /usr/local/lib/libopencv_objdetect.so.3.4.1
backProjection: /usr/local/lib/libopencv_shape.so.3.4.1
backProjection: /usr/local/lib/libopencv_video.so.3.4.1
backProjection: /usr/local/lib/libopencv_dnn.so.3.4.1
backProjection: /usr/local/lib/libopencv_photo.so.3.4.1
backProjection: /usr/local/lib/libopencv_features2d.so.3.4.1
backProjection: /usr/local/lib/libopencv_flann.so.3.4.1
backProjection: /usr/local/lib/libopencv_highgui.so.3.4.1
backProjection: /usr/local/lib/libopencv_videoio.so.3.4.1
backProjection: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
backProjection: /usr/local/lib/libopencv_imgproc.so.3.4.1
backProjection: /usr/local/lib/libopencv_core.so.3.4.1
backProjection: CMakeFiles/backProjection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shuaibin/opencv_workspace/examples/backProjection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable backProjection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/backProjection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/backProjection.dir/build: backProjection

.PHONY : CMakeFiles/backProjection.dir/build

CMakeFiles/backProjection.dir/requires: CMakeFiles/backProjection.dir/backProjection.cpp.o.requires

.PHONY : CMakeFiles/backProjection.dir/requires

CMakeFiles/backProjection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/backProjection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/backProjection.dir/clean

CMakeFiles/backProjection.dir/depend:
	cd /home/shuaibin/opencv_workspace/examples/backProjection && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shuaibin/opencv_workspace/examples/backProjection /home/shuaibin/opencv_workspace/examples/backProjection /home/shuaibin/opencv_workspace/examples/backProjection /home/shuaibin/opencv_workspace/examples/backProjection /home/shuaibin/opencv_workspace/examples/backProjection/CMakeFiles/backProjection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/backProjection.dir/depend

