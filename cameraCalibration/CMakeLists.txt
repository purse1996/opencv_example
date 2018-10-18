cmake_minimum_required(VERSION 2.8)
project( MyCameraCalibration )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( MyCameraCalibration MyCameraCalibration.cpp )
target_link_libraries( MyCameraCalibration ${OpenCV_LIBS} )



