cmake_minimum_required(VERSION 2.8)
project( mySobel )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(mySobel my_sobel.cpp )
target_link_libraries( mySobel ${OpenCV_LIBS} )
