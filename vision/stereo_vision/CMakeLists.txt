cmake_minimum_required(VERSION 3.10)

project(stereo_vision)


find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(
    stereo_vision
    stereo_vision.cpp
)

target_link_libraries(
    stereo_vision
    Eigen3::Eigen ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
)
