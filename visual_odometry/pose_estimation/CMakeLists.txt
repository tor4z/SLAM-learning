cmake_minimum_required(VERSION 3.10)

project(pose_estimation)


find_package(OpenCV REQUIRED)


add_executable(
    pose_estimation
    pose_estimation.cpp
)

target_link_libraries(
    pose_estimation
    ${OpenCV_LIBS}
)
