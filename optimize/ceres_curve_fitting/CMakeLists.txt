cmake_minimum_required(VERSION 3.10)

project(ceres_curve_fitting)


find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(
    ceres_curve_fitting
    ceres_curve_fitting.cpp
)

target_link_libraries(
    ceres_curve_fitting
    Ceres::ceres ${OpenCV_LIBS}
)
