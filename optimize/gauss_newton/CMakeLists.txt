cmake_minimum_required(VERSION 3.10)

project(gauss_newtown)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(
    gauss_newton
    gauss_newton.cpp
)

target_link_libraries(
    gauss_newton
    ${OpenCV_LIBS} Eigen3::Eigen
)
