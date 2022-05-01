cmake_minimum_required(VERSION 3.10)

project(coordinate_transform)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(
    coordinate_transform
    coordinate_transform.cpp
)

target_link_libraries(
    coordinate_transform
    Eigen3::Eigen
)
