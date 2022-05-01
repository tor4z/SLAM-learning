cmake_minimum_required(VERSION 3.10)

project(eigen_geometry)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(
    eigen_geometry
    eigen_geometry.cpp
)

target_link_libraries(
    eigen_geometry
    Eigen3::Eigen
)
