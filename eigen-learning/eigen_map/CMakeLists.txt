cmake_minimum_required(VERSION 3.10)


project(eigen_map)

find_package(Eigen3 REQUIRED)

add_executable(
    eigen_map
    eigen_map.cpp
)

target_link_libraries(
    eigen_map
    Eigen3::Eigen
)
