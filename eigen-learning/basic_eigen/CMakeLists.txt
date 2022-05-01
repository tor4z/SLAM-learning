cmake_minimum_required(VERSION 3.10)

project(basic_eigen)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(
    eigen_matrix
    eigen_matrix.cpp
)

target_link_libraries(
    eigen_matrix
    Eigen3::Eigen
)
