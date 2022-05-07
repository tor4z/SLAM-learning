cmake_minimum_required(VERSION 3.10)

project(ldlt_solve)

find_package(Eigen3 3.3 REQUIRED)

add_executable(
    ldlt_solve
    ldlt_solve.cpp
)

target_link_libraries(
    ldlt_solve
    Eigen3::Eigen
)

