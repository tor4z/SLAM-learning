cmake_minimum_required(VERSION 3.10)

project(trajectory_error)


find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Pangolin REQUIRED)


add_executable(
    trajectory_error
    trajectory_error.cpp
)

target_link_libraries(
    trajectory_error
    ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES}
    Eigen3::Eigen fmt::fmt
)
