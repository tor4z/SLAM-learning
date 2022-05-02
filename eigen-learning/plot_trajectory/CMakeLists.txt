cmake_minimum_required(VERSION 3.10)

project(plot_trajectory)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)


add_executable(
    plot_trajectory
    plot_trajectory.cpp
)


target_link_libraries(
    plot_trajectory
    Eigen3::Eigen ${Pangolin_LIBRARIES}
)
