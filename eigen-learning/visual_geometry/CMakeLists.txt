cmake_minimum_required(VERSION 3.10)

project(visual_gemetry)

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(
    visual_geometry
    visual_geometry.cpp
)

target_link_libraries(
    visual_geometry
    Eigen3::Eigen ${Pangolin_LIBRARIES}
)
