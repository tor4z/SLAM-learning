cmake_minimum_required(VERSION 3.10)

project(triangulation)

find_package(OpenCV REQUIRED)

add_executable(
    triangulation
    triangulation.cpp
)

target_link_libraries(
    triangulation
    ${OpenCV_LIBS}
)
