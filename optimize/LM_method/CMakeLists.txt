cmake_minimum_required(VERSION 3.10)

project(LM_method)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

add_executable(
    LM_method
    LM_method.cpp
)

target_link_libraries(
    LM_method
    ${OpenCV_LIBS} Eigen3::Eigen
)
