cmake_minimum_required(VERSION 3.10)

project(create_map)


find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)


add_executable(
    dense_mapping
    dense_mapping.cpp
)

target_link_libraries(
    dense_mapping
    ${OpenCV_LIBS} Eigen3::Eigen ${Sophus_LIBRARIES} fmt::fmt
)
