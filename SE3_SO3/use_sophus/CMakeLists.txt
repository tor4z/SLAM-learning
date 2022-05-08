cmake_minimum_required(VERSION 3.10)

project(use_sophus)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

add_executable(
    use_sophus
    use_sophus.cpp
)

target_link_libraries(
    use_sophus
    ${Sophus_LIBRARIES} Eigen3::Eigen fmt::fmt
)
