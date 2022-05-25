cmake_minimum_required(VERSION 3.10)

project(glmatrix)


find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)


add_executable(
    glmatrix
    main.cpp
)

target_link_libraries(
    glmatrix
    Eigen3::Eigen
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
)
