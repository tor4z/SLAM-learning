cmake_minimum_required(VERSION 3.10)

project(simulator_line_3d)
add_compile_options(-std=c++14)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)


add_executable(
    simulator_line_3d
    simulator_line_3d.cpp
)

target_link_libraries(
    simulator_line_3d
    g2o_core g2o_solver_eigen g2o_stuff
    g2o_types_slam3d g2o_types_slam3d_addons
    Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES}
)
