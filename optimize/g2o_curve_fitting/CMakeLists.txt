cmake_minimum_required(VERSION 3.10)

project(g2o_curve_fitting)
add_compile_options(-std=c++14)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)


add_executable(
    g2o_curve_fitting
    g2o_curve_fitting.cpp
)

target_link_libraries(
    g2o_curve_fitting
    g2o_core g2o_solver_dense g2o_stuff
    Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES}
)
