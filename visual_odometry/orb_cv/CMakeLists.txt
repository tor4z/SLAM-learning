cmake_minimum_required(VERSION 3.10)

project(orb_cv)


find_package(OpenCV REQUIRED)


add_executable(
    orb_cv
    orb_cv.cpp
)

target_link_libraries(
    orb_cv
    ${OpenCV_LIBS}
)
