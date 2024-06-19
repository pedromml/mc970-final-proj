cmake_minimum_required(VERSION 3.9)

project(smoothing
    LANGUAGES C CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(serial   src/smooth-serial.c)
add_executable(parallel src/smooth-parallel.cu)

if(UNIX)
    target_link_libraries(serial PRIVATE m)
    target_link_libraries(parallel PRIVATE m)
endif()

target_link_libraries(serial
    PRIVATE OpenMP::OpenMP_C)

target_link_libraries(parallel
    PRIVATE OpenMP::OpenMP_C)

target_compile_options(parallel
    PRIVATE "-arch=sm_61")

set_property(
    TARGET   parallel
    PROPERTY CUDA_ARCHITECTURES OFF)
