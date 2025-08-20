cmake_minimum_required(VERSION 3.18)
project(cuda-pybind LANGUAGES CXX CUDA)

# Set policy to use modern Python finding
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

find_package(pybind11 REQUIRED)
find_package(Catch2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cuda-pybind src/main.cpp)

target_link_libraries(cuda-pybind Catch2::Catch2WithMain pybind11_all_do_not_use)

pybind11_add_module(vector_ops src/vector_ops.cpp src/vector_add.cu)