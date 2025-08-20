cmake_minimum_required(VERSION 3.18)
project(cuda-pybind LANGUAGES CXX CUDA)

# Set policy to use modern Python finding
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# NOTE: This was needed to find pybind installation on vast.ai system!!!
# Include Conan-generated CMake dependency files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

find_package(pybind11 REQUIRED)
find_package(Catch2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cuda-pybind src/main.cpp)

# Link with proper pybind11 target (not the internal _all_do_not_use target)
target_link_libraries(cuda-pybind Catch2::Catch2WithMain pybind11::headers)

# pybind11_add_module automatically handles proper linking - no need to manually link pybind11
pybind11_add_module(vector_ops src/vector_ops.cpp src/vector_add.cu)

add_subdirectory(test)