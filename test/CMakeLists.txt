


find_package(Catch2 REQUIRED)

add_executable(test_vector_add test_vector_add.cpp ../src/vector_add.cu)
target_include_directories(test_vector_add PRIVATE ../include)
target_link_libraries(test_vector_add PRIVATE Catch2::Catch2WithMain)

# Enable C++17 (or higher)
target_compile_features(test_vector_add PRIVATE cxx_std_17)

# Optional: Add test to `ctest`
include(CTest)
include(Catch)
catch_discover_tests(test_vector_add)
enable_testing()
add_test(NAME CUDA_VectorAdd_Tests COMMAND test_vector_add)
