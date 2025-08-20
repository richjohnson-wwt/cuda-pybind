

# Test executable for CUDA vector operations
add_executable(test_vector_add test_vector_add.cpp ../src/vector_add.cu)
target_include_directories(test_vector_add PRIVATE ../include)
target_link_libraries(test_vector_add PRIVATE Catch2::Catch2WithMain)

add_executable(test_vector_redux test_vector_redux.cpp ../src/vector_redux.cu)
target_include_directories(test_vector_redux PRIVATE ../include)
target_link_libraries(test_vector_redux PRIVATE Catch2::Catch2WithMain)

# Enable C++17 (or higher) 
target_compile_features(test_vector_add PRIVATE cxx_std_17)
target_compile_features(test_vector_redux PRIVATE cxx_std_17)

# Register test with CTest
add_test(NAME CUDA_VectorAdd_Tests COMMAND test_vector_add)

# Try to use Catch2's auto-discovery if available
if(TARGET Catch2::Catch2)
  include(Catch)
  if(COMMAND catch_discover_tests)
    catch_discover_tests(test_vector_add)
    catch_discover_tests(test_vector_redux)
  endif()
endif()
