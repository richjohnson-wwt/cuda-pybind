#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "../include/vector_add.h"

using Catch::Approx;

TEST_CASE("vector_redux_cuda reduces size 12 array increments of 4", "[cuda]") {
    int segments = 3;  // 12 / 4
    int blockSize = 4;
    // array size 12 with redux (4): 
    //    1, 2, 3, 4 - 5, 6, 7, 8, - 9, 10, 11, 12
    //    10 - 26 - 42 
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> result(segments);

    vector_redux_cuda(a.data(), result.data(), blockSize, segments);

    std::cout << "RESULTS: " << result[0] << "," << result[1] << "," << result[2] << std::endl;

    REQUIRE(result[0] == Approx(10.0f));
    REQUIRE(result[1] == Approx(26.0f));
    REQUIRE(result[2] == Approx(42.0f));
}