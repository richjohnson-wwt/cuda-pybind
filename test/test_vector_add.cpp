#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include "../include/vector_add.h"

using Catch::Approx;

TEST_CASE("vector_add_cuda adds small arrays correctly", "[cuda]") {
    int n = 3;
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    std::vector<float> result(n);

    vector_add_cuda(a.data(), b.data(), result.data(), n);

    REQUIRE(result[0] == Approx(5.0f));
    REQUIRE(result[1] == Approx(7.0f));
    REQUIRE(result[2] == Approx(9.0f));
}

TEST_CASE("vector_add_cuda handles empty arrays", "[cuda]") {
    std::vector<float> a, b, result;
    vector_add_cuda(a.data(), b.data(), result.data(), 0);
    REQUIRE(result.empty());
}

TEST_CASE("vector_add_cuda with large input", "[cuda][stress]") {
    int n = 1 << 20;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> result(n);

    vector_add_cuda(a.data(), b.data(), result.data(), n);

    for (int i = 0; i < n; ++i) {
        REQUIRE(result[i] == Approx(3.0f));
    }
}
