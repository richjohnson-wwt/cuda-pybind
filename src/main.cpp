#include <iostream>
#include <vector>
#include "../include/vector_add.h"

void run_redux() {
    std::cout << "C++ Redux Example" << std::endl;
    const int segments = 3;
    const int blockSize = 4;
    float a[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float result[segments];
    vector_redux_cuda(a, result, blockSize, segments);
    std::cout << "Result:   ";
    for (int i = 0; i < segments; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}

void run_add() {
    std::cout << "C++ Add Example" << std::endl;
    
    // Create test vectors - same values as in app.py
    const int n = 3;
    float a[n] = {1.0f, 2.0f, 3.0f};
    float b[n] = {4.0f, 5.0f, 6.0f};
    float result[n];
    
    // Call CUDA vector addition function directly
    vector_add_cuda(a, b, result, n);
    
    std::cout << "Result:   ";
    for (int i = 0; i < n; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    run_add();
    run_redux();    
    return 0;
}

