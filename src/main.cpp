#include <iostream>
#include <vector>
#include "../include/vector_add.h"

int main() {
    std::cout << "CUDA Vector Addition Demo (C++)" << std::endl;
    
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
    
    std::cout << "Expected: 5.0 7.0 9.0" << std::endl;
    
    return 0;
}
