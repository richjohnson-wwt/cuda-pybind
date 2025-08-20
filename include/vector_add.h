#pragma once

void vector_add_cuda(const float* a, const float* b, float* result, int n);
void vector_redux_cuda(const float* a, float* result, int blockSize, int segments);