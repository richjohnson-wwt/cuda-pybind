// vector_ops.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/vector_add.h"

namespace py = pybind11;

py::array_t<float> add_vectors_cuda(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();

    if (buf_a.size != buf_b.size)
        throw std::runtime_error("Input sizes must match");

    int n = buf_a.size;

    auto result = py::array_t<float>(n);
    auto buf_result = result.request();

    const float* ptr_a = static_cast<float*>(buf_a.ptr);
    const float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    vector_add_cuda(ptr_a, ptr_b, ptr_result, n);

    return result;
}

py::array_t<float> redux_vectors_cuda(py::array_t<float> a, int blockSize, int segments) {
    auto buf_a = a.request();

    auto result = py::array_t<float>(segments);
    auto buf_result = result.request();

    const float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_result = static_cast<float*>(buf_result.ptr);

    vector_redux_cuda(ptr_a, ptr_result, blockSize, segments);

    return result;
}

PYBIND11_MODULE(vector_ops, m) {
    m.def("add_vectors", &add_vectors_cuda, "Add two float vectors using CUDA");
    m.def("redux_vectors", &redux_vectors_cuda, "Redux float vectors using CUDA");
}
