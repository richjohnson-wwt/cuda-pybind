// vector_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Function to add two NumPy arrays
py::array_t<float> add_vectors(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();

    if (buf_a.size != buf_b.size)
        throw std::runtime_error("Input sizes must match");

    auto result = py::array_t<float>(buf_a.size);
    auto buf_result = result.request();

    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_res = static_cast<float*>(buf_result.ptr);

    for (ssize_t i = 0; i < buf_a.size; i++) {
        ptr_res[i] = ptr_a[i] + ptr_b[i];
    }

    return result;
}

PYBIND11_MODULE(vector_ops, m) {
    m.def("add_vectors", &add_vectors, "Add two float vectors (1D NumPy arrays)");
}
