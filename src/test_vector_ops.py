import numpy as np
import pytest
import vector_ops

def test_add_basic():
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    expected = a + b
    np.testing.assert_array_equal(result, expected)

def test_add_zeros():
    a = np.zeros(5, dtype=np.float32)
    b = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    np.testing.assert_array_equal(result, b)

def test_add_negative():
    a = np.array([-1, -2, -3], dtype=np.float32)
    b = np.array([3, 2, 1], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    np.testing.assert_array_equal(result, np.array([2, 0, -2], dtype=np.float32))

def test_empty_input():
    a = np.array([], dtype=np.float32)
    b = np.array([], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    assert result.size == 0

def test_single_element():
    a = np.array([42.0], dtype=np.float32)
    b = np.array([58.0], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    np.testing.assert_array_equal(result, np.array([100.0], dtype=np.float32))

def test_size_mismatch():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0], dtype=np.float32)
    with pytest.raises(RuntimeError):
        vector_ops.add_vectors(a, b)

def test_float_precision():
    a = np.array([1e-7, 1e7], dtype=np.float32)
    b = np.array([1e-7, -1e7], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    expected = np.array([2e-7, 0.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_nan_inf():
    a = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    b = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    result = vector_ops.add_vectors(a, b)

    # Check first value is still NaN
    assert np.isnan(result[0])

    # inf + (-1.0) is still inf (floating-point rounding behavior)
    assert np.isinf(result[1])
    assert result[1] == np.inf

    # -inf + 1.0 is still -inf
    assert np.isneginf(result[2])


def test_large_input():
    a = np.ones(10_000_000, dtype=np.float32)
    b = np.ones(10_000_000, dtype=np.float32)
    result = vector_ops.add_vectors(a, b)
    np.testing.assert_array_equal(result, a + b)

def test_multiple_calls():
    for _ in range(1000):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        result = vector_ops.add_vectors(a, b)
        np.testing.assert_array_equal(result, a + b)
