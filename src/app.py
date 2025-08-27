import numpy as np
import vector_ops

def redux_vectors(a, blockSize, segments):
    return vector_ops.redux_vectors(a, blockSize, segments)

def add_vectors(a, b):
    return vector_ops.add_vectors(a, b)

print("Python ADD EXAMPLE")
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
result = add_vectors(a, b)
print(result)

print("Python REDUX EXAMPLE")
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
result = redux_vectors(a, 4, 3)
print(result)