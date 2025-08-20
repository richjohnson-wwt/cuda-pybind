import numpy as np
import vector_ops

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

result = vector_ops.add_vectors(a, b)
print(result)  # Output: [5. 7. 9.]
