import mlx.core as mx
import time
import numpy as np

size = 1024
dtype = mx.float16
print(size)

mx_a = mx.random.normal((size, size), dtype=dtype)
mx_b = mx.random.normal((size, size), dtype=dtype)
print('perform matmul')
start_time = time.time()
mx_ab = mx.matmul(mx_a, mx_b)
mx.eval(mx_ab)
print(time.time() - start_time)

print('convert to numpy')
start_time = time.time()
mx_ab_np = np.array(mx_ab)
print(time.time() - start_time)

mx_a_np = np.random.normal((size, size))
mx_b_np = np.random.normal((size, size))
# mx_a_np = np.array(mx_a)
# mx_b_np = np.array(mx_b)
print('perform matmul')
start_time = time.time()
np_ab = np.matmul(mx_a_np, mx_b_np)
print(time.time() - start_time)