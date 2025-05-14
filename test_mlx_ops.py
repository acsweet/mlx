import mlx.core as mx
import numpy as np

# cases = [(np.float32, 1e-6), (np.float16, 1e-3)]

# for dtype, atol in cases:
#     a_npy = np.random.randn(16, 8, 32).astype(dtype)
#     a_mlx = mx.array(a_npy)

#     def np_softmax(x, axis):
#         ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
#         return ex / np.sum(ex, axis=axis, keepdims=True)

#     for axes in (None, 0, 1, 2, (0, 1), (1, 2), (0, 2), (0, 1, 2)):
#         b_npy = np_softmax(a_npy, axes)
#         b_mlx = mx.softmax(a_mlx, axes)
#         assert np.allclose(b_npy, b_mlx, atol=atol), "not close"

a = mx.array([0, float("nan")], dtype=mx.float32)
print(a, mx.isnan(a))

x = mx.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan")])
y = mx.isnan(x)
print(x, y)