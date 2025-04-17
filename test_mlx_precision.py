import mlx.core as mx
import numpy as np
import ml_dtypes

size = 128
dtype = mx.float16
mx_a = mx.random.normal((size, size), dtype=dtype)
mx_b = mx.random.normal((size, size), dtype=dtype)
mx_ab = mx.matmul(mx_a, mx_b)
print(mx_ab.shape, mx_ab.dtype)

a = np.array(mx_a)
b = np.array(mx_b) #.astype(mx.float32)).astype(ml_dtypes.bfloat16)
np_ab = np.matmul(a, b)
print(np_ab.shape, np_ab.dtype)

all_close = np.allclose(np_ab, np.array(mx_ab))
if not all_close:
    max_abs_diff = np.max(np.abs(mx_ab - np_ab))
    total_abs_diff = np.sum(np.abs(mx_ab - np_ab))
    print('max_abs_diff', max_abs_diff)
    print('total_abs_diff', total_abs_diff)
    print('a', a.max(), a.min())
    print('b', b.max(), b.min())
    print('a', mx_a.max(), mx_a.min())
    print('b', mx_b.max(), mx_b.min())
# print('allclose:', np.allclose(np_ab, np.array(mx_ab)))

# array([[-1.10449, -0.057312],
#        [0, 0],
#        [0, 0],
#        ...,
#        [0, 0],
#        [0, 0],
#        [0.832031, 1.45996]], dtype=float16)
# [[-1.1045  -0.0573 ]
#  [-0.09705  1.577  ]
#  [-0.8423   1.672  ]
#  [-2.223    3.398  ]
#  [-0.412   -1.627  ]
#  [-1.802    0.604  ]
#  [ 0.832    1.46   ]]