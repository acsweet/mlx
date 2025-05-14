import mlx.core as mx

key = mx.random.key(0)

inputs = mx.random.normal((2, 8, 8, 8, 3), dtype=mx.float32, key=key)
kernel = mx.random.normal((2, 3, 3, 3, 3), dtype=mx.float32, key=key)
strides = (2, 2, 2)
mlx_padding = ([0, 0, 0], [1, 1, 1])
dilation_rate = (1, 1, 1)
groups = 1

result = mx.conv_general(
    inputs,
    kernel,
    stride=strides,
    padding=mlx_padding,
    kernel_dilation=dilation_rate,
    input_dilation=1,
    groups=groups,
    flip=False,
)

result_cpu = mx.conv_general(
    inputs,
    kernel,
    stride=strides,
    padding=mlx_padding,
    kernel_dilation=dilation_rate,
    input_dilation=1,
    groups=groups,
    flip=False,
    stream=mx.cpu
)

result_diff = result - result_cpu
print(f'(conv3d) max_diff: {mx.max(result_diff)}')
print(f'(conv3d) total_absolute_diff: {mx.sum(mx.abs(result_diff))}')
# (conv3d) max_diff: 15.351866722106934
# (conv3d) total_absolute_diff: 544.01220703125

inputs = mx.random.normal((2, 10, 10, 3), dtype=mx.float32, key=key)
kernel = mx.random.normal((2, 2, 2, 3), dtype=mx.float32, key=key)
strides = (1, 2)
mlx_padding = ([0, 0], [1, 0])
dilation_rate = (1, 1)
groups = 1

result = mx.conv_general(
    inputs,
    kernel,
    stride=strides,
    padding=mlx_padding,
    kernel_dilation=dilation_rate,
    input_dilation=1,
    groups=groups,
    flip=False,
)

result_cpu = mx.conv_general(
    inputs,
    kernel,
    stride=strides,
    padding=mlx_padding,
    kernel_dilation=dilation_rate,
    input_dilation=1,
    groups=groups,
    flip=False,
    stream=mx.cpu
)

result_diff = result - result_cpu
print(f'(conv2d) max_diff: {mx.max(result_diff)}')
print(f'(conv2d) total_absolute_diff: {mx.sum(mx.abs(result_diff))}')
# (conv2d) max_diff: 2.334087371826172
# (conv2d) total_absolute_diff: 19.29632568359375