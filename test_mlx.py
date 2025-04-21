import mlx.core as mx
import time
import pandas as pd
import numpy as np

def benchmark(dtype, sizes, warumup=25, bench_runs=50):
    df_bench = pd.DataFrame()
    for size in sizes:
        a = mx.random.normal((size, size), dtype=dtype)
        b = mx.random.normal((size, size), dtype=dtype)

        for _ in range(warumup):
            c = mx.matmul(a, b)
            mx.eval(c)

        times = []
        for _ in range(bench_runs):
            start = time.time()
            c = mx.matmul(a, b)
            mx.eval(c)
            duration = time.time() - start
            times.append(duration * 1000)
        _df = pd.DataFrame({"time": times})
        _df['size'] = size
        df_bench = pd.concat([df_bench, _df], ignore_index=True)
    df_bench['dtype'] = str(dtype).replace('mlx.core.', '')

    return df_bench

def benchmark_np(dtype, sizes, warumup=25, bench_runs=50):
    df_bench = pd.DataFrame()
    for size in sizes:
        # rng = np.random.default_rng()
        # a = rng.standard_normal(size=(size, size), dtype=dtype)
        # b = rng.standard_normal(size=(size, size), dtype=dtype)
        a = np.random.normal((size, size)).astype(dtype)
        b = np.random.normal((size, size)).astype(dtype)
        _ = a.sum()
        _ = b.sum()

        for _ in range(warumup):
            c = np.matmul(a, b)

        times = []
        for _ in range(bench_runs):
            start = time.time()
            c = np.matmul(a, b)
            res = c.sum()
            duration = time.time() - start
            times.append(duration * 1000)
        _df = pd.DataFrame({"time": times})
        _df['size'] = size
        df_bench = pd.concat([df_bench, _df], ignore_index=True)
    df_bench['dtype'] = 'np.' + np.dtype(dtype).name

    return df_bench

df_result = pd.DataFrame()
sizes = [2**i for i in range(4, 11)]

dtypes = [mx.float16, mx.bfloat16, mx.float32]
for dtype in dtypes:
    _df_result = benchmark(dtype, sizes)
    df_result = pd.concat([df_result, _df_result], ignore_index=True)

dtypes = [np.float32] # np.float16, 
for dtype in dtypes:
    _df_result = benchmark_np(dtype, sizes)
    df_result = pd.concat([df_result, _df_result], ignore_index=True)

df_results_grouped = df_result.groupby(['dtype', 'size']).agg({'time': 'mean'}).reset_index()
df_results_grouped_pivot = df_results_grouped.pivot(index='size', columns='dtype', values='time')
print(df_results_grouped_pivot.to_markdown())
