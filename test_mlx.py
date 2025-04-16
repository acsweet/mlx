import mlx.core as mx
import time
import pandas as pd

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

sizes = [2**i for i in range(4, 11)]
dtypes = [mx.float16, mx.bfloat16, mx.float32]

df_result = pd.DataFrame()
for dtype in dtypes:
    _df_result = benchmark(dtype, sizes)
    df_result = pd.concat([df_result, _df_result], ignore_index=True)

df_results_grouped = df_result.groupby(['dtype', 'size']).agg({'time': 'mean'}).reset_index()
df_results_grouped_pivot = df_results_grouped.pivot(index='size', columns='dtype', values='time')
print(df_results_grouped_pivot.to_markdown())
