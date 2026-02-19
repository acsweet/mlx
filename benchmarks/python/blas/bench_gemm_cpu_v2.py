# Copyright © 2023 Apple Inc.

import argparse
import math
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
import torch

# device_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
# device_name = device_name.decode("utf-8").strip("\n")

N_warmup = 8
N_iter_bench = 80
N_iter_func = 5


def bench(f, a, b):
    for i in range(N_warmup):
        f(a, b)
    # torch.mps.synchronize()

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(a, b)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def gemm_nn_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_nt_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b.transpose((0, 2, 1))
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_tn_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose((0, 2, 1)) @ b
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_tt_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose((0, 2, 1)) @ b.transpose((0, 2, 1))
        ys.append(y)
    mx.eval(ys)
    return ys


@torch.no_grad()
def gemm_nn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    # torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_nt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b.transpose(-1, -2)
        ys.append(y)
    # torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_tn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b
        ys.append(y)
    # torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_tt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b.transpose(-1, -2)
        ys.append(y)
    # torch.mps.synchronize()
    return ys


def gemm_nn_numpy(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    return ys


def gemm_nt_numpy(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ np.swapaxes(b, -1, -2)
        ys.append(y)
    return ys


def gemm_tn_numpy(a, b):
    ys = []
    for i in range(N_iter_func):
        y = np.swapaxes(a, -1, -2) @ b
        ys.append(y)
    return ys


def gemm_tt_numpy(a, b):
    ys = []
    for i in range(N_iter_func):
        y = np.swapaxes(a, -1, -2) @ np.swapaxes(b, -1, -2)
        ys.append(y)
    return ys


def bench_shape(B, M, N, K, np_dtype, transpose="nn", backends=("mlx", "torch", "numpy")):
    shape_a = (B, M, K) if transpose[0] == "n" else (B, K, M)
    shape_b = (B, K, N) if transpose[1] == "n" else (B, N, K)

    a_np = np.random.normal(0.0, 1.0 / math.sqrt(M + K), shape_a).astype(np_dtype)
    b_np = np.random.normal(0.0, 1.0 / math.sqrt(N + K), shape_b).astype(np_dtype)

    f_mx_map = {
        "nn": gemm_nn_mlx,
        "nt": gemm_nt_mlx,
        "tn": gemm_tn_mlx,
        "tt": gemm_tt_mlx,
    }
    f_pt_map = {
        "nn": gemm_nn_torch,
        "nt": gemm_nt_torch,
        "tn": gemm_tn_torch,
        "tt": gemm_tt_torch,
    }
    f_np_map = {
        "nn": gemm_nn_numpy,
        "nt": gemm_nt_numpy,
        "tn": gemm_tn_numpy,
        "tt": gemm_tt_numpy,
    }

    times = {}

    if "mlx" in backends:
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        times["mlx"] = bench(f_mx_map[transpose], a_mx, b_mx)

    if "torch" in backends:
        a_pt = torch.from_numpy(a_np)
        b_pt = torch.from_numpy(b_np)
        times["torch"] = bench(f_pt_map[transpose], a_pt, b_pt)

    if "numpy" in backends:
        times["numpy"] = bench(f_np_map[transpose], a_np, b_np)

    # Correctness check (mlx vs numpy fp32 reference)
    if "mlx" in backends:
        t_a = (0, 1, 2) if transpose[0] == "n" else (0, 2, 1)
        t_b = (0, 1, 2) if transpose[1] == "n" else (0, 2, 1)

        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        c_mlx = a_mx.transpose(t_a) @ b_mx.transpose(t_b)
        c_npy = a_np.transpose(t_a).astype(np.float32) @ b_np.transpose(t_b).astype(
            np.float32
        )

        atol = 1e-5 if np_dtype == np.float32 else 1e-4

        if not np.allclose(c_mlx, c_npy.astype(np_dtype), atol=atol):
            print(
                f"Failed at {(B, M, N, K)} [transpose = {transpose}] with max(|a - b|) = {np.max(np.abs(c_npy - c_mlx))}"
            )

    return times


def get_gflop_count(B, M, N, K):
    return float(2.0 * N_iter_bench * N_iter_func * B * M * N * K) / float(1024.0**3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gemm benchmarks")
    parser.add_argument(
        "--backends", nargs="+", default=["mlx", "numpy", "torch"],
        choices=["mlx", "torch", "numpy"],
        help="Which backends to benchmark (default: mlx numpy torch)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced iteration counts for faster runs",
    )
    args = parser.parse_args()

    if args.quick:
        N_warmup = 2
        N_iter_bench = 10
        N_iter_func = 2

    backends = tuple(args.backends)

    dtypes = ("float16", "float32")
    transposes = ("nn", ) # ("nn", "nt", "tn")
    shapes = (
        (16, 234, 768, 3072),
        (1, 64, 64, 25344),
        (16, 512, 512, 512),
        (1, 1024, 1024, 2048),
        # (4, 1024, 1024, 4096),
        # (4, 1024, 4096, 1024),
        # (1, 4096, 4096, 4096),
    )

    # Print header
    header = f"{'B':>3s}, {'M':>4s}, {'N':>4s}, {'K':>4s}, {'dtype':>7s}, {'tp':>2s}"
    for b in backends:
        header += f", {b:>7s}"
    if len(backends) >= 2:
        header += f", {'diff':>7s}"
    print(header)

    for dtype in dtypes:
        for transpose in transposes:
            for B, M, N, K in shapes:
                np_dtype = getattr(np, dtype)
                times = bench_shape(B, M, N, K, np_dtype, transpose, backends)

                gflop_count = get_gflop_count(B, M, N, K)
                gflops = {b: gflop_count / times[b] for b in backends if b in times}

                line = f"{B:3d}, {M:4d}, {N:4d}, {K:4d}, {dtype:>7s}, {transpose:>2s}"
                for b in backends:
                    if b in gflops:
                        line += f", {gflops[b]:7.3f}"
                    else:
                        line += f", {'N/A':>7s}"

                # Show diff: mlx vs best of the other backends
                if "mlx" in gflops and len(gflops) >= 2:
                    others = {b: g for b, g in gflops.items() if b != "mlx"}
                    best_other = max(others.values())
                    diff = gflops["mlx"] / best_other - 1.0
                    line += f", {100. * diff:+6.1f}%"

                print(line)
