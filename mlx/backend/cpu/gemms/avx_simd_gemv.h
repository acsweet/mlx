// Copyright © 2025 Apple Inc.
#pragma once

#include <algorithm>
#include <cstring>
#include <immintrin.h>

#include "mlx/backend/cpu/gemms/aligned_buffer.h"
#include "mlx/backend/cpu/simd/avx_simd.h"

namespace mlx::core {

// Block size for output dimension in outer-product GEMV.
// 4096 floats = 16KB of fp32 accumulator, fits comfortably in L1 cache
// alongside the B row data and vector operand.
constexpr int GEMV_NC_BLOCK = 4096;

// --------------------------------------------------------------------------
// Outer-product GEMV core.
// acc[0:width] += sum_k vec[k] * mat[k * mat_stride + 0 : width]
//
// vec: K contiguous T elements (the "vector" operand)
// mat: K rows of `width` T elements, row stride = mat_stride
// acc: `width` fp32 elements (caller-initialized)
//
// Blocks along the output dimension so the accumulator fits in L1.
// --------------------------------------------------------------------------
template <typename T>
static void gemv_outer_product(
    const T* vec,
    const T* mat,
    float* acc,
    int K, int width, int mat_stride)
{
    constexpr int sw = 8;

    for (int jc = 0; jc < width; jc += GEMV_NC_BLOCK) {
        int nc = std::min(GEMV_NC_BLOCK, width - jc);
        float* acc_block = acc + jc;

        for (int k = 0; k < K; k++) {
            float v = static_cast<float>(vec[k]);
            simd::float8 v_bcast(v);
            const T* mat_row = mat + k * mat_stride + jc;

            // Prefetch start of next row for this block
            if (k + 1 < K) {
                _mm_prefetch(
                    reinterpret_cast<const char*>(mat + (k + 1) * mat_stride + jc),
                    _MM_HINT_T0);
            }

            int j = 0;
            for (; j + sw <= nc; j += sw) {
                simd::float8 m = simd::load_convert_to_float<T>(mat_row + j);
                simd::float8 c = simd::load<float, sw>(acc_block + j);
                simd::store<float, sw>(acc_block + j,
                    simd::fma<float, sw>(v_bcast, m, c));
            }
            for (; j < nc; j++) {
                acc_block[j] += v * static_cast<float>(mat_row[j]);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Dot-product GEMV core.
// acc[i] += dot(mat[i * mat_stride : +K], vec[0:K])   for i = 0..n_outputs-1
//
// Processes 4 rows at once to amortize vec loads across rows.
// --------------------------------------------------------------------------
template <typename T>
static void gemv_dot_product(
    const T* mat,
    const T* vec,
    float* acc,
    int n_outputs, int K, int mat_stride)
{
    constexpr int sw = 8;
    constexpr int UNROLL = 4;

    int i = 0;
    for (; i + UNROLL <= n_outputs; i += UNROLL) {
        simd::float8 s0, s1, s2, s3;

        const T* r0 = mat + (i + 0) * mat_stride;
        const T* r1 = mat + (i + 1) * mat_stride;
        const T* r2 = mat + (i + 2) * mat_stride;
        const T* r3 = mat + (i + 3) * mat_stride;

        int k = 0;
        for (; k + sw <= K; k += sw) {
            simd::float8 v = simd::load_convert_to_float<T>(vec + k);
            s0 = simd::fma<float, sw>(simd::load_convert_to_float<T>(r0 + k), v, s0);
            s1 = simd::fma<float, sw>(simd::load_convert_to_float<T>(r1 + k), v, s1);
            s2 = simd::fma<float, sw>(simd::load_convert_to_float<T>(r2 + k), v, s2);
            s3 = simd::fma<float, sw>(simd::load_convert_to_float<T>(r3 + k), v, s3);
        }

        float d0 = simd::sum(s0);
        float d1 = simd::sum(s1);
        float d2 = simd::sum(s2);
        float d3 = simd::sum(s3);

        for (; k < K; k++) {
            float vk = static_cast<float>(vec[k]);
            d0 += vk * static_cast<float>(r0[k]);
            d1 += vk * static_cast<float>(r1[k]);
            d2 += vk * static_cast<float>(r2[k]);
            d3 += vk * static_cast<float>(r3[k]);
        }

        acc[i + 0] += d0;
        acc[i + 1] += d1;
        acc[i + 2] += d2;
        acc[i + 3] += d3;
    }

    for (; i < n_outputs; i++) {
        simd::float8 s;
        const T* row = mat + i * mat_stride;

        int k = 0;
        for (; k + sw <= K; k += sw) {
            simd::float8 v = simd::load_convert_to_float<T>(vec + k);
            s = simd::fma<float, sw>(simd::load_convert_to_float<T>(row + k), v, s);
        }

        float d = simd::sum(s);
        for (; k < K; k++) {
            d += static_cast<float>(vec[k]) * static_cast<float>(row[k]);
        }
        acc[i] += d;
    }
}

// --------------------------------------------------------------------------
// Public GEMV interface.
// Handles M=1 and N=1 with all transpose combinations.
// C = alpha * op(A) * op(B) + beta * C
//
// Dispatch logic:
//   M=1, B not transposed → outer product (SIMD along N, stream B rows)
//   M=1, B transposed     → dot product   (SIMD along K, one dot per j)
//   N=1, A not transposed → dot product   (SIMD along K, one dot per i)
//   N=1, A transposed     → outer product (SIMD along M, stream A cols)
// --------------------------------------------------------------------------
template <typename T>
void simd_gemv(
    const T* a, const T* b, T* c,
    bool a_trans, bool b_trans,
    int M, int N, int K,
    int ldA, int ldB, int ldC,
    float alpha, float beta)
{
    int out_len = (M == 1) ? N : M;

    // fp32 accumulator (thread-local, grow-only)
    thread_local aligned_unique_ptr<float> acc_buf(1);
    acc_buf.reset(out_len);
    float* acc = acc_buf.get();

    // Initialize accumulator: acc = beta * C
    // When M=1, C is 1×N contiguous. When N=1, C is M×1 contiguous (ldC=1).
    constexpr int sw = 8;

    if (beta != 0.0f) {
        simd::float8 beta_vec(beta);
        int j = 0;
        for (; j + sw <= out_len; j += sw) {
            simd::float8 cv = simd::load_convert_to_float<T>(c + j);
            simd::store<float, sw>(acc + j, beta_vec * cv);
        }
        for (; j < out_len; j++) {
            acc[j] = beta * static_cast<float>(c[j]);
        }
    } else {
        std::memset(acc, 0, out_len * sizeof(float));
    }

    // Accumulate: acc += op(A) * op(B)
    if (M == 1) {
        // A is always contiguous for M=1: a[k] for k=0..K-1
        if (!b_trans) {
            // B is row-major K×N, stride ldB → outer product along N
            gemv_outer_product(a, b, acc, K, N, ldB);
        } else {
            // B stored as N×K, stride ldB → dot product per output j
            gemv_dot_product(b, a, acc, N, K, ldB);
        }
    } else {
        // N=1: B is always contiguous: b[k] for k=0..K-1
        if (!a_trans) {
            // A is row-major M×K, stride ldA → dot product per output i
            gemv_dot_product(a, b, acc, M, K, ldA);
        } else {
            // A stored as K×M, stride ldA → outer product along M
            gemv_outer_product(b, a, acc, K, M, ldA);
        }
    }

    // Write back: C = alpha * acc (convert fp32 → T)
    bool apply_alpha = (alpha != 1.0f);
    simd::float8 alpha_vec(alpha);
    int j = 0;
    for (; j + sw <= out_len; j += sw) {
        simd::float8 val = simd::load<float, sw>(acc + j);
        if (apply_alpha) val = alpha_vec * val;
        simd::store_convert_from_float<T>(c + j, val);
    }
    for (; j < out_len; j++) {
        float val = acc[j];
        if (apply_alpha) val *= alpha;
        c[j] = static_cast<T>(val);
    }
}

} // namespace mlx::core
