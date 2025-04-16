// Copyright © 2025 Apple Inc.
#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>

// Ensure SIMD header is included first
#include "mlx/backend/cpu/simd/avx_simd_matmul.h"

namespace mlx::core {

// Helper for integer division rounded up
inline int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// --- Packing Functions ---
template <typename T, typename AccT, int MC, int KC>
static void pack_A_block(
    const T* A, AccT* A_packed, int M, int K, int M_offset, int K_offset,
    int m_block, int k_block, bool a_trans)
{
    static_assert(std::is_same_v<AccT, float>, "Packing requires float accumulator type");
    if (!a_trans) {
        for (int i = 0; i < m_block; ++i) {
            const T* a_src_row = A + (M_offset + i) * K + K_offset;
            AccT* a_dst_row = A_packed + i * KC;
            for (int k = 0; k < k_block; ++k) {
                a_dst_row[k] = static_cast<AccT>(a_src_row[k]);
            }
            for (int k = k_block; k < KC; ++k) {
                a_dst_row[k] = AccT(0);
            }
        }
        for (int i = m_block; i < MC; ++i) {
           AccT* a_dst_row = A_packed + i * KC;
           std::fill(a_dst_row, a_dst_row + KC, AccT(0));
        }
    } else {
        for (int k = 0; k < k_block; ++k) {
             for (int i = 0; i < m_block; ++i) {
                A_packed[i * KC + k] = static_cast<AccT>(A[(K_offset + k) * M + (M_offset + i)]);
             }
             for(int i = m_block; i < MC; ++i) {
                A_packed[i * KC + k] = AccT(0);
             }
        }
        for(int k = k_block; k < KC; ++k) {
            for(int i=0; i< MC; ++i) {
                 A_packed[i * KC + k] = AccT(0);
            }
        }
    }
}

template <typename T, typename AccT, int KC, int NC>
static void pack_B_block(
    const T* B, AccT* B_packed, int K, int N, int K_offset, int N_offset,
    int k_block, int n_block, bool b_trans)
{
    static_assert(std::is_same_v<AccT, float>, "Packing requires float accumulator type");
    if (!b_trans) {
        for (int k = 0; k < k_block; ++k) {
            const T* b_src_row = B + (K_offset + k) * N + N_offset;
            AccT* b_dst_row = B_packed + k * NC;
            for (int j = 0; j < n_block; ++j) {
                b_dst_row[j] = static_cast<AccT>(b_src_row[j]);
            }
            for(int j = n_block; j < NC; ++j) {
                b_dst_row[j] = AccT(0);
            }
        }
        for (int k = k_block; k < KC; ++k) {
            AccT* b_dst_row = B_packed + k * NC;
            std::fill(b_dst_row, b_dst_row + NC, AccT(0));
        }
    } else {
        for (int k = 0; k < k_block; ++k) {
            AccT* b_dst_row = B_packed + k * NC;
             for (int j = 0; j < n_block; ++j) {
                 b_dst_row[j] = static_cast<AccT>(B[(N_offset + j) * K + (K_offset + k)]);
             }
             for(int j = n_block; j < NC; ++j) {
                b_dst_row[j] = AccT(0);
             }
        }
        for (int k = k_block; k < KC; ++k) {
            AccT* b_dst_row = B_packed + k * NC;
            std::fill(b_dst_row, b_dst_row + NC, AccT(0));
        }
    }
}

// --- Micro Kernel ---
template <int MR, int NR, typename T, typename AccT>
static void compute_fma_microkernel(
    const AccT* A_panel,
    const AccT* B_panel,
    T* C,
    int ldc,
    int kc,
    int nc_packed,
    float alpha,
    float beta)
{
    static_assert(std::is_same_v<AccT, float>, "FMA microkernel requires float accumulator");
    static_assert(NR > 0 && NR % simd::max_size<AccT> == 0, "NR must be a positive multiple of SIMD width");

    constexpr int simd_width = simd::max_size<AccT>;
    constexpr int num_b_vectors = NR / simd_width;

    using Vec = simd::Simd<AccT, simd_width>;

    // Initialize accumulation registers to zero
    Vec c_regs[MR][num_b_vectors];
    for (int i = 0; i < MR; ++i) {
        for (int bj = 0; bj < num_b_vectors; ++bj) {
            c_regs[i][bj] = Vec(0.0f);
        }
    }
    
    Vec alpha_vec(alpha);
    Vec beta_vec(beta);

    // Main computation loop
    for (int k = 0; k < kc; ++k) {
        Vec b_k[num_b_vectors];
        for(int bj = 0; bj < num_b_vectors; ++bj) {
            b_k[bj] = simd::load<AccT, simd_width>(B_panel + k * nc_packed + bj * simd_width);
        }

        for (int i = 0; i < MR; ++i) {
            Vec a_ik = simd::broadcast<AccT, simd_width>(A_panel + i * kc + k);
            for (int bj = 0; bj < num_b_vectors; ++bj) {
                c_regs[i][bj] = simd::fma<AccT, simd_width>(a_ik, b_k[bj], c_regs[i][bj]);
            }
        }
    }

    // Store results back to memory
    for (int i = 0; i < MR; ++i) {
        for (int bj = 0; bj < num_b_vectors; ++bj) {
            T* c_ptr = C + i * ldc + bj * simd_width;
            simd::scale_accumulate_store<T>(c_ptr, c_regs[i][bj], alpha_vec, beta_vec);
        }
    }
}

// --- Scalar Kernel for Edges/Partial Tiles ---
template <typename T, typename AccT>
static void compute_block_scalar_partial(
    const AccT* A_panel, const AccT* B_panel, T* C, int ldc,
    int m_micro, int n_micro, int k_block, int nc_packed,
    float alpha, float beta)
{
    static_assert(std::is_same_v<AccT, float>, "Scalar kernel needs float accumulator");
    for (int i = 0; i < m_micro; ++i) {
        for (int j = 0; j < n_micro; ++j) {
            AccT acc = 0.0;
            const AccT* a_ptr = A_panel + i * k_block;
            for (int k = 0; k < k_block; ++k) {
                acc += a_ptr[k] * B_panel[k * nc_packed + j];
            }

            T* c_ptr = C + i * ldc + j;
            // Perform beta scaling safely
            AccT c_old = (beta == 0.0f) ? AccT(0) : static_cast<AccT>(*c_ptr);
            *c_ptr = static_cast<T>(alpha * acc + beta * c_old);
        }
    }
}

/**
 * Optimized single-threaded matrix multiplication using AVX/FMA
 */
template <typename T, typename AccT>
void simd_gemm_optimized(
    const T* a, const T* b, T* c,
    bool a_trans, bool b_trans,
    int M, int N, int K, float alpha, float beta)
{
    static_assert(std::is_same_v<AccT, float>, "Optimized GEMM requires float accumulator");

    // Blocking Parameters
    constexpr int MC = 128;
    constexpr int NC = 256;
    constexpr int KC = 128;
    constexpr int MR = 6;
    constexpr int NR = 16;
    static_assert(NR % simd::max_size<AccT> == 0, "NR must be multiple of SIMD width");

    std::vector<AccT> A_packed_vec(MC * KC);
    std::vector<AccT> B_packed_vec(KC * NC);

    // Determine original dimensions for packing based on transpose flags
    int A_rows = a_trans ? K : M;
    int A_cols = a_trans ? M : K;
    int B_rows = b_trans ? N : K;
    int B_cols = b_trans ? K : N;

    for (int j = 0; j < N; j += NC) {
        int n_block = std::min(NC, N - j);

        for (int k = 0; k < K; k += KC) {
            int k_block = std::min(KC, K - k);

            // Pack B Panel (KC x NC)
            pack_B_block<T, AccT, KC, NC>(
                b, B_packed_vec.data(),
                B_rows, B_cols,
                k, j,
                k_block, n_block,
                b_trans);

            for (int i = 0; i < M; i += MC) {
                int m_block = std::min(MC, M - i);

                // Pack A Panel (MC x KC)
                pack_A_block<T, AccT, MC, KC>(
                    a, A_packed_vec.data(),
                    A_rows, A_cols,
                    i, k,
                    m_block, k_block,
                    a_trans);

                // Micro-kernel Computation
                for (int ii = 0; ii < m_block; ii += MR) {
                    int m_micro = std::min(MR, m_block - ii);

                    for (int jj = 0; jj < n_block; jj += NR) {
                        int n_micro = std::min(NR, n_block - jj);

                        // Pointer calculations
                        const AccT* a_sub_panel = A_packed_vec.data() + ii * KC;
                        const AccT* b_sub_panel = B_packed_vec.data() + jj;
                        T* c_sub = c + (i + ii) * N + (j + jj);

                        if (m_micro == MR && n_micro == NR) {
                            // Full tile - use optimized kernel
                            compute_fma_microkernel<MR, NR, T, AccT>(
                                a_sub_panel,
                                b_sub_panel,
                                c_sub,
                                N,        // ldc
                                k_block,  // kc
                                NC,       // nc_packed
                                alpha,
                                beta
                            );
                        } else {
                            // Partial tile - use scalar kernel
                            compute_block_scalar_partial<T, AccT>(
                                a_sub_panel,
                                b_sub_panel,
                                c_sub,
                                N,        // ldc
                                m_micro, n_micro, k_block,
                                NC,       // nc_packed
                                alpha, beta
                             );
                        }
                    }
                }
            }
        }
    }
}

/**
 * Public interface for SIMD GEMM
 */
template <typename T, typename AccT = float>
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f) {

    simd_gemm_optimized<T, AccT>(a, b, c, a_trans, b_trans, M, N, K, alpha, beta);
}

} // namespace mlx::core