// Copyright © 2025 Apple Inc.
#pragma once // Remove if this is a pure .cpp file

#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept> // For invalid_argument
#include <limits>    // For numeric_limits
#include <cstring>   // For std::fill needed in packing

// Use the revised SIMD header (only float support needed now)
#include "mlx/backend/cpu/simd/avx_simd_matmul.h"

namespace mlx::core {

// Helper for integer division rounded up (remains the same)
inline int ceildiv(int a, int b) {
    if (b == 0) throw std::invalid_argument("Division by zero in ceildiv");
    if (a <= 0) return 0; // Handle non-positive a
    return (a + b - 1) / b;
}

// --- Packing Functions (T -> float) ---
// Pack A block (m_block x k_block) from T into A_packed (MC x KC float), column-major layout
template <typename T, int MC, int KC>
static void pack_A_block(
    const T* A, float* A_packed, // Target is float buffer
    int M, int K, int ldA,
    int M_offset, int K_offset,
    int m_block, int k_block, bool a_trans)
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>, "T must be float16 or bfloat16");
    // Zero the destination buffer (important for padding)
    std::fill(A_packed, A_packed + MC * KC, 0.0f);

    if (!a_trans) { // A is M x K, row-major (ldA >= K)
        for (int k = 0; k < k_block; ++k) {
            const T* a_src_col_elem_ptr = A + M_offset * ldA + (K_offset + k); // Ptr to A[M_offset, K_offset+k]
            float* a_dst_col_ptr = A_packed + k * MC; // Packed column k
            for (int i = 0; i < m_block; ++i) {
                // Read A[M_offset+i, K_offset+k] (accessed via strides)
                // Store into packed[i + k*MC]
                a_dst_col_ptr[i] = static_cast<float>(*(a_src_col_elem_ptr + i * ldA));
            }
        }
    } else { // A is K x M, row-major (ldA >= M), accessed as transposed M x K
        for (int k = 0; k < k_block; ++k) {
            const T* a_src_row_ptr = A + (K_offset + k) * ldA + M_offset; // Ptr to A[K_offset+k, M_offset]
            float* a_dst_col_ptr = A_packed + k * MC; // Packed column k
            for (int i = 0; i < m_block; ++i) {
                 // Read A[K_offset+k, M_offset+i] (contiguous in row)
                 // Store into packed[i + k*MC]
                a_dst_col_ptr[i] = static_cast<float>(a_src_row_ptr[i]);
            }
        }
    }
}

// Pack B block (k_block x n_block) from T into B_packed (KC x NC float), row-major layout
template <typename T, int KC, int NC>
static void pack_B_block(
    const T* B, float* B_packed, // Target is float buffer
    int K, int N, int ldB,
    int K_offset, int N_offset,
    int k_block, int n_block, bool b_trans)
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>, "T must be float16 or bfloat16");
    // Zero the destination buffer (important for padding)
    std::fill(B_packed, B_packed + KC * NC, 0.0f);

    if (!b_trans) { // B is K x N, row-major (ldB >= N)
        for (int k = 0; k < k_block; ++k) {
            const T* b_src_row_ptr = B + (K_offset + k) * ldB + N_offset; // Ptr to B[K_offset+k, N_offset]
            float* b_dst_row_ptr = B_packed + k * NC; // Packed row k
            for (int j = 0; j < n_block; ++j) {
                 // Read B[K_offset+k, N_offset+j] (contiguous in row)
                 // Store into packed[k*NC + j]
                 b_dst_row_ptr[j] = static_cast<float>(b_src_row_ptr[j]);
            }
        }
    } else { // B is N x K, row-major (ldB >= K), accessed as transposed K x N
        for (int k = 0; k < k_block; ++k) {
             float* b_dst_row_ptr = B_packed + k * NC; // Packed row k
             // Need pointer to column k in B (which is row k of B^T)
             // The element B[N_offset+j, K_offset+k] is at B + (N_offset+j)*ldB + (K_offset+k)
             for (int j = 0; j < n_block; ++j) {
                 // Read B[N_offset+j, K_offset+k]
                 const T* b_src_elem_ptr = B + (N_offset + j) * ldB + (K_offset + k);
                 // Store into packed[k*NC + j]
                 b_dst_row_ptr[j] = static_cast<float>(*b_src_elem_ptr);
             }
        }
    }
}


// --- Micro Kernel (AccT = float) ---
// Computes C[:MR, :NR] = alpha * (A[:MR, :kc] * B[:kc, :NR]) + beta * C_old[:MR, :NR]
template <int MR, int NR, typename T> // T = float16 or bfloat16
static void compute_fma_microkernel(
    const float* A_panel, // Packed A (MC x KC, col-major access: A[i + k*MC_pack_stride])
    const float* B_panel, // Packed B (KC x NC, row-major access: B[k*NC_pack_stride + j])
    T* C,                 // Output C (row major, M x N)
    int ldc,              // Leading dimension of C
    int kc,               // Inner dimension length for this block (<= KC)
    int MC_pack_stride,   // Row stride in packed A (MC_BLOCK)
    int NC_pack_stride,   // Col stride in packed B (NC_BLOCK) - Actually row length!
    float alpha,          // Scalar alpha
    float effective_beta) // Scalar beta for THIS K-panel pass
{
    using float8 = simd::float8;
    constexpr int simd_width = 8;
    static_assert(NR > 0 && NR % simd_width == 0, "NR must be multiple of SIMD width 8");
    constexpr int num_b_vectors = NR / simd_width;

    float8 c_regs[MR][num_b_vectors]; // FP32 Accumulators for A*B panel result

    // Initialize accumulation registers to zero for this panel
    for (int i = 0; i < MR; ++i) {
        for (int bj = 0; bj < num_b_vectors; ++bj) {
            c_regs[i][bj] = float8(0.0f);
        }
    }

    // Accumulate A*B panel result into c_regs
    for (int k = 0; k < kc; ++k) {
        // B panel is row-major packed (KC x NC), stride is NC_pack_stride
        const float* b_row_k_ptr = B_panel + k * NC_pack_stride;
        float8 b_k_vecs[num_b_vectors];
        for(int bj = 0; bj < num_b_vectors; ++bj) {
            b_k_vecs[bj] = simd::load<float, 8>(b_row_k_ptr + bj * simd_width);
        }

        // A panel is col-major packed (MC x KC), stride is MC_pack_stride
        const float* a_col_k_ptr = A_panel + k * MC_pack_stride;
        for (int i = 0; i < MR; ++i) {
            float8 a_ik = simd::broadcast<float, 8>(a_col_k_ptr + i);
            for (int bj = 0; bj < num_b_vectors; ++bj) {
                c_regs[i][bj] = simd::fma<float, 8>(a_ik, b_k_vecs[bj], c_regs[i][bj]);
            }
        }
    }

    // Update C matrix using accumulated panel result (c_regs) and effective beta
    float8 alpha_vec(alpha);
    float8 beta_vec(effective_beta);

    for (int i = 0; i < MR; ++i) {
        for (int bj = 0; bj < num_b_vectors; ++bj) {
            T* c_ptr = C + i * ldc + bj * simd_width;
            simd::scale_accumulate_store<T>(
                c_ptr,
                c_regs[i][bj], // The accumulated A*B result for this block
                alpha_vec,
                beta_vec       // Use the effective beta passed in
            );
        }
    }
}

// --- Scalar Kernel for Edges/Partial Tiles (AccT = float) ---
template <typename T> // T = float16 or bfloat16
static void compute_block_scalar_partial(
    const float* A_panel, // Packed A (MC x KC, col-major)
    const float* B_panel, // Packed B (KC x NC, row-major)
    T* C,                 // Output C (row major)
    int ldc,              // Leading dimension of C
    int m_micro, int n_micro, int k_block, // Actual dimensions of this partial block
    int MC_pack_stride,   // Packing stride for A
    int NC_pack_stride,   // Packing stride for B (row length)
    float alpha, float effective_beta) // Effective alpha/beta for this pass
{
    for (int i = 0; i < m_micro; ++i) {
        for (int j = 0; j < n_micro; ++j) {
            // Accumulate A*B for C[i,j] in float
            float acc = 0.0f;
            for (int k = 0; k < k_block; ++k) {
                 // Access packed A (col-major): A[i + k*MC_pack_stride]
                 float a_ik = A_panel[i + k * MC_pack_stride];
                 // Access packed B (row-major): B[k*NC_pack_stride + j]
                 float b_kj = B_panel[k * NC_pack_stride + j];
                 // Use fmaf for potentially better scalar precision? Or simple mul+add.
                 // acc = std::fmaf(a_ik, b_kj, acc);
                 acc += a_ik * b_kj;
            }

            // Apply alpha and effective_beta
            T* c_ptr = C + i * ldc + j;
            float result_fp32 = alpha * acc; // Calculate alpha * (A*B) part

            if (effective_beta != 0.0f) {
                 float c_old_fp32 = static_cast<float>(*c_ptr); // Load C_old (T) -> float
                 result_fp32 = std::fmaf(effective_beta, c_old_fp32, result_fp32); // Use fmaf
            }

            // Store result (float -> T, relies on T's conversion operator for rounding/saturation)
            *c_ptr = static_cast<T>(result_fp32);
        }
    }
}

/**
 * Optimized single-threaded matrix multiplication using AVX/FMA (FP32 Accumulator)
 * For T = float16_t or bfloat16_t inputs/outputs.
 */
template <typename T> // T = float16_t or bfloat16_t
void simd_gemm_optimized_fp32acc(
    const T* a, const T* b, T* c,
    bool a_trans, bool b_trans,
    int M, int N, int K,
    int ldA, int ldB, int ldC, // Leading dimensions
    float alpha, float beta)   // User-provided alpha/beta
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                  "GEMM kernel requires float16_t or bfloat16_t.");

    using AccT = float; // Accumulator type is float

    // --- Blocking Parameters (Tune these!) ---
    constexpr int MR = 6;
    constexpr int NR = 16; // Needs to be multiple of 8 for float8
    static_assert(NR % 8 == 0, "NR must be multiple of float SIMD width (8)");

    // Example Cache-level blocking parameters (adjust based on CPU cache sizes)
    // Target L1d: ~32-64 KiB, L2: ~256-1024 KiB
    // Try to fit B panel (KC*NC*4) in L1/L2, A panel (MC*KC*4) in L2/L3
    constexpr int KC_BLOCK = 256;  // (tune: 128-512) Affects L1/L2 usage for B, register pressure if too small
    constexpr int MC_BLOCK = 72;   // Multiple of MR=6 (tune: ~48-144) Affects L2/L3 usage for A
    constexpr int NC_BLOCK = 512;  // Multiple of NR=16 (tune: ~256-1024+) Affects L1/L2/L3 usage for B & C

    static_assert(MC_BLOCK % MR == 0, "MC_BLOCK must be a multiple of MR");
    static_assert(NC_BLOCK % NR == 0, "NC_BLOCK must be a multiple of NR");

    // Packed Buffer Allocation (consider std::vector<float, aligned_allocator<float, 32>> potentially)
    std::vector<AccT> A_packed_vec(MC_BLOCK * KC_BLOCK); // Stores float
    std::vector<AccT> B_packed_vec(KC_BLOCK * NC_BLOCK); // Stores float

    // --- Main Loop Structure (jc -> pc -> ic -> ir -> jr) ---
    for (int jc = 0; jc < N; jc += NC_BLOCK) {
        int nc = std::min(NC_BLOCK, N - jc);

        for (int pc = 0; pc < K; pc += KC_BLOCK) {
            int kc = std::min(KC_BLOCK, K - pc);

            // *** K-Blocking Beta Calculation ***
            float effective_beta = (pc == 0) ? beta : 1.0f;

            // Pack B Panel (T -> float, KC x NC row-major)
            pack_B_block<T, KC_BLOCK, NC_BLOCK>(
                b, B_packed_vec.data(), K, N, ldB, pc, jc, kc, nc, b_trans);

            for (int ic = 0; ic < M; ic += MC_BLOCK) {
                int mc = std::min(MC_BLOCK, M - ic);

                // Pack A Panel (T -> float, MC x KC col-major)
                pack_A_block<T, MC_BLOCK, KC_BLOCK>(
                    a, A_packed_vec.data(), M, K, ldA, ic, pc, mc, kc, a_trans);

                // Micro-Kernel Execution
                for (int ir = 0; ir < mc; ir += MR) {
                    int m_micro = std::min(MR, mc - ir);

                    for (int jr = 0; jr < nc; jr += NR) {
                        int n_micro = std::min(NR, nc - jr);

                        // Pointers to packed data for the micro-kernel block
                        // A_packed is MC_BLOCK x KC_BLOCK (col-major). Kernel starts at row `ir`.
                        const float* a_kernel_ptr = A_packed_vec.data() + ir;
                        // B_packed is KC_BLOCK x NC_BLOCK (row-major). Kernel starts at col `jr`.
                        const float* b_kernel_ptr = B_packed_vec.data() + jr;
                        // Output C block pointer
                        T* c_sub = c + (ic + ir) * ldC + (jc + jr);

                        // Choose Kernel
                        if (m_micro == MR && n_micro == NR) {
                            compute_fma_microkernel<MR, NR, T>(
                                a_kernel_ptr,   // Start row `ir` in packed A
                                b_kernel_ptr,   // Start col `jr` in packed B
                                c_sub,
                                ldC,            // C stride
                                kc,             // Current K dimension
                                MC_BLOCK,       // Column stride in packed A
                                NC_BLOCK,       // Row length in packed B
                                alpha,
                                effective_beta
                            );
                        } else {
                            // Partial tile - use scalar kernel
                            compute_block_scalar_partial<T>(
                                a_kernel_ptr,
                                b_kernel_ptr,
                                c_sub,
                                ldC,
                                m_micro, n_micro, kc,
                                MC_BLOCK,       // Column stride in packed A
                                NC_BLOCK,       // Row length in packed B
                                alpha,
                                effective_beta
                             );
                        }
                    } // End loop jr (NR blocks)
                } // End loop ir (MR blocks)
            } // End loop ic (MC blocks)
        } // End loop pc (KC blocks)
    } // End loop jc (NC blocks)
}


/**
 * Public interface wrapper for SIMD GEMM (float16/bfloat16 specific)
 * Uses float accumulator internally.
 */
template <typename T> // T = float16_t or bfloat16_t
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    int ldA, // Leading dimension of A
    int ldB, // Leading dimension of B
    int ldC, // Leading dimension of C
    float alpha = 1.0f,
    float beta = 0.0f) {

    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                  "simd_gemm interface called with unsupported type T.");

    // Handle NOP cases
    if (M <= 0 || N <= 0) { return; }

    // Handle K=0: C = beta * C
    if (K <= 0) {
        if (beta == 0.0f) {
            for (int i = 0; i < M; ++i) {
                 T zero_val = static_cast<T>(0.0f); // Get correctly typed zero
                 std::fill(c + i * ldC, c + i * ldC + N, zero_val);
            }
        } else if (beta != 1.0f) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    // Apply beta scaling: C[i,j] = beta * C[i,j]
                    // Perform calculation in float for potentially better precision before converting back
                    float c_old_f = static_cast<float>(c[i * ldC + j]);
                    c[i * ldC + j] = static_cast<T>(beta * c_old_f);
                }
            }
        } // If beta == 1.0f, C is unchanged, do nothing.
        return;
    }

    // Call the optimized implementation
    simd_gemm_optimized_fp32acc<T>(a, b, c, a_trans, b_trans, M, N, K, ldA, ldB, ldC, alpha, beta);
}

// Explicit Instantiation: Should be done in the .cpp file that *calls* this,
// e.g., in matmul.cpp, after the template definitions are visible.
// template void simd_gemm<float16_t>(...);
// template void simd_gemm<bfloat16_t>(...);

} // namespace mlx::core