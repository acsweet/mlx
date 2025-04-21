// Copyright © 2025 Apple Inc.
#pragma once // Remove if this is a pure .cpp file

#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept> // For invalid_argument
#include <limits>    // For numeric_limits
#include <cstring>   // For std::fill needed in packing
#include <immintrin.h> // Needed for _mm_prefetch

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
    constexpr int simd_width = 8; // AVX float width

    // Zero the destination buffer (important for padding)
    std::fill(A_packed, A_packed + MC * KC, 0.0f);

    if (!a_trans) { // A is M x K, row-major (ldA >= K), accessed as M x K
                    // Packing into A_packed (MC x KC), column-major
        for (int k = 0; k < k_block; ++k) {
            const T* a_src_col_start = A + M_offset * ldA + (K_offset + k); // Start of source column slice A[M_offset, K_offset+k]
            float* a_dst_col_ptr = A_packed + k * MC; // Destination column k in packed buffer
            int i = 0;

            // *** SIMD Optimization Attempt (Load is strided, Store is contiguous) ***
            // This is harder to vectorize efficiently due to strided loads.
            // We load scalar, convert, buffer, then store vector.
            for (; i + simd_width <= m_block; i += simd_width) {
                // Load 8 strided elements scalar -> temporary float buffer
                alignas(32) float tmp_buf[simd_width];
                for(int ii=0; ii < simd_width; ++ii) {
                    tmp_buf[ii] = static_cast<float>(*(a_src_col_start + (i + ii) * ldA));
                }
                // Load float vector from temp buffer
                simd::float8 a_vec = simd::load<float, simd_width>(tmp_buf);
                // Store contiguous block in packed A column
                simd::store<float, simd_width>(a_dst_col_ptr + i, a_vec);
            }

            // Handle remaining elements scalar
            for (; i < m_block; ++i) {
                a_dst_col_ptr[i] = static_cast<float>(*(a_src_col_start + i * ldA));
            }
        }
    } else { // A is K x M, row-major (ldA >= M), accessed as transposed M x K
             // Packing into A_packed (MC x KC), column-major
        for (int k = 0; k < k_block; ++k) {
            // Source data is contiguous within a row of A (which corresponds to a column of A^T)
            const T* a_src_row_ptr = A + (K_offset + k) * ldA + M_offset; // Ptr to A[K_offset+k, M_offset]
            float* a_dst_col_ptr = A_packed + k * MC; // Destination column k
            int i = 0;

            // *** SIMD Optimization (Load is contiguous, Store is contiguous) ***
            for (; i + simd_width <= m_block; i += simd_width) {
                // Load 8 contiguous T elements, convert to float8
                simd::float8 a_vec = simd::load_convert_to_float<T>(a_src_row_ptr + i);
                // Store 8 contiguous float elements
                simd::store<float, simd_width>(a_dst_col_ptr + i, a_vec);
            }
            // Handle remaining elements scalar
            for (; i < m_block; ++i) {
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
    constexpr int simd_width = 8; // AVX float width

    // Zero the destination buffer
    std::fill(B_packed, B_packed + KC * NC, 0.0f);

    if (!b_trans) { // B is K x N, row-major (ldB >= N)
                    // Packing into B_packed (KC x NC), row-major
        for (int k = 0; k < k_block; ++k) {
            // Source data is contiguous within a row of B
            const T* b_src_row_ptr = B + (K_offset + k) * ldB + N_offset; // Ptr to B[K_offset+k, N_offset]
            float* b_dst_row_ptr = B_packed + k * NC; // Packed row k (stride NC)
            int j = 0;

            // *** SIMD Optimization (Load is contiguous, Store is contiguous) ***
            for (; j + simd_width <= n_block; j += simd_width) {
                // Load 8 contiguous T elements, convert to float8
                simd::float8 b_vec = simd::load_convert_to_float<T>(b_src_row_ptr + j);
                // Store 8 contiguous float elements
                simd::store<float, simd_width>(b_dst_row_ptr + j, b_vec);
            }
            // Handle remaining elements scalar
            for (; j < n_block; ++j) {
                 b_dst_row_ptr[j] = static_cast<float>(b_src_row_ptr[j]);
            }
        }
    } else { // B is N x K, row-major (ldB >= K), accessed as transposed K x N
             // Packing into B_packed (KC x NC), row-major
        for (int k = 0; k < k_block; ++k) {
             float* b_dst_row_ptr = B_packed + k * NC; // Packed row k
             int j = 0;

             // *** SIMD Optimization Attempt (Load is strided, Store is contiguous) ***
             // Similar issue to non-transposed A packing. Load is the challenge.
             for (; j + simd_width <= n_block; j += simd_width) {
                 // Load 8 strided elements scalar -> temporary float buffer
                 alignas(32) float tmp_buf[simd_width];
                 for(int jj=0; jj < simd_width; ++jj) {
                    // Read B[N_offset+j+jj, K_offset+k]
                    const T* b_src_elem_ptr = B + (N_offset + j + jj) * ldB + (K_offset + k);
                    tmp_buf[jj] = static_cast<float>(*b_src_elem_ptr);
                 }
                 // Load float vector from temp buffer
                 simd::float8 b_vec = simd::load<float, simd_width>(tmp_buf);
                 // Store 8 contiguous float elements
                 simd::store<float, simd_width>(b_dst_row_ptr + j, b_vec);
             }

             // Handle remaining elements scalar
             for (; j < n_block; ++j) {
                 const T* b_src_elem_ptr = B + (N_offset + j) * ldB + (K_offset + k);
                 b_dst_row_ptr[j] = static_cast<float>(*b_src_elem_ptr);
             }
        }
    }
}

/**
 * Optimized single-threaded matrix multiplication using AVX/FMA with full float32 accumulation
 * For T = float16_t or bfloat16_t inputs/outputs.
 */
template <typename T> // T = float16_t or bfloat16_t
void simd_gemm_optimized_higher_precision(
    const T* a, const T* b, T* c,
    bool a_trans, bool b_trans,
    int M, int N, int K,
    int ldA, int ldB, int ldC, // Leading dimensions
    float alpha, float beta)   // User-provided alpha/beta
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                  "GEMM kernel requires float16_t or bfloat16_t.");

    // // --- Blocking Parameters (Tune these!) ---
    // constexpr int MR = 4;
    // constexpr int NR = 24; // Needs to be multiple of 8 for float8
    // static_assert(NR % 8 == 0, "NR must be multiple of float SIMD width (8)");

    // constexpr int KC_BLOCK = 384; // L2 cache
    // constexpr int MC_BLOCK = 120; // multiple of MR, fits in L1
    // constexpr int NC_BLOCK = 1920; // L3 cache (large?)

    // // Prefetch distances (tune these based on profiling!)
    // constexpr int PREFETCH_K_DIST = 8;  // How many k iterations ahead in microkernel
    // constexpr int PREFETCH_PACK_DIST = 1; // How many outer blocks ahead for packing

    // --- Blocking Parameters (Tune these!) ---
    constexpr int MR = 6;
    constexpr int NR = 16; // Needs to be multiple of 8 for float8
    static_assert(NR % 8 == 0, "NR must be multiple of float SIMD width (8)");

    constexpr int KC_BLOCK = 256; // L2 cache
    constexpr int MC_BLOCK = 48; // multiple of MR, fits in L1
    constexpr int NC_BLOCK = 256; // L3 cache (large?)

    // Prefetch distances (tune these based on profiling!)
    constexpr int PREFETCH_K_DIST = 2;  // How many k iterations ahead in microkernel
    constexpr int PREFETCH_PACK_DIST = 0; // How many outer blocks ahead for packing

    static_assert(MC_BLOCK % MR == 0, "MC_BLOCK must be a multiple of MR");
    static_assert(NC_BLOCK % NR == 0, "NC_BLOCK must be a multiple of NR");

    // --- Allocate full-sized float32 accumulator ---
    std::vector<float> C_acc(M * ldC, 0.0f);

    // Initialize C_acc with beta * C if beta != 0
    if (beta != 0.0f) {
        // SIMD-optimized initialization of C_acc with beta * C
        constexpr int simd_width = 8;
        simd::float8 beta_vec(beta);
        
        for (int i = 0; i < M; ++i) {
            int j = 0;
            // Process 8 elements at a time with SIMD
            for (; j + simd_width <= N; j += simd_width) {
                T* c_ptr = c + i * ldC + j;
                float* acc_ptr = C_acc.data() + i * ldC + j;
                
                // Load and convert C (T) to float8
                simd::float8 c_old_vec = simd::load_convert_to_float<T>(c_ptr);
                
                // Multiply by beta and store to C_acc
                c_old_vec = beta_vec * c_old_vec;
                simd::store<float, 8>(acc_ptr, c_old_vec);
            }
            
            // Handle remaining elements
            for (; j < N; ++j) {
                C_acc[i * ldC + j] = beta * static_cast<float>(c[i * ldC + j]);
            }
        }
    }

    // --- Packed Buffer Allocation ---
    std::vector<float> A_packed_vec(MC_BLOCK * KC_BLOCK);
    std::vector<float> B_packed_vec(KC_BLOCK * NC_BLOCK);

    // --- Modified Microkernel Function (for full blocks) ---
    auto compute_fma_microkernel_float_acc = [&](
        const float* A_panel,
        const float* B_panel,
        float* C_acc_sub,
        int ldc_acc,
        int kc,
        int MC_pack_stride,
        int NC_pack_stride)
    {
        using float8 = simd::float8;
        constexpr int simd_width = 8;
        constexpr int num_b_vectors = NR / simd_width;

        float8 c_regs[MR][num_b_vectors]; // FP32 Accumulators for A*B panel result

        // Load existing accumulator values
        for (int i = 0; i < MR; ++i) {
            for (int bj = 0; bj < num_b_vectors; ++bj) {
                c_regs[i][bj] = simd::load<float, 8>(C_acc_sub + i * ldc_acc + bj * simd_width);
            }
        }

        // Accumulate A*B panel result into c_regs
        for (int k = 0; k < kc; ++k) {
            // *** Prefetch for Microkernel ***
            // Prefetch A and B data needed PREFETCH_K_DIST iterations later
            if (k + PREFETCH_K_DIST < kc) {
                // Prefetch next slice of B panel (row k + PREFETCH_K_DIST)
                 _mm_prefetch((const char*)(B_panel + (k + PREFETCH_K_DIST) * NC_pack_stride), _MM_HINT_T0);
                 // Prefetch next slice of A panel (column k + PREFETCH_K_DIST)
                 _mm_prefetch((const char*)(A_panel + (k + PREFETCH_K_DIST) * MC_pack_stride), _MM_HINT_T0);
                 // Optionally prefetch multiple rows of A for next k iteration
                 for (int pf_i = 0; pf_i < MR; pf_i += 4) { // Skip every 4 to reduce prefetch overhead
                    _mm_prefetch((const char*)(A_panel + pf_i + (k + PREFETCH_K_DIST) * MC_pack_stride), _MM_HINT_T0);
                 }
            }

            // B panel is row-major packed (KC x NC), stride is NC_pack_stride
            const float* b_row_k_ptr = B_panel + k * NC_pack_stride;
            float8 b_k_vecs[num_b_vectors];
            for (int bj = 0; bj < num_b_vectors; ++bj) {
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

        // Store accumulated results back to C_acc
        for (int i = 0; i < MR; ++i) {
            for (int bj = 0; bj < num_b_vectors; ++bj) {
                float* acc_ptr = C_acc_sub + i * ldc_acc + bj * simd_width;
                simd::store<float, 8>(acc_ptr, c_regs[i][bj]);
            }
        }
    };

    // --- Modified Scalar Kernel for Edges/Partial Tiles ---
    auto compute_block_scalar_partial_float_acc = [&](
        const float* A_panel,
        const float* B_panel,
        float* C_acc_sub,
        int ldc_acc,
        int m_micro, int n_micro, int k_block,
        int MC_pack_stride,
        int NC_pack_stride)
    {
        for (int i = 0; i < m_micro; ++i) {
            for (int j = 0; j < n_micro; ++j) {
                float* acc_ptr = C_acc_sub + i * ldc_acc + j;
                float acc = *acc_ptr; // Load existing accumulator value
                
                for (int k = 0; k < k_block; ++k) {
                    float a_ik = A_panel[i + k * MC_pack_stride];
                    float b_kj = B_panel[k * NC_pack_stride + j];
                    acc = std::fmaf(a_ik, b_kj, acc); // Use fmaf for better precision
                }
                
                *acc_ptr = acc; // Store back to accumulator
            }
        }
    };

    // --- Main Loop Structure (jc -> pc -> ic -> ir -> jr) ---
    for (int jc = 0; jc < N; jc += NC_BLOCK) {
        int nc = std::min(NC_BLOCK, N - jc);

        for (int pc = 0; pc < K; pc += KC_BLOCK) {
            int kc = std::min(KC_BLOCK, K - pc);

            // *** Prefetch B for the NEXT pc iteration ***
            if (pc + KC_BLOCK < K) {
                int next_pc = pc + KC_BLOCK;
                
                // Prefetch first row of the next B block
                if (!b_trans) {
                    const T* next_b_prefetch_addr = b + (next_pc) * ldB + jc;
                    _mm_prefetch((const char*)next_b_prefetch_addr, _MM_HINT_T1);
                    // Prefetch a few more positions along the row
                    if (nc > 64) {
                        _mm_prefetch((const char*)(next_b_prefetch_addr + 64), _MM_HINT_T1);
                    }
                } else {
                    // For transposed B, prefetch the first few columns
                    const T* next_b_prefetch_addr = b + jc * ldB + next_pc;
                    _mm_prefetch((const char*)next_b_prefetch_addr, _MM_HINT_T1);
                    // Prefetch a few more positions down the column
                    if (kc > 64) {
                        _mm_prefetch((const char*)(next_b_prefetch_addr + 64 * ldB), _MM_HINT_T1);
                    }
                }
            }

            // Pack B Panel (T -> float, KC x NC row-major)
            pack_B_block<T, KC_BLOCK, NC_BLOCK>(
                b, B_packed_vec.data(), K, N, ldB, pc, jc, kc, nc, b_trans);

            for (int ic = 0; ic < M; ic += MC_BLOCK) {
                int mc = std::min(MC_BLOCK, M - ic);

                // *** Prefetch A for the NEXT ic iteration ***
                if (ic + MC_BLOCK < M) {
                    int next_ic = ic + MC_BLOCK;
                     
                    // Prefetch first element of the next A block
                    if (!a_trans) {
                        const T* next_a_prefetch_addr = a + next_ic * ldA + pc;
                        _mm_prefetch((const char*)next_a_prefetch_addr, _MM_HINT_T1);
                        // Prefetch a few more positions along the row
                        if (K - pc > 64) {
                            _mm_prefetch((const char*)(next_a_prefetch_addr + 64), _MM_HINT_T1);
                        }
                    } else {
                        // For transposed A, prefetch the first few columns
                        const T* next_a_prefetch_addr = a + pc * ldA + next_ic;
                        _mm_prefetch((const char*)next_a_prefetch_addr, _MM_HINT_T1);
                        // Prefetch a few more positions down the column
                        if (mc > 64) {
                            _mm_prefetch((const char*)(next_a_prefetch_addr + 64 * ldA), _MM_HINT_T1);
                        }
                    }
                }

                // Pack A Panel (T -> float, MC x KC col-major)
                pack_A_block<T, MC_BLOCK, KC_BLOCK>(
                    a, A_packed_vec.data(), M, K, ldA, ic, pc, mc, kc, a_trans);

                // Micro-Kernel Execution
                for (int ir = 0; ir < mc; ir += MR) {
                    int m_micro = std::min(MR, mc - ir);

                    for (int jr = 0; jr < nc; jr += NR) {
                        int n_micro = std::min(NR, nc - jr);

                        // Pointers to packed data for the micro-kernel block
                        const float* a_kernel_ptr = A_packed_vec.data() + ir;
                        const float* b_kernel_ptr = B_packed_vec.data() + jr;
                        
                        // Pointer to C_acc submatrix
                        float* c_acc_sub = C_acc.data() + (ic + ir) * ldC + (jc + jr);

                        // Choose Kernel
                        if (m_micro == MR && n_micro == NR) {
                            compute_fma_microkernel_float_acc(
                                a_kernel_ptr,
                                b_kernel_ptr,
                                c_acc_sub,
                                ldC,
                                kc,
                                MC_BLOCK,
                                NC_BLOCK
                            );
                        } else {
                            // Partial tile - use scalar kernel
                            compute_block_scalar_partial_float_acc(
                                a_kernel_ptr,
                                b_kernel_ptr,
                                c_acc_sub,
                                ldC,
                                m_micro, n_micro, kc,
                                MC_BLOCK,
                                NC_BLOCK
                            );
                        }
                    } // End loop jr (NR blocks)
                } // End loop ir (MR blocks)
            } // End loop ic (MC blocks)
        } // End loop pc (KC blocks)
    } // End loop jc (NC blocks)

    // --- Final conversion from float accumulator to output type T ---
    // Apply alpha scaling and convert to T using SIMD
    constexpr int simd_width = 8;
    simd::float8 alpha_vec(alpha);
    
    for (int i = 0; i < M; ++i) {
        int j = 0;
        // Process 8 elements at a time using SIMD
        for (; j + simd_width <= N; j += simd_width) {
            float* acc_ptr = C_acc.data() + i * ldC + j;
            T* c_ptr = c + i * ldC + j;
            
            // Load accumulated values
            simd::float8 acc_vec = simd::load<float, 8>(acc_ptr);
            
            // Apply alpha
            acc_vec = alpha_vec * acc_vec;
            
            // Convert and store to output type T
            simd::store_convert_from_float<T>(c_ptr, acc_vec);
        }
        
        // Handle remaining elements
        for (; j < N; ++j) {
            float result = alpha * C_acc[i * ldC + j];
            c[i * ldC + j] = static_cast<T>(result);
        }
    }
}

// Update the public interface wrapper
template <typename T, typename AccT>
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    size_t M_s,
    size_t N_s,
    size_t K_s,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                  "simd_gemm interface requires T = float16_t or bfloat16_t.");
    static_assert(std::is_same_v<AccT, float>,
                  "simd_gemm interface requires AccT = float for float16/bfloat16.");

    // --- Dimension Conversion and Validation ---
    if (M_s > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N_s > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        K_s > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Matrix dimensions exceed integer limits in simd_gemm.");
    }
    int M = static_cast<int>(M_s);
    int N = static_cast<int>(N_s);
    int K = static_cast<int>(K_s);

    // Handle NOP cases
    if (M <= 0 || N <= 0) { return; }

    // --- Infer Leading Dimensions (Assuming Contiguous Row-Major within batch) ---
    int ldA = (!a_trans) ? K : M;
    int ldB = (!b_trans) ? N : K;
    int ldC = N;

    // Handle K=0 case (C = beta * C) using inferred ldC
    if (K <= 0) {
        if (beta == 0.0f) {
            for (int i = 0; i < M; ++i) {
                T zero_val = static_cast<T>(0.0f);
                std::fill(c + i * ldC, c + i * ldC + N, zero_val);
            }
        } else if (beta != 1.0f) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float c_old_f = static_cast<float>(c[i * ldC + j]);
                    c[i * ldC + j] = static_cast<T>(beta * c_old_f);
                }
            }
        }
        return;
    }

    // --- Call the higher precision implementation ---
    simd_gemm_optimized_higher_precision<T>(
        a, b, c,
        a_trans, b_trans,
        M, N, K,
        ldA, ldB, ldC,
        alpha, beta);
}

} // namespace mlx::core