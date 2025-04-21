// Copyright © 2025 Apple Inc.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h> // For _mm_prefetch only (AVX intrinsics should be in avx_simd_matmul.h)
#include <limits>
#include <memory> // For aligned_alloc
#include <stdexcept>
#include <type_traits>

// Use the SIMD header
#include "mlx/backend/cpu/simd/avx_simd_matmul.h"

namespace mlx::core {

// Helper for integer division rounded up
inline int ceildiv(int a, int b) {
    if (b == 0) throw std::invalid_argument("Division by zero in ceildiv");
    if (a <= 0) return 0; // Handle non-positive a
    return (a + b - 1) / b;
}

// Aligned memory allocation helper
template <typename T>
class aligned_unique_ptr {
private:
    T* ptr_;
    size_t size_;

public:
    aligned_unique_ptr() : ptr_(nullptr), size_(0) {}
    
    explicit aligned_unique_ptr(size_t size) : size_(size) {
        // Allocate with 32-byte alignment for AVX
        ptr_ = static_cast<T*>(aligned_alloc(32, size * sizeof(T)));
        if (!ptr_) throw std::bad_alloc();
    }
    
    ~aligned_unique_ptr() {
        if (ptr_) free(ptr_);
    }
    
    // Move semantics
    aligned_unique_ptr(aligned_unique_ptr&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    aligned_unique_ptr& operator=(aligned_unique_ptr&& other) noexcept {
        if (this != &other) {
            if (ptr_) free(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disallow copying
    aligned_unique_ptr(const aligned_unique_ptr&) = delete;
    aligned_unique_ptr& operator=(const aligned_unique_ptr&) = delete;
    
    // Access
    T* get() const { return ptr_; }
    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }
    
    // Reset with new size if needed
    void reset(size_t new_size) {
        if (new_size > size_) {
            if (ptr_) free(ptr_);
            ptr_ = static_cast<T*>(aligned_alloc(32, new_size * sizeof(T)));
            if (!ptr_) throw std::bad_alloc();
            size_ = new_size;
        }
    }
};

// Use the optimized transpose function from simd header
// This moves AVX-specific code to the appropriate file
template <typename T>
inline void pack_transpose_8x8(const T* src, float* dst, int src_stride, int dst_stride) {
    simd::transpose_8x8_block<T>(src, dst, src_stride, dst_stride);
}

// --- Optimized Packing Functions (T -> float) ---
// Pack A block (m_block x k_block) from T into A_packed (MC x KC float), column-major layout
template <typename T, int MC, int KC>
static void pack_A_block(
    const T* A, float* A_packed,
    int M, int K, int ldA,
    int M_offset, int K_offset,
    int m_block, int k_block, bool a_trans)
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>, "T must be float16 or bfloat16");
    constexpr int simd_width = 8; // AVX float width

    // ⭐⭐⭐ OPTIMIZATION: Only zero-fill the edge cases, not the entire buffer
    // Only zero the parts we'll actually access
    if (m_block < MC || k_block < KC) {
        // Only zero needed portions
        for (int k = 0; k < k_block; ++k) {
            std::fill(A_packed + k * MC, A_packed + k * MC + m_block, 0.0f);
        }
    }

    if (!a_trans) { // A is M x K, row-major (ldA >= K), accessed as M x K
                    // Packing into A_packed (MC x KC), column-major
        for (int k = 0; k < k_block; k += 8) {
            int k_chunk = std::min(8, k_block - k);
            
            if (k_chunk == 8) {
                // Full SIMD width for k dimension
                for (int i = 0; i < m_block; i += 8) {
                    int m_chunk = std::min(8, m_block - i);
                    
                    if (m_chunk == 8) {
                        // Process 8x8 block with full SIMD - use the fast transpose function
                        // This avoids the strided load issue
                        const T* a_block_start = A + (M_offset + i) * ldA + K_offset + k;
                        // ⭐⭐⭐ FIX: Pass the destination stride (MC) to the transpose function
                        pack_transpose_8x8<T>(a_block_start, A_packed + k * MC + i, ldA, MC);
                    } else {
                        // Handle partial m_chunk < 8
                        for (int ii = 0; ii < m_chunk; ++ii) {
                            const T* a_src_row_ptr = A + (M_offset + i + ii) * ldA + K_offset + k;
                            for (int kk = 0; kk < k_chunk; ++kk) {
                                A_packed[(k + kk) * MC + (i + ii)] = static_cast<float>(a_src_row_ptr[kk]);
                            }
                        }
                    }
                }
            } else {
                // Handle partial k_chunk < 8
                for (int i = 0; i < m_block; ++i) {
                    const T* a_src_row_ptr = A + (M_offset + i) * ldA + K_offset + k;
                    for (int kk = 0; kk < k_chunk; ++kk) {
                        A_packed[(k + kk) * MC + i] = static_cast<float>(a_src_row_ptr[kk]);
                    }
                }
            }
        }
    } else { // A is K x M, row-major (ldA >= M), accessed as transposed M x K
             // Packing into A_packed (MC x KC), column-major
        for (int k = 0; k < k_block; ++k) {
            // Source data is contiguous within a row of A (which corresponds to a column of A^T)
            const T* a_src_row_ptr = A + (K_offset + k) * ldA + M_offset;
            float* a_dst_col_ptr = A_packed + k * MC;
            int i = 0;

            // Process 8 elements at a time
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
    const T* B, float* B_packed,
    int K, int N, int ldB,
    int K_offset, int N_offset,
    int k_block, int n_block, bool b_trans)
{
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>, "T must be float16 or bfloat16");
    constexpr int simd_width = 8; // AVX float width

    // ⭐⭐⭐ OPTIMIZATION: Only zero-fill the edge cases, not the entire buffer
    if (k_block < KC || n_block < NC) {
        // Only zero needed portions
        for (int k = 0; k < k_block; ++k) {
            std::fill(B_packed + k * NC, B_packed + k * NC + n_block, 0.0f);
        }
    }

    if (!b_trans) { // B is K x N, row-major (ldB >= N)
                    // Packing into B_packed (KC x NC), row-major
        for (int k = 0; k < k_block; ++k) {
            // Source data is contiguous within a row of B
            const T* b_src_row_ptr = B + (K_offset + k) * ldB + N_offset;
            float* b_dst_row_ptr = B_packed + k * NC;
            int j = 0;

            // Process 8 elements at a time
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
        for (int k = 0; k < k_block; k += 8) {
            int k_chunk = std::min(8, k_block - k);
            
            if (k_chunk == 8) {
                // Full SIMD width for k dimension
                for (int j = 0; j < n_block; j += 8) {
                    int n_chunk = std::min(8, n_block - j);
                    
                    if (n_chunk == 8) {
                        // Process 8x8 block with fast transpose function
                        const T* b_block_start = B + (N_offset + j) * ldB + K_offset + k;
                        
                        // Need to transpose differently for B since output is row-major
                        float tmp_transpose[64];
                        // ⭐⭐⭐ FIX: Pass the destination stride for the B block transpose
                        pack_transpose_8x8<T>(b_block_start, tmp_transpose, ldB, 8);
                        
                        // Copy transposed block to B_packed with proper stride
                        for (int kk = 0; kk < 8; ++kk) {
                            for (int jj = 0; jj < 8; ++jj) {
                                B_packed[(k + kk) * NC + (j + jj)] = tmp_transpose[kk * 8 + jj];
                            }
                        }
                    } else {
                        // Handle partial n_chunk < 8
                        for (int kk = 0; kk < k_chunk; ++kk) {
                            float* b_dst_row_ptr = B_packed + (k + kk) * NC + j;
                            for (int jj = 0; jj < n_chunk; ++jj) {
                                b_dst_row_ptr[jj] = static_cast<float>(B[(N_offset + j + jj) * ldB + (K_offset + k + kk)]);
                            }
                        }
                    }
                }
            } else {
                // Handle partial k_chunk < 8
                for (int kk = 0; kk < k_chunk; ++kk) {
                    float* b_dst_row_ptr = B_packed + (k + kk) * NC;
                    for (int j = 0; j < n_block; ++j) {
                        b_dst_row_ptr[j] = static_cast<float>(B[(N_offset + j) * ldB + (K_offset + k + kk)]);
                    }
                }
            }
        }
    }
}

// // ⭐⭐ OPTIMIZATION: Unrolled microkernel with k-loop unrolling by 4
// template <int MR, int NR, int UNROLL_K = 4>
// void compute_fma_microkernel_optimized(
//     const float* A_panel,
//     const float* B_panel,
//     float* C_block, 
//     int ldc,
//     int kc,
//     int MC_pack_stride,
//     int NC_pack_stride)
// {
//     using float8 = simd::float8;
//     constexpr int simd_width = 8;
//     constexpr int num_b_vectors = NR / simd_width;
    
//     // Keep accumulators in registers
//     float8 c_regs[MR][num_b_vectors];
    
//     // Load existing C values
//     for (int i = 0; i < MR; ++i) {
//         for (int bj = 0; bj < num_b_vectors; ++bj) {
//             c_regs[i][bj] = simd::load<float, 8>(C_block + i * ldc + bj * simd_width);
//         }
//     }
    
//     // Main k-loop with unrolling
//     int k = 0;
//     for (; k + UNROLL_K <= kc; k += UNROLL_K) {
//         // Prefetch
//         _mm_prefetch((const char*)(A_panel + (k + UNROLL_K) * MC_pack_stride), _MM_HINT_T0);
//         _mm_prefetch((const char*)(B_panel + (k + UNROLL_K) * NC_pack_stride), _MM_HINT_T0);
        
//         // Load B vectors for multiple k iterations
//         float8 b_vecs[UNROLL_K][num_b_vectors];
        
//         // Preload all B vectors for the unrolled iterations
//         for (int ku = 0; ku < UNROLL_K; ++ku) {
//             const float* b_row_ptr = B_panel + (k + ku) * NC_pack_stride;
//             for (int bj = 0; bj < num_b_vectors; ++bj) {
//                 b_vecs[ku][bj] = simd::load<float, 8>(b_row_ptr + bj * simd_width);
//             }
//         }
        
//         // Process each microkernel row
//         for (int i = 0; i < MR; ++i) {
//             // Load A scalars for multiple k iterations
//             float8 a_scalars[UNROLL_K];
//             for (int ku = 0; ku < UNROLL_K; ++ku) {
//                 a_scalars[ku] = simd::broadcast<float, 8>(A_panel + (k + ku) * MC_pack_stride + i);
//             }
            
//             // Unrolled FMA operations for each B vector
//             for (int bj = 0; bj < num_b_vectors; ++bj) {
//                 for (int ku = 0; ku < UNROLL_K; ++ku) {
//                     c_regs[i][bj] = simd::fma<float, 8>(a_scalars[ku], b_vecs[ku][bj], c_regs[i][bj]);
//                 }
//             }
//         }
//     }
    
//     // Handle remaining k iterations
//     for (; k < kc; ++k) {
//         const float* b_row_k_ptr = B_panel + k * NC_pack_stride;
//         float8 b_k_vecs[num_b_vectors];
        
//         for (int bj = 0; bj < num_b_vectors; ++bj) {
//             b_k_vecs[bj] = simd::load<float, 8>(b_row_k_ptr + bj * simd_width);
//         }
        
//         const float* a_col_k_ptr = A_panel + k * MC_pack_stride;
//         for (int i = 0; i < MR; ++i) {
//             float8 a_ik = simd::broadcast<float, 8>(a_col_k_ptr + i);
//             for (int bj = 0; bj < num_b_vectors; ++bj) {
//                 c_regs[i][bj] = simd::fma<float, 8>(a_ik, b_k_vecs[bj], c_regs[i][bj]);
//             }
//         }
//     }
    
//     // Store accumulated results back to C
//     for (int i = 0; i < MR; ++i) {
//         for (int bj = 0; bj < num_b_vectors; ++bj) {
//             float* acc_ptr = C_block + i * ldc + bj * simd_width;
//             simd::store<float, 8>(acc_ptr, c_regs[i][bj]);
//         }
//     }
// }

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

    // --- Blocking Parameters ---
    // constexpr int MR = 6;
    constexpr int MR = 6; 
    constexpr int NR = 16; // Needs to be multiple of 8 for float8
    static_assert(NR % 8 == 0, "NR must be multiple of float SIMD width (8)");

    constexpr int KC_BLOCK = 64; // L2 cache
    constexpr int MC_BLOCK = 72;  // multiple of MR, fits in L1
    constexpr int NC_BLOCK = 256; // L3 cache

    // constexpr int PREFETCH_K_DIST = 2;
    // constexpr int K_UNROLL = 4;   // Unroll the k-loop in microkernel

    static_assert(MC_BLOCK % MR == 0, "MC_BLOCK must be a multiple of MR");
    static_assert(NC_BLOCK % NR == 0, "NC_BLOCK must be a multiple of NR");

    // ⭐ OPTIMIZATION: Use aligned memory allocation, allocated once
    // Thread-local static buffers to avoid reallocation
    thread_local aligned_unique_ptr<float> A_packed_buf(MC_BLOCK * KC_BLOCK);
    thread_local aligned_unique_ptr<float> B_packed_buf(KC_BLOCK * NC_BLOCK);
    thread_local aligned_unique_ptr<float> C_acc_buf(M * ldC);
    
    // Ensure buffers are large enough (should rarely need resizing)
    A_packed_buf.reset(MC_BLOCK * KC_BLOCK);
    B_packed_buf.reset(KC_BLOCK * NC_BLOCK);
    C_acc_buf.reset(M * ldC);
    
    float* A_packed = A_packed_buf.get();
    float* B_packed = B_packed_buf.get();
    float* C_acc = C_acc_buf.get();

    // --- Initialize C_acc with beta * C if beta != 0 ---
    // ⭐⭐⭐ OPTIMIZATION: Convert from half to float just once
    // ⭐⭐⭐ OPTIMIZATION: Convert while streaming - convert at the beginning and only once
    if (beta != 0.0f) {
        constexpr int simd_width = 8;
        simd::float8 beta_vec(beta);
        
        for (int i = 0; i < M; ++i) {
            int j = 0;
            // Process 8 elements at a time with SIMD
            for (; j + simd_width <= N; j += simd_width) {
                T* c_ptr = c + i * ldC + j;
                float* acc_ptr = C_acc + i * ldC + j;
                
                // Load and convert C (T) to float8 - we do this conversion just once
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
    } else {
        // If beta is 0, just zero out C_acc
        std::memset(C_acc, 0, M * ldC * sizeof(float));
    }

    // --- Modified Scalar Kernel for Edges/Partial Tiles ---
    auto compute_block_scalar_partial = [&](
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
                    // Use AVX FMA directly instead of std::fmaf for better performance
                    acc = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a_ik), _mm_set_ss(b_kj), _mm_set_ss(acc)));
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

            // ⭐⭐ OPTIMIZATION: Pack B block once for this panel
            pack_B_block<T, KC_BLOCK, NC_BLOCK>(
                b, B_packed, K, N, ldB, pc, jc, kc, nc, b_trans);

            for (int ic = 0; ic < M; ic += MC_BLOCK) {
                int mc = std::min(MC_BLOCK, M - ic);

                // Pack A Panel (T -> float, MC x KC col-major)
                pack_A_block<T, MC_BLOCK, KC_BLOCK>(
                    a, A_packed, M, K, ldA, ic, pc, mc, kc, a_trans);

                // ⭐⭐ OPTIMIZATION: Process micro-kernels more efficiently
                for (int ir = 0; ir < mc; ir += MR) {
                    int m_micro = std::min(MR, mc - ir);

                    for (int jr = 0; jr < nc; jr += NR) {
                        int n_micro = std::min(NR, nc - jr);

                        // Pointers to packed data for the micro-kernel block
                        const float* a_kernel_ptr = A_packed + ir;
                        const float* b_kernel_ptr = B_packed + jr;
                        
                        // Pointer to C_acc submatrix
                        float* c_acc_sub = C_acc + (ic + ir) * ldC + (jc + jr);

                        // Choose Kernel
                        // if (m_micro == 8 && n_micro == 16) {      // hot path
                        //     simd::micro_kernel_8x16(
                        //         a_kernel_ptr,
                        //         b_kernel_ptr,
                        //         c_acc_sub,
                        //         ldC,
                        //         kc,
                        //         MC_BLOCK,   // a_stride
                        //         NC_BLOCK);  // b_stride
                        if (m_micro == 6 && n_micro == 16) {      // hot path
                                    simd::micro_kernel_6x16(
                                        a_kernel_ptr,
                                        b_kernel_ptr,
                                        c_acc_sub,
                                        ldC,
                                        kc,
                                        MC_BLOCK,   // a_stride
                                        NC_BLOCK);  // b_stride
                        } else {
                            // Partial tile - use scalar kernel
                            compute_block_scalar_partial(
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
            float* acc_ptr = C_acc + i * ldC + j;
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

// Public interface wrapper
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