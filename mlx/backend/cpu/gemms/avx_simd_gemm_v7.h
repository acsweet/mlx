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

/**
 * Optimized register-tiled microkernel (8x16)
 * - Keeps values in registers as much as possible
 * - Minimizes memory access
 * - Uses SIMD operations for maximum performance
 */
template <typename T, typename AccT>
static void register_tiled_microkernel_8x16(
    const T* A, const float* B, T* C,
    int lda, int ldb, int ldc,
    int k_size, float alpha, float beta) 
{
    static_assert(std::is_same_v<AccT, float>, "Register-tiled kernel requires float accumulator");
    constexpr int MR = 8;
    constexpr int NR = 16;
    constexpr int simd_width = simd::max_size<AccT>;
    constexpr int num_b_vectors = NR / simd_width;
    
    using Vec = simd::Simd<AccT, simd_width>;
    
    // Register allocation for accumulation
    Vec c_regs[MR][num_b_vectors];
    
    // Initialize accumulation registers to zero
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < num_b_vectors; j++) {
            c_regs[i][j] = Vec(0.0f);
        }
    }
    
    // Vector constants
    Vec alpha_vec(alpha);
    Vec beta_vec(beta);
    
    // Prefetch entire first iteration A and B panels
    for (int i = 0; i < MR; i++) {
        simd::prefetch_a(A + i * lda, 0, k_size);
    }
    
    for (int j = 0; j < num_b_vectors; j++) {
        simd::prefetch_b(B + j * simd_width, 0, k_size);
    }
    
    // Main computation loop with aggressive register reuse
    for (int k = 0; k < k_size; k++) {
        // Load B values (multiple vectors for NR elements)
        Vec b_vectors[num_b_vectors];
        for (int j = 0; j < num_b_vectors; j++) {
            // Prefetch next iterations B values to L1 cache
            if (k < k_size - 4) {
                simd::prefetch_b(B + (k + 4) * ldb + j * simd_width, 0);
            }
            b_vectors[j] = simd::load<AccT, simd_width>(B + k * ldb + j * simd_width);
        }
        
        // Process each row of A 
        for (int i = 0; i < MR; i++) {
            // Prefetch next iterations A values (only on first iteration of the inner loop)
            if (k < k_size - 4 && i == 0) {
                for (int pf = 0; pf < MR; pf++) {
                    simd::prefetch_a(A + pf * lda + k + 4, 0);
                }
            }
            
            // Load single A value and broadcast to vector
            // Handle different types with explicit conversion
            float a_val = static_cast<float>(A[i * lda + k]);
            Vec a_vec(a_val);  // Use constructor, not broadcast (same effect, fewer issues)
            
            // Accumulate a[i,k] * b[k,j] for all j using direct intrinsics
            for (int j = 0; j < num_b_vectors; j++) {
                #ifdef __AVX2__
                c_regs[i][j].value = _mm256_fmadd_ps(a_vec.value, b_vectors[j].value, c_regs[i][j].value);
                #else
                __m256 mul = _mm256_mul_ps(a_vec.value, b_vectors[j].value);
                c_regs[i][j].value = _mm256_add_ps(mul, c_regs[i][j].value);
                #endif
            }
        }
    }
    
    // Store results back to C
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < num_b_vectors; j++) {
            simd::scale_accumulate_store<T>(
                C + i * ldc + j * simd_width,
                c_regs[i][j], alpha_vec, beta_vec);
        }
    }
}

/**
 * Scalar edge case handler for partial tiles
 */
template <typename T, typename AccT>
static void register_tiled_scalar_edge(
    const T* A, const float* B, T* C,
    int lda, int ldb, int ldc,
    int m_micro, int n_micro, int k_size,
    float alpha, float beta)
{
    static_assert(std::is_same_v<AccT, float>, "Scalar edge handler requires float accumulator");
    
    // Allocate accumulation registers
    alignas(32) float c_values[8][16] = {0};
    
    // Main computation loop
    for (int k = 0; k < k_size; k++) {
        // Load A values for this k
        alignas(32) float a_vals[8];
        for (int i = 0; i < m_micro; i++) {
            a_vals[i] = static_cast<float>(A[i * lda + k]);
        }
        
        // Load B values for this k
        alignas(32) float b_vals[16];
        for (int j = 0; j < n_micro; j++) {
            b_vals[j] = B[k * ldb + j];
        }
        
        // Compute matrix multiplication for this k
        for (int i = 0; i < m_micro; i++) {
            for (int j = 0; j < n_micro; j++) {
                c_values[i][j] += a_vals[i] * b_vals[j];
            }
        }
    }
    
    // Store results with alpha/beta scaling
    for (int i = 0; i < m_micro; i++) {
        for (int j = 0; j < n_micro; j++) {
            float c_old = (beta == 0.0f) ? 0.0f : static_cast<float>(C[i * ldc + j]) * beta;
            float result = alpha * c_values[i][j] + c_old;
            C[i * ldc + j] = static_cast<T>(result);
        }
    }
}

/**
 * Cache-aware macro-kernel implementation
 * - Uses register tiling for inner loops
 * - Panel sizes tuned for L1/L2 cache
 */
template <typename T, typename AccT>
static void register_tiled_macro_kernel(
    const T* A, const float* B, T* C,
    int m_size, int n_size, int k_size,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    static_assert(std::is_same_v<AccT, float>, "Macro kernel requires float accumulator");
    
    // Tiling parameters
    constexpr int MR = 8;    // Register tile height
    constexpr int NR = 16;   // Register tile width
    
    // Loop over the matrix in register-sized tiles
    for (int i = 0; i < m_size; i += MR) {
        int m_remain = std::min(MR, m_size - i);
        
        for (int j = 0; j < n_size; j += NR) {
            int n_remain = std::min(NR, n_size - j);
            
            const T* a_panel = A + i * lda;
            const float* b_panel = B + j;
            T* c_block = C + i * ldc + j;
            
            if (m_remain == MR && n_remain == NR) {
                // Full tile case - use optimized microkernel
                register_tiled_microkernel_8x16<T, AccT>(
                    a_panel, b_panel, c_block,
                    lda, ldb, ldc, k_size, alpha, beta);
            } else {
                // Partial tile case - use scalar edge handler
                register_tiled_scalar_edge<T, AccT>(
                    a_panel, b_panel, c_block,
                    lda, ldb, ldc, m_remain, n_remain, k_size,
                    alpha, beta);
            }
        }
    }
}

/**
 * Optimized cache-blocking implementation with register tiling
 * - Uses three levels of blocking for memory hierarchy
 * - Vectorized conversions for maximum performance
 */
template <typename T, typename AccT>
void register_tiled_gemm_optimized(
    const T* a, const T* b, T* c,
    bool a_trans, bool b_trans,
    int M, int N, int K, float alpha, float beta)
{
    static_assert(std::is_same_v<AccT, float>, "Optimized GEMM requires float accumulator");

    // Cache blocking parameters 
    constexpr int MR = 8;    // Register tile height
    constexpr int NR = 16;   // Register tile width
    constexpr int MC = 64;   // L1 cache block height 
    constexpr int NC = 256;  // L1 cache block width
    constexpr int KC = 256;  // L2 cache block depth
    
    // Allocate B packing buffer with alignment for SIMD operations
    alignas(64) float B_packed[KC * NC];
    
    // Create loops for cache blocking
    for (int k_outer = 0; k_outer < K; k_outer += KC) {
        int k_block = std::min(KC, K - k_outer);
        
        for (int j_outer = 0; j_outer < N; j_outer += NC) {
            int n_block = std::min(NC, N - j_outer);
            
            // Pack B panel for L1 cache efficiency using vectorized conversion
            if (b_trans) {
                // Transposed packing with improved memory access pattern for B
                // Group by panels of 16 columns for better cache behavior
                for (int jp = 0; jp < n_block; jp += 16) {
                    int j_panel = std::min(16, n_block - jp);
                    
                    for (int k = 0; k < k_block; k++) {
                        // Direct vectorized conversion for better performance
                        alignas(16) float buffer[16] = {0};
                        
                        for (int j = 0; j < j_panel; j++) {
                            buffer[j] = static_cast<float>(b[(j_outer + jp + j) * K + (k_outer + k)]);
                        }
                        
                        // Copy to packed storage
                        std::memcpy(&B_packed[k * NC + jp], buffer, j_panel * sizeof(float));
                    }
                }
            } else {
                // Non-transposed packing with improved memory access pattern
                for (int k = 0; k < k_block; k++) {
                    // Vectorized conversion using SIMD
                    simd::batch_convert_to_float<T>(
                        b + (k_outer + k) * N + j_outer,
                        &B_packed[k * NC],
                        n_block);
                    
                    // Zero-pad remainder
                    for (int j = n_block; j < NC; j++) {
                        B_packed[k * NC + j] = 0.0f;
                    }
                }
            }
            
            // Process A in vertical strips
            for (int i_outer = 0; i_outer < M; i_outer += MC) {
                int m_block = std::min(MC, M - i_outer);
                
                // Create a pointer to the start of the current A block
                const T* a_block = a_trans ? 
                    a + k_outer * M + i_outer : 
                    a + i_outer * K + k_outer;
                    
                // Create a pointer to the current C block
                T* c_block = c + i_outer * N + j_outer;
                
                // Compute using register tiling
                register_tiled_macro_kernel<T, AccT>(
                    a_block,
                    B_packed,
                    c_block,
                    m_block, n_block, k_block,
                    a_trans ? M : K, NC, N,
                    alpha, (k_outer == 0) ? beta : 1.0f);
            }
        }
    }
}

/**
 * Public interface for register-tiled GEMM
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
    float beta = 0.0f)
{
    register_tiled_gemm_optimized<T, AccT>(a, b, c, a_trans, b_trans, M, N, K, alpha, beta);
}

} // namespace mlx::core