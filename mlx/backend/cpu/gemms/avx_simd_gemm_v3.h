// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/backend/cpu/simd/avx_simd_matmul.h"
#include <cstring>

namespace mlx::core {

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

/**
 * Prefetch memory to reduce cache misses
 */
inline void prefetch(const void* ptr) {
#ifdef __GNUC__
  __builtin_prefetch(ptr, 0, 3);
#endif
}

/**
 * Fast batch conversion from float16 to float with vectorization when possible
 */
template <typename T, typename AccT>
void fast_convert(const T* src, AccT* dst, int n) {
    // Handle special case for float16/bfloat16 to float conversion
    if constexpr (std::is_same_v<AccT, float>) {
        // Unroll small loops
        if (n <= 16) {
            for (int i = 0; i < n; i++) {
                dst[i] = static_cast<AccT>(src[i]);
            }
            return;
        }
        
        // Process in chunks for better cache locality
        constexpr int chunk = 16;
        int i = 0;
        
        // Use vectorization for aligned chunks if possible
        for (; i + chunk <= n; i += chunk) {
            // Prefetch next chunk to reduce cache misses
            if (i + 2*chunk <= n) {
                prefetch(src + i + chunk);
            }
            
            // Convert chunk of values
            for (int j = 0; j < chunk; j++) {
                dst[i + j] = static_cast<AccT>(src[i + j]);
            }
        }
        
        // Handle remaining elements
        for (; i < n; i++) {
            dst[i] = static_cast<AccT>(src[i]);
        }
    } else {
        // Generic conversion for other types
        for (int i = 0; i < n; i++) {
            dst[i] = static_cast<AccT>(src[i]);
        }
    }
}

/**
 * Optimized block loading with better cache utilization and prefetching
 */
template <int block_size, typename T, typename AccT>
void load_block_optimized(
    const T* in,
    AccT* out,
    int M,
    int N,
    int i,
    int j,
    bool transpose) {
    
    if (!transpose) {
        // Non-transposed case: process rows with stride-1 access
        const int rows = std::min(block_size, M - i * block_size);
        const int cols = std::min(block_size, N - j * block_size);
        
        for (int ii = 0; ii < rows; ++ii) {
            const T* row_src = in + (i * block_size + ii) * N + j * block_size;
            AccT* row_dst = out + ii * block_size;
            
            // Fast path for same-type copying
            if constexpr (std::is_same_v<T, AccT>) {
                std::wmemcpy(row_dst, row_src, cols * sizeof(T));
            } else {
                // Optimized conversion
                fast_convert<T, AccT>(row_src, row_dst, cols);
            }
        }
    }     else {
        // Transposed case: reduce cache misses with tiling
        const int rows = std::min(block_size, N - i * block_size); // Using N instead of K
        const int cols = std::min(block_size, M - j * block_size);
        
        // Process in small tiles to improve cache locality
        constexpr int tile = 8;
        
        for (int jj = 0; jj < cols; jj += tile) {
            const int jj_end = std::min(jj + tile, cols);
            
            for (int ii = 0; ii < rows; ii += tile) {
                const int ii_end = std::min(ii + tile, rows);
                
                // Process tile
                for (int jjj = jj; jjj < jj_end; ++jjj) {
                    for (int iii = ii; iii < ii_end; ++iii) {
                        out[jjj * block_size + iii] = 
                            static_cast<AccT>(in[(j * block_size + jjj) * N + i * block_size + iii]);
                    }
                }
            }
        }
    }
}


/**
 * Optimized macro-kernel for computing a block of the output matrix
 * Uses aggressive loop unrolling and register blocking for better performance
 */
template <int tile_m, int tile_n, int tile_k, typename AccT>
void compute_block_optimized(
    const AccT* A,
    const AccT* B,
    AccT* C,
    int lda,
    int ldb,
    int ldc,
    int k_blocks) {
    
    constexpr int simd_size = simd::max_size<AccT>;
    
    // Initialize accumulators to zero
    AccT c[tile_m][tile_n] = {};
    
    // Main computation loop
    for (int k = 0; k < k_blocks; k++) {
        // Process tile_k x tile_m x tile_n block
        for (int i = 0; i < tile_m; i++) {
            for (int j = 0; j < tile_n; j++) {
                const AccT* a_ptr = A + i * lda + k * tile_k;
                const AccT* b_ptr = B + j * ldb + k * tile_k;
                
                // Process in SIMD chunks
                int kk = 0;
                for (; kk + simd_size <= tile_k; kk += simd_size) {
                    // Use SIMD dot product
                    c[i][j] += simd::dot_product_ptr<AccT, simd_size>(
                        a_ptr + kk, b_ptr + kk);
                }
                
                // Handle remaining elements
                for (; kk < tile_k; kk++) {
                    c[i][j] += a_ptr[kk] * b_ptr[kk];
                }
            }
        }
    }
    
    // Write results to output
    for (int i = 0; i < tile_m; i++) {
        for (int j = 0; j < tile_n; j++) {
            C[i * ldc + j] = c[i][j];
        }
    }
}

/**
 * Highly optimized single-threaded matrix multiplication implementation
 */
template <typename T, typename AccT>
void simd_gemm_optimized(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    float alpha,
    float beta) {
    
    // // Optimal blocking parameters determined empirically
    // constexpr int mc = 64;  // Block size for M dimension
    // constexpr int nc = 128; // Block size for N dimension
    // constexpr int kc = 64;  // Block size for K dimension
    
    // // Micro-kernel tile sizes
    // constexpr int mr = 4;   // Tile size for M
    // constexpr int nr = 4;   // Tile size for N

    // Optimal blocking parameters for your CPU
    constexpr int mc = 64;  // Block size for M dimension
    constexpr int nc = 128;  // Block size for N dimension
    constexpr int kc = 64;  // Block size for K dimension
    
    // Micro-kernel tile sizes
    constexpr int mr = 4;   // Tile size for M
    constexpr int nr = 4;   // Tile size for N
    
    // Calculate number of blocks
    int M_blocks = ceildiv(M, mc);
    int N_blocks = ceildiv(N, nc);
    int K_blocks = ceildiv(K, kc);
    
    
    // Pack A and B into contiguous memory for better cache utilization
    // Note: For large matrices, we would allocate these dynamically,
    // but for simplicity we use stack allocation here assuming matrices aren't too large
    AccT A_packed[mc * kc] = {0}; // Packed block of A
    AccT B_packed[kc * nc] = {0}; // Packed block of B
    AccT C_block[mr * nr] = {0};  // Block of output

    // std::vector<AccT> A_packed(mc * kc, 0);
    // std::vector<AccT> B_packed(kc * nc, 0);
    // AccT C_block[mr * nr] = {0};
    
    // Main computation loop
    for (int i = 0; i < M_blocks; i++) {
        int m_block = std::min(mc, M - i * mc);
        
        for (int j = 0; j < N_blocks; j++) {
            int n_block = std::min(nc, N - j * nc);
            
            // Initialize output block if beta = 0
            if (beta == 0) {
                for (int ii = 0; ii < m_block; ii++) {
                    for (int jj = 0; jj < n_block; jj++) {
                        int c_idx = (i * mc + ii) * N + j * nc + jj;
                        c[c_idx] = static_cast<T>(0);
                    }
                }
            }
            
            for (int k = 0; k < K_blocks; k++) {
                int k_block = std::min(kc, K - k * kc);
                
                // Pack A and B for better cache efficiency
                if (a_trans) {
                    load_block_optimized<kc>(a, A_packed, K, M, k, i, true);
                } else {
                    load_block_optimized<mc>(a, A_packed, M, K, i, k, false);
                }
                
                if (b_trans) {
                    load_block_optimized<nc>(b, B_packed, N, K, j, k, false);
                } else {
                    load_block_optimized<kc>(b, B_packed, K, N, k, j, true);
                }
                
                // Compute C_block = A_packed * B_packed
                for (int ii = 0; ii < m_block; ii += mr) {
                    int m_micro = std::min(mr, m_block - ii);
                    
                    for (int jj = 0; jj < n_block; jj += nr) {
                        int n_micro = std::min(nr, n_block - jj);
                        
                        // Call micro-kernel for small block
                        if (m_micro == mr && n_micro == nr) {
                            // Full tile case - use optimized kernel
                            compute_block_optimized<mr, nr, kc, AccT>(
                                A_packed + ii * kc,
                                B_packed + jj * kc,
                                C_block,
                                kc, kc, nr,
                                1);
                                
                            // Apply alpha/beta and store
                            for (int iii = 0; iii < mr; iii++) {
                                for (int jjj = 0; jjj < nr; jjj++) {
                                    int c_idx = (i * mc + ii + iii) * N + j * nc + jj + jjj;
                                    if (beta != 0) {
                                        c[c_idx] = static_cast<T>(
                                            alpha * C_block[iii * nr + jjj] + beta * c[c_idx]);
                                    } else {
                                        c[c_idx] = static_cast<T>(alpha * C_block[iii * nr + jjj]);
                                    }
                                }
                            }
                        } else {
                            // Partial tile case - handle edge case manually
                            for (int iii = 0; iii < m_micro; iii++) {
                                for (int jjj = 0; jjj < n_micro; jjj++) {
                                    AccT sum = 0;
                                    
                                    for (int kk = 0; kk < k_block; kk++) {
                                        sum += A_packed[(ii + iii) * kc + kk] * 
                                              B_packed[(jj + jjj) * kc + kk];
                                    }
                                    
                                    int c_idx = (i * mc + ii + iii) * N + j * nc + jj + jjj;
                                    if (beta != 0) {
                                        c[c_idx] = static_cast<T>(
                                            alpha * sum + beta * c[c_idx]);
                                    } else {
                                        c[c_idx] = static_cast<T>(alpha * sum);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Drop-in replacement for existing simd_gemm that switches to optimized implementation
 */
template <typename T, typename AccT>
void simd_gemm(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    float alpha,
    float beta) {
    
    // Use the optimized implementation
    simd_gemm_optimized<T, AccT>(a, b, c, a_trans, b_trans, M, N, K, alpha, beta);
}

} // namespace mlx::core