// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/backend/cpu/simd/avx_simd_matmul.h"
#include <thread>
#include <vector>

namespace mlx::core {

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

/**
 * Get the optimal number of threads to use
 */
inline int get_num_threads() {
    return std::thread::hardware_concurrency();
}

/**
 * Batch convert float16 to float using SIMD when possible
 * This avoids element-by-element conversion overhead
 */
template <typename T, typename AccT>
void batch_convert(const T* src, AccT* dst, int n) {
    // Simple scalar conversion for now
    // In a real implementation, this would use F16C instructions for float16
    // if the hardware supports it
    for (int i = 0; i < n; i++) {
        dst[i] = static_cast<AccT>(src[i]);
    }
}

/**
 * Optimized block loading with vectorized type conversion
 */
template <int block_size, typename T, typename AccT>
void load_block(
    const T* in,
    AccT* out,
    int M,
    int N,
    int i,
    int j,
    bool transpose) {
    
    constexpr int simd_size = simd::max_size<AccT>;
    
    if (!transpose) {
        // Non-transposed case: try to use batch conversion
        for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
            const T* row_start = in + (i * block_size + ii) * N + j * block_size;
            AccT* out_row = out + ii * block_size;
            
            const int actual_width = std::min(block_size, N - j * block_size);
            
            if constexpr (std::is_same_v<T, AccT>) {
                // Same type, use memcpy for better performance
                std::wmemcpy(out_row, row_start, actual_width * sizeof(T));
            } else {
                // Different types, use batch conversion
                batch_convert<T, AccT>(row_start, out_row, actual_width);
            }
        }
    } else {
        // Transposed case with blocking for better cache efficiency
        const int actual_rows = std::min(block_size, M - i * block_size);
        const int actual_cols = std::min(block_size, N - j * block_size);
        
        // Use tiling for better cache behavior
        constexpr int tile_size = 4;
        
        for (int jj = 0; jj < actual_cols; jj += tile_size) {
            const int jj_end = std::min(jj + tile_size, actual_cols);
            
            for (int ii = 0; ii < actual_rows; ii += tile_size) {
                const int ii_end = std::min(ii + tile_size, actual_rows);
                
                for (int jjj = jj; jjj < jj_end; ++jjj) {
                    for (int iii = ii; iii < ii_end; ++iii) {
                        // Transpose during load
                        out[jjj * block_size + iii] = 
                            static_cast<AccT>(in[(j * block_size + jjj) * M + i * block_size + iii]);
                    }
                }
            }
        }
    }
}

/**
 * Worker function for processing a range of M blocks in parallel
 */
template <int block_size, int tile_size, typename T, typename AccT>
void process_blocks(
    const T* a,
    const T* b,
    T* c,
    bool a_trans,
    bool b_trans,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int i_start,
    int i_end,
    int j_start,
    int j_end) {
    
    constexpr int simd_size = simd::max_size<AccT>;
    int K_blocks = ceildiv(K, block_size);
    
    for (int i = i_start; i < i_end; i++) {
        for (int j = j_start; j < j_end; j++) {
            // Allocate thread-local blocks
            AccT c_block[block_size * block_size] = {0.0};
            AccT a_block[block_size * block_size];
            AccT b_block[block_size * block_size];
            
            for (int k = 0; k < K_blocks; k++) {
                // Compute actual block dimensions
                int K_block = std::min(block_size, K - k * block_size);
                
                // Load input blocks with optimized memory access
                if (a_trans) {
                    load_block<block_size>(a, a_block, K, M, k, i, true);
                } else {
                    load_block<block_size>(a, a_block, M, K, i, k, false);
                }
                
                if (b_trans) {
                    load_block<block_size>(b, b_block, N, K, j, k, false);
                } else {
                    load_block<block_size>(b, b_block, K, N, k, j, true);
                }
                
                // Process block in tiles
                for (int ii = 0; ii < block_size && i * block_size + ii < M; ii += tile_size) {
                    for (int jj = 0; jj < block_size && j * block_size + jj < N; jj += tile_size) {
                        // Get actual tile dimensions
                        int ii_limit = std::min(tile_size, M - i * block_size - ii);
                        int jj_limit = std::min(tile_size, N - j * block_size - jj);
                        
                        // Process each element in the tile
                        for (int iii = 0; iii < ii_limit; iii++) {
                            for (int jjj = 0; jjj < jj_limit; jjj++) {
                                AccT* c_ptr = &c_block[(ii + iii) * block_size + (jj + jjj)];
                                const AccT* a_row = &a_block[(ii + iii) * block_size];
                                const AccT* b_row = &b_block[(jj + jjj) * block_size];
                                
                                // Process K dimension in SIMD chunks
                                int kk = 0;
                                for (; kk + simd_size <= K_block; kk += simd_size) {
                                    // Use dot product for better performance
                                    *c_ptr += simd::dot_product_ptr<AccT, simd_size>(
                                        a_row + kk, b_row + kk);
                                }
                                
                                // Handle remaining elements
                                for (; kk < K_block; kk++) {
                                    *c_ptr += a_row[kk] * b_row[kk];
                                }
                            }
                        }
                    }
                }
            }
            
            // Write results back to output matrix with alpha/beta scaling
            for (int ii = 0; ii < block_size && i * block_size + ii < M; ii++) {
                for (int jj = 0; jj < block_size && j * block_size + jj < N; jj++) {
                    int c_idx = (i * block_size + ii) * N + j * block_size + jj;
                    if (beta != 0) {
                        c[c_idx] = static_cast<T>(
                            alpha * c_block[ii * block_size + jj] + beta * c[c_idx]);
                    } else {
                        c[c_idx] = static_cast<T>(alpha * c_block[ii * block_size + jj]);
                    }
                }
            }
        }
    }
}

/**
 * Optimized GEMM computation with multi-threading and improved memory access
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
    
    // Block size constants - these must be compile-time constants
    constexpr int block_size = 32;     // Larger block size for better parallelism
    constexpr int tile_size = 4;       // 4x4 tiles for register blocking
    
    // Calculate number of blocks
    int M_blocks = ceildiv(M, block_size);
    int N_blocks = ceildiv(N, block_size);
    
    // Determine if this is a large enough matrix to benefit from parallelism
    bool use_threading = (M * N >= 64 * 64) && (K >= 64);
    
    if (use_threading) {
        // Multi-threaded implementation
        int num_threads = get_num_threads();
        std::vector<std::thread> threads;
        
        // Simple 1D block distribution by rows
        int blocks_per_thread = ceildiv(M_blocks, num_threads);
        
        for (int t = 0; t < num_threads; t++) {
            int i_start = t * blocks_per_thread;
            int i_end = std::min((t + 1) * blocks_per_thread, M_blocks);
            
            // Skip empty ranges
            if (i_start >= i_end) {
                continue;
            }
            
            threads.emplace_back(
                process_blocks<block_size, tile_size, T, AccT>,
                a, b, c, a_trans, b_trans, M, N, K, alpha, beta,
                i_start, i_end, 0, N_blocks
            );
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded implementation
        process_blocks<block_size, tile_size, T, AccT>(
            a, b, c, a_trans, b_trans, M, N, K, alpha, beta,
            0, M_blocks, 0, N_blocks
        );
    }
}

} // namespace mlx::core