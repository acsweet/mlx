// Copyright © 2025 Apple Inc.
#pragma once

// #include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/simd/avx_simd_matmul.h"

namespace mlx::core {

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

/**
 * Optimized load_block function that uses vectorized loads where possible
 * and improves memory access patterns.
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
    // Non-transposed case: try to use direct vectorized loads
    for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
      const T* row_start = in + (i * block_size + ii) * N + j * block_size;
      AccT* out_row = out + ii * block_size;
      
      // Process full SIMD chunks
      int jj = 0;
      const int jj_limit = std::min(block_size, N - j * block_size);
      
      // Use vectorized loads for chunks of simd_size
      for (; jj + simd_size <= jj_limit; jj += simd_size) {
        // Perform vectorized load and conversion in one step if possible
        if constexpr (std::is_same_v<T, AccT>) {
          // Same type, just load directly
          auto vec = simd::load<T, simd_size>(row_start + jj);
          simd::store<AccT, simd_size>(out_row + jj, vec);
        } else {
          // Different types, load and convert
          // Let the existing MLX conversion handle this
          for (int k = 0; k < simd_size; k++) {
            // This lets MLX's existing conversion infrastructure handle the conversion
            out_row[jj + k] = static_cast<AccT>(row_start[jj + k]);
          }
        }
      }
      
      // Handle remaining elements
      for (; jj < jj_limit; ++jj) {
        out_row[jj] = static_cast<AccT>(row_start[jj]);
      }
    }
  } else {
    // Transposed case: use a more cache-friendly approach
    // First figure out actual dimensions of this block
    const int actual_rows = std::min(block_size, M - i * block_size);
    const int actual_cols = std::min(block_size, N - j * block_size);
    
    // Process in tiles for better cache locality
    constexpr int tile_size = 4; // Small enough to fit in L1 cache
    
    for (int ii = 0; ii < actual_rows; ii += tile_size) {
      for (int jj = 0; jj < actual_cols; jj += tile_size) {
        // Process a small tile at a time
        const int ii_end = std::min(ii + tile_size, actual_rows);
        const int jj_end = std::min(jj + tile_size, actual_cols);
        
        for (int iii = ii; iii < ii_end; ++iii) {
          for (int jjj = jj; jjj < jj_end; ++jjj) {
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
 * Compute a 4x4 block of output matrix elements, optimized for SIMD
 * operations and register reuse.
 */
template <typename AccT>
void compute_block_4x4(
    const AccT* a_block,
    const AccT* b_block,
    AccT* c_block,
    int block_size,
    int k_start,
    int k_end) {
  
  constexpr int tile_size = 4; // Process 4x4 tiles
  constexpr int simd_size = simd::max_size<AccT>;
  
  // Initialize accumulators for 4x4 tile
  AccT acc[tile_size][tile_size] = {};
  
  // Process k dimension in chunks
  for (int k = k_start; k < k_end; k += simd_size) {
    const int k_remaining = std::min(simd_size, k_end - k);
    
    // Compute full 4x4 tile using SIMD
    for (int i = 0; i < tile_size; ++i) {
      for (int j = 0; j < tile_size; ++j) {
        if (k_remaining == simd_size) {
          // Full SIMD chunk - use dot_product_ptr directly on pointers
          acc[i][j] += simd::dot_product_ptr<AccT, simd_size>(
              a_block + i * block_size + k,
              b_block + j * block_size + k);
        } else {
          // Handle remaining elements (partial chunk)
          AccT sum = 0;
          for (int kk = 0; kk < k_remaining; ++kk) {
            sum += a_block[i * block_size + k + kk] * b_block[j * block_size + k + kk];
          }
          acc[i][j] += sum;
        }
      }
    }
  }
  
  // Store accumulated results to output
  for (int i = 0; i < tile_size; ++i) {
    for (int j = 0; j < tile_size; ++j) {
      c_block[i * block_size + j] = acc[i][j];
    }
  }
}

/**
 * Main SIMD GEMM implementation optimized for better cache utilization and
 * SIMD efficiency.
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
  constexpr int block_size = 16;
  constexpr int simd_size = simd::max_size<AccT>;
  constexpr int tile_size = 4; // For 4x4 tile processing

  // Make sure block size is properly divisible
  static_assert(
      (block_size % tile_size) == 0,
      "Block size must be divisible by tile size");
      
  // Note: We don't require block_size to be divisible by simd_size
  // This allows us to work with different SIMD widths flexibly
  for (int i = 0; i < ceildiv(M, block_size); i++) {
    for (int j = 0; j < ceildiv(N, block_size); j++) {
      // Initialize result block
      AccT c_block[block_size * block_size] = {0.0};
      AccT a_block[block_size * block_size];
      AccT b_block[block_size * block_size];

      // Process K dimension in blocks
      for (int k = 0; k < ceildiv(K, block_size); k++) {
        // Determine actual block size for last block
        int k_block_size = std::min(block_size, K - k * block_size);
        
        // Load a and b blocks with optimized memory access
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

        // Compute result using tiled approach for better register utilization
        // Process the block in 4x4 tiles
        for (int ii = 0; ii < block_size && i * block_size + ii < M; ii += tile_size) {
          for (int jj = 0; jj < block_size && j * block_size + jj < N; jj += tile_size) {
            // Handle full tiles - 4x4 blocks
            if (ii + tile_size <= block_size && 
                i * block_size + ii + tile_size <= M &&
                jj + tile_size <= block_size && 
                j * block_size + jj + tile_size <= N) {
              
              // Compute a 4x4 block efficiently
              compute_block_4x4<AccT>(
                  a_block + ii * block_size, 
                  b_block + jj * block_size,
                  c_block + ii * block_size + jj,
                  block_size,
                  0,
                  k_block_size);
            } else {
              // Handle edge cases (partial tiles)
              const int ii_end = std::min(ii + tile_size, block_size);
              const int jj_end = std::min(jj + tile_size, block_size);
              
              for (int iii = ii; iii < ii_end && i * block_size + iii < M; ++iii) {
                for (int jjj = jj; jjj < jj_end && j * block_size + jjj < N; ++jjj) {
                  // Compute dot product using SIMD when possible
                  AccT sum = 0;
                  int kk = 0;
                  
                  // Process in SIMD chunks
                  for (; kk + simd_size <= k_block_size; kk += simd_size) {
                    // Use optimized pointer-based dot product for better performance
                    sum += simd::dot_product_ptr<AccT, simd_size>(
                        a_block + iii * block_size + kk,
                        b_block + jjj * block_size + kk);
                  }
                  
                  // Handle remaining elements
                  for (; kk < k_block_size; ++kk) {
                    sum += a_block[iii * block_size + kk] * b_block[jjj * block_size + kk];
                  }
                  
                  c_block[iii * block_size + jjj] += sum;
                }
              }
            }
          }
        }
      }

      // Apply alpha/beta and store the result back to output matrix
      for (int ii = 0; ii < block_size && i * block_size + ii < M; ++ii) {
        for (int jj = 0; jj < block_size && j * block_size + jj < N; ++jj) {
          auto c_idx = (i * block_size + ii) * N + j * block_size + jj;
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

} // namespace mlx::core