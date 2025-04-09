// mlx/backend/cpu/simd/avx_simd_matmul.h
#pragma once

#include <immintrin.h> // For AVX intrinsics
#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

#ifdef HAVE_AVX
// AVX optimized matrix multiplication helper functions
template <>
constexpr int max_size<float> = 8;

// Optimized load for 8 floats
inline __m256 avx_load_float8(const float* x) {
    return _mm256_loadu_ps(x);
}

// Optimized multiplication for 8 floats
inline __m256 avx_mul_float8(__m256 a, __m256 b) {
    return _mm256_mul_ps(a, b);
}

// Optimized horizontal sum for 8 floats
inline float avx_sum_float8(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// AVX optimized dot product for 8 floats
inline float avx_dot_product8(const float* a, const float* b) {
    __m256 av = avx_load_float8(a);
    __m256 bv = avx_load_float8(b);
    __m256 prod = avx_mul_float8(av, bv);
    return avx_sum_float8(prod);
}

// Apply scaling (alpha) to float vector
inline __m256 avx_scale_floats(__m256 values, float alpha) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    return _mm256_mul_ps(values, alpha_vec);
}

// Apply scaling and addition (alpha*x + beta*y)
inline __m256 avx_scale_add_floats(__m256 x, float alpha, __m256 y, float beta) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    // Compute alpha*x + beta*y using FMA if available
#ifdef __AVX2__
    __m256 scaled_x = _mm256_mul_ps(x, alpha_vec);
    return _mm256_fmadd_ps(y, beta_vec, scaled_x);
#else
    __m256 scaled_x = _mm256_mul_ps(x, alpha_vec);
    __m256 scaled_y = _mm256_mul_ps(y, beta_vec);
    return _mm256_add_ps(scaled_x, scaled_y);
#endif
}
#endif // HAVE_AVX
} // namespace mlx::core::simd