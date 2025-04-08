// mlx/backend/cpu/simd/avx_simd_matmul.h
#pragma once

#include <immintrin.h> // For AVX intrinsics
#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

#ifdef HAVE_AVX
// AVX optimized matrix multiplication helper functions

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
#endif

} // namespace mlx::core::simd