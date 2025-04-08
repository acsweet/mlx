// mlx/backend/cpu/simd/avx_simd.h
#pragma once

#include <immintrin.h> // For AVX intrinsics
#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

// Template specialization for max_size
template <>
constexpr int max_size<float> = 8; // 8 floats in 256-bit AVX register

// AVX implementation for float
template <>
struct Simd<float, 8> {
    static constexpr int size = 8;
    __m256 value;
    
    Simd() : value(_mm256_setzero_ps()) {}
    Simd(float v) : value(_mm256_set1_ps(v)) {}
    explicit Simd(__m256 v) : value(v) {}
    
    // Allow scalar conversion
    operator float() const {
        return _mm256_cvtss_f32(value);
    }
};

// Load/store operations for float
template <>
inline Simd<float, 8> load<float, 8>(const float* x) {
    return Simd<float, 8>(_mm256_loadu_ps(x));
}

template <>
inline void store<float, 8>(float* dst, Simd<float, 8> x) {
    _mm256_storeu_ps(dst, x.value);
}

// Multiplication operation (essential for matmul)
inline Simd<float, 8> operator*(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_mul_ps(a.value, b.value));
}

// Reduction: sum - essential for matmul
inline float sum(Simd<float, 8> x) {
    // Horizontal sum using AVX
    __m128 hi = _mm256_extractf128_ps(x.value, 1);
    __m128 lo = _mm256_castps256_ps128(x.value);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

} // namespace mlx::core::simd