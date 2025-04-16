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

// Addition operation
inline Simd<float, 8> operator+(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_add_ps(a.value, b.value));
}

// Subtraction operation
inline Simd<float, 8> operator-(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_sub_ps(a.value, b.value));
}

// Division operation
inline Simd<float, 8> operator/(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_div_ps(a.value, b.value));
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

// Optimized dot_product for two float arrays using AVX
// This computes sum(a[i] * b[i]) for i=0..n-1 efficiently
inline float dot_product(const float* a, const float* b, int n) {
    float result = 0.0f;
    int i = 0;
    
    // Process 8 elements at a time with AVX
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 mul = _mm256_mul_ps(va, vb);
        
        // Horizontal sum of 8 multiplied values
        result += sum(Simd<float, 8>(mul));
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Templated dot_product for raw pointers - matches what simd_gemm might expect
template <typename T, int simd_width>
inline float dot_product_ptr(const T* a, const T* b) {
    if constexpr (std::is_same_v<T, float> && simd_width == 8) {
        // Use optimized AVX implementation for float arrays
        __m256 va = _mm256_loadu_ps(a);
        __m256 vb = _mm256_loadu_ps(b);
        __m256 mul = _mm256_mul_ps(va, vb);
        return sum(Simd<float, 8>(mul));
    } else {
        // Generic implementation for other types
        T result = 0;
        for (int i = 0; i < simd_width; i++) {
            result += a[i] * b[i];
        }
        return static_cast<float>(result);
    }
}

// Dot product for Simd types - useful for simd_gemm.h
inline float dot_product(Simd<float, 8> a, Simd<float, 8> b) {
    return sum(a * b);
}

// FMA operation if AVX2 is available
#ifdef __AVX2__
inline Simd<float, 8> fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
    return Simd<float, 8>(_mm256_fmadd_ps(a.value, b.value, c.value));
}
#endif


} // namespace mlx::core::simd