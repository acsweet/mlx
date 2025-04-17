// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h> // AVX, AVX2, FMA, F16C intrinsics
#include <cstdint>     // For uint16_t, uint32_t
#include <cstring>     // For memcpy
#include <type_traits> // For std::is_same_v

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

// Forward declarations for pointer-based operations (maximizing performance)
template <typename T, int N>
inline Simd<T, N> load(const T* x);

template <typename T, int N>
inline void store(T* dst, Simd<T, N> x);

template <typename T, int N>
inline Simd<T, N> fma(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c);

// Specialization for float using AVX (256-bit registers)
template <>
constexpr int max_size<float> = 8;

template <>
struct Simd<float, 8> {
    static constexpr int size = 8;
    __m256 value;

    // Constructors
    Simd() : value(_mm256_setzero_ps()) {}           // Zero initialization
    Simd(float v) : value(_mm256_set1_ps(v)) {}      // Set all elements to v
    explicit Simd(__m256 v) : value(v) {}            // From intrinsic type
    Simd(const Simd& other) = default;               // Copy constructor
    Simd& operator=(const Simd& other) = default;    // Copy assignment
    
    // Allow implicit conversion *to* __m256 for use in intrinsics
    operator __m256() const {
        return value;
    }
};

// --- Batch Conversion Functions ---

// Generic batch conversion from any type to float - optimized for vectorized hardware
template <typename T>
inline void batch_convert_to_float(const T* src, float* dst, int n) {
    // Generic fallback - static_cast each element
    for (int i = 0; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Specialization for float (direct memcpy)
template <>
inline void batch_convert_to_float<float>(const float* src, float* dst, int n) {
    std::memcpy(dst, src, n * sizeof(float));
}

#ifdef __F16C__
// Optimized float16-to-float conversion using F16C instructions
// This specialization can be selected at compile-time when hardware support exists
template <typename T>
inline void batch_convert_from_float16(const T* src, float* dst, int n) {
    int i = 0;
    // Process blocks of 8 elements using vectorized F16C instructions
    for (; i <= n - 8; i += 8) {
        __m128i f16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f32_vals = _mm256_cvtph_ps(f16_vals);
        _mm256_storeu_ps(dst + i, f32_vals);
    }
    
    // Handle remainder elements individually
    for (; i < n; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}
#endif

// --- Optimized Pointer-Based Operations ---

// Unaligned load for float
template <>
inline Simd<float, 8> load<float, 8>(const float* x) {
    return Simd<float, 8>(_mm256_loadu_ps(x));
}

// Unaligned store for float
template <>
inline void store<float, 8>(float* dst, Simd<float, 8> x) {
    _mm256_storeu_ps(dst, x.value);
}

// --- Load/Convert/Store Operations ---

// Load data of type T and convert to float vector
template <typename T>
inline Simd<float, 8> load_convert_to_float(const T* src) {
    // Handle different input types
    alignas(32) float buffer[8];
    
    // Convert input to float (type-specific approach)
    for (int i = 0; i < 8; i++) {
        buffer[i] = static_cast<float>(src[i]);
    }
    
    return load<float, 8>(buffer);
}

// Store float vector, converting to destination type T
template <typename T>
inline void store_convert_from_float(T* dst, Simd<float, 8> src) {
    alignas(32) float buffer[8];
    store<float, 8>(buffer, src);
    
    // Convert float to output type (type-specific approach)
    for (int i = 0; i < 8; i++) {
        dst[i] = static_cast<T>(buffer[i]);
    }
}

// Performs: C = alpha * acc + beta * C (optimized for register-based tiling)
template <typename T>
inline void scale_accumulate_store(
    T* c_ptr,
    Simd<float, 8> acc,
    Simd<float, 8> alpha_vec,
    Simd<float, 8> beta_vec) {

    // Calculate alpha * accumulation - use direct intrinsics rather than operators
    // Avoid using operators since they're causing issues
    __m256 result = _mm256_mul_ps(alpha_vec.value, acc.value);

    // Check if beta is exactly 0.0f using vectorized comparison
    alignas(32) float beta_arr[8];
    _mm256_store_ps(beta_arr, beta_vec.value);

    if (beta_arr[0] != 0.0f) {
        // Load existing C values: beta * C[i, j]
        Simd<float, 8> c_old_vec = load_convert_to_float(c_ptr);
        
        // result = beta * c_old_vec + result (using direct FMA intrinsic)
        #ifdef __AVX2__
        result = _mm256_fmadd_ps(beta_vec.value, c_old_vec.value, result);
        #else
        __m256 beta_mul = _mm256_mul_ps(beta_vec.value, c_old_vec.value);
        result = _mm256_add_ps(result, beta_mul);
        #endif
    }

    // Store final result back to C (converts float vector back to T)
    store_convert_from_float<T>(c_ptr, Simd<float, 8>(result));
}

// --- Arithmetic Operations using Direct Intrinsics ---

// These operator overloads are explicitly defined for Simd<float, 8>
// to avoid template deduction issues in the core GEMM implementation

inline Simd<float, 8> operator+(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_add_ps(a.value, b.value));
}

inline Simd<float, 8> operator-(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_sub_ps(a.value, b.value));
}

inline Simd<float, 8> operator*(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_mul_ps(a.value, b.value));
}

inline Simd<float, 8> operator/(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_div_ps(a.value, b.value));
}

// --- Fused Multiply-Add ---
template <>
inline Simd<float, 8> fma<float, 8>(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
#ifdef __AVX2__
    return Simd<float, 8>(_mm256_fmadd_ps(a.value, b.value, c.value));
#else
    __m256 mul = _mm256_mul_ps(a.value, b.value);
    return Simd<float, 8>(_mm256_add_ps(mul, c.value));
#endif
}

// --- Prefetch Operations ---

// Prefetch operations for register tiling
template <typename T>
inline void prefetch_a(const T* a, int stride, int ahead = 1) {
    _mm_prefetch(reinterpret_cast<const char*>(a + stride * ahead), _MM_HINT_T0);
}

template <typename T>
inline void prefetch_b(const T* b, int stride, int ahead = 1) {
    _mm_prefetch(reinterpret_cast<const char*>(b + stride * ahead), _MM_HINT_T0);
}

} // namespace mlx::core::simd