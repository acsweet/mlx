// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h> // AVX, AVX2, FMA, F16C intrinsics
#include <cstdint>     // For uint16_t, uint32_t
#include <cstring>     // For memcpy
#include <type_traits> // For std::is_same_v

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

// Forward declarations
template <typename T, int N>
inline Simd<T, N> broadcast(const T* x);

template <typename T, int N>
inline Simd<T, N> fma(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c);

template <typename T>
inline void scale_accumulate_store(
    T* c_ptr, 
    Simd<float, 8> acc, 
    Simd<float, 8> alpha_vec, 
    Simd<float, 8> beta_vec);

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

// --- Load Operations ---

// Unaligned load for float
template <>
inline Simd<float, 8> load<float, 8>(const float* x) {
    return Simd<float, 8>(_mm256_loadu_ps(x));
}

// Broadcast a single float value to all elements
template <>
inline Simd<float, 8> broadcast<float, 8>(const float* x) {
    return Simd<float, 8>(_mm256_broadcast_ss(x));
}

// Load data of type T and convert to float vector
template <typename T>
inline Simd<float, 8> load_convert_to_float(const T* src) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float16_t> ||
            std::is_same_v<T, bfloat16_t>,
        "load_convert_to_float requires float, float16_t, or bfloat16_t input");

    if constexpr (std::is_same_v<T, float>) {
        return load<float, 8>(src);
    } else if constexpr (std::is_same_v<T, float16_t>) {
#ifdef __F16C__
        // Requires -mf16c, assumes float16_t has standard binary16 layout
        __m128i f16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        return Simd<float, 8>(_mm256_cvtph_ps(f16_vals));
#else
        // Scalar fallback
        float buffer[8];
        for (int i = 0; i < 8; ++i) buffer[i] = static_cast<float>(src[i]);
        return load<float, 8>(buffer);
#endif
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        // No direct AVX2 conversion. Manual bit manipulation
        float buffer[8];
        alignas(alignof(bfloat16_t)) uint16_t raw_bits[8];
        std::memcpy(raw_bits, src, 8 * sizeof(bfloat16_t));

        for (int i = 0; i < 8; ++i) {
            uint32_t val_int = static_cast<uint32_t>(raw_bits[i]) << 16;
            std::memcpy(&buffer[i], &val_int, sizeof(float));
        }
        return load<float, 8>(buffer);
    }
}

// --- Store Operations ---

// Unaligned store for float
template <>
inline void store<float, 8>(float* dst, Simd<float, 8> x) {
    _mm256_storeu_ps(dst, x.value);
}

// Store float vector, converting to destination type T
template <typename T>
inline void store_convert_from_float(T* dst, Simd<float, 8> src) {
     static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float16_t> ||
            std::is_same_v<T, bfloat16_t>,
        "store_convert_from_float requires float, float16_t, or bfloat16_t output");

    if constexpr (std::is_same_v<T, float>) {
        store<float, 8>(dst, src);
    } else if constexpr (std::is_same_v<T, float16_t>) {
#ifdef __F16C__
        __m128i f16_result = _mm256_cvtps_ph(src.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), f16_result);
#else
        // Scalar fallback
        float buffer[8];
        store<float, 8>(buffer, src);
        for(int i=0; i<8; ++i) dst[i] = static_cast<float16_t>(buffer[i]);
#endif
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        float buffer[8];
        store<float, 8>(buffer, src);
        alignas(alignof(bfloat16_t)) uint16_t bf16_bits_arr[8];

        for (int i = 0; i < 8; ++i) {
            uint32_t val_int;
            std::memcpy(&val_int, &buffer[i], sizeof(float));
            bf16_bits_arr[i] = static_cast<uint16_t>(val_int >> 16);
        }
        std::memcpy(dst, bf16_bits_arr, 8 * sizeof(bfloat16_t));
    }
}

// --- Prefetching ---
template <int locality = 3, bool read_only = true>
inline void prefetch(const void* ptr) {
#ifdef __GNUC__
    __builtin_prefetch(ptr, read_only ? 0 : 1, locality);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // Microsoft compiler version
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0 + (3 - locality));
#endif
}

// --- Arithmetic Operations ---

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
    return (a * b) + c;
#endif
}

// --- Horizontal Operations ---

// Horizontal sum of all elements in the vector
inline float sum(Simd<float, 8> x) {
    __m256 val = x.value;
    __m128 hi = _mm256_extractf128_ps(val, 1);
    __m128 lo = _mm256_castps256_ps128(val);
    __m128 sum128 = _mm_add_ps(hi, lo);
    __m128 hadd1 = _mm_hadd_ps(sum128, sum128);
    __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
    return _mm_cvtss_f32(hadd2);
}

// --- Combined Operations for GEMM ---

// Performs: C = alpha * acc + beta * C (element-wise for a vector)
template <typename T>
inline void scale_accumulate_store(
    T* c_ptr,
    Simd<float, 8> acc,
    Simd<float, 8> alpha_vec,
    Simd<float, 8> beta_vec) {

    // Calculate alpha * accumulation
    Simd<float, 8> result = alpha_vec * acc;

    // Check if beta is exactly 0.0f
    float beta_scalar;
    _mm_store_ss(&beta_scalar, _mm256_castps256_ps128(beta_vec.value));

    if (beta_scalar != 0.0f) {
        // Load existing C values: beta * C[i, j]
        Simd<float, 8> c_old_vec = load_convert_to_float(c_ptr);
        // result = beta * c_old_vec + result; (using FMA)
        result = fma<float, 8>(beta_vec, c_old_vec, result);
    }

    // Store final result back to C (converts float vector back to T)
    store_convert_from_float<T>(c_ptr, result);
}

} // namespace mlx::core::simd