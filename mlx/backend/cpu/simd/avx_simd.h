#pragma once

#include <immintrin.h> // For AVX intrinsics
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>

#include "mlx/backend/cpu/simd/base_simd.h" // For fallback and base operations

namespace mlx::core::simd {

// Forward declarations of all specialized template classes
template<> struct Simd<float, 4>;
template<> struct Simd<float, 8>;
template<> struct Simd<double, 4>;
template<> struct Simd<int, 4>;
template<> struct Simd<int, 8>;
template<> struct Simd<uint32_t, 4>;
template<> struct Simd<uint32_t, 8>;
template<> struct Simd<int64_t, 4>;
template<> struct Simd<uint64_t, 4>;
template<> struct Simd<bool, 4>;
template<> struct Simd<bool, 8>;

// Override max_size for various types to use AVX
template <> inline constexpr int max_size<float> = 8;      // 8 floats in 256-bit AVX register
template <> inline constexpr int max_size<bool> = 8;       // 8 bools in 256-bit AVX register
template <> inline constexpr int max_size<int> = 8;        // 8 ints in 256-bit AVX register 
template <> inline constexpr int max_size<uint32_t> = 8;   // 8 uint32_t in 256-bit AVX register
template <> inline constexpr int max_size<int64_t> = 4;    // 4 int64_t in 256-bit AVX register
template <> inline constexpr int max_size<uint64_t> = 4;   // 4 uint64_t in 256-bit AVX register
template <> inline constexpr int max_size<double> = 4;     // 4 doubles in 256-bit AVX register

// Forward declarations for all sizes and types
template <typename T, int N> Simd<T, N> load(const T* x);
template <typename T, int N> void store(T* dst, Simd<T, N> x);

// Forward declarations for reduction operations
template <typename T, int N> T sum(Simd<T, N> x);
template <typename T, int N> T max(Simd<T, N> x);
template <typename T, int N> T min(Simd<T, N> x);
template <typename T, int N> T prod(Simd<T, N> x);
// template <int N> bool all(Simd<bool, N> x);
// template <int N> bool any(Simd<bool, N> x);

// ====== SIMD STRUCT DEFINITIONS ======

// Define 4-element float SIMD
template <>
struct Simd<float, 4> {
    static constexpr int size = 4;
    __m128 value;

    Simd() {}
    Simd(float v) : value(_mm_set1_ps(v)) {}
    Simd(__m128 v) : value(v) {}

    // Add general arithmetic constructor for converting scalars
    template <typename T>
    Simd(T v, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr) 
        : value(_mm_set1_ps(static_cast<float>(v))) {}

    float operator[](int idx) const {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, value);
        return tmp[idx];
    }

    float& operator[](int idx) {
        static thread_local float tmp_float;
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, value);
        tmp_float = tmp[idx];
        return tmp_float;
    }
};

// Define 8-element float SIMD
template <>
struct Simd<float, 8> {
    static constexpr int size = 8;
    __m256 value;

    Simd() {}
    Simd(float v) : value(_mm256_set1_ps(v)) {}
    Simd(__m256 v) : value(v) {}
    
    // Constructor from two half-size vectors now can use fully defined Simd<float, 4>
    Simd(const Simd<float, 4>& lo, const Simd<float, 4>& hi) {
        value = _mm256_insertf128_ps(_mm256_castps128_ps256(lo.value), hi.value, 1);
    }
    
    // Explicitly declare conversion constructors
    Simd(const Simd<int, 8>& v);
    Simd(const Simd<uint32_t, 8>& v);
    Simd(const Simd<bool, 8>& v);

    float operator[](int idx) const {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, value);
        return tmp[idx];
    }

    float& operator[](int idx) {
        static thread_local float tmp_float;
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, value);
        tmp_float = tmp[idx];
        return tmp_float;
    }
};

// Define 4-element int SIMD
template <>
struct Simd<int, 4> {
    static constexpr int size = 4;
    __m128i value;

    Simd() {}
    Simd(int v) : value(_mm_set1_epi32(v)) {}
    Simd(__m128i v) : value(v) {}

    Simd(const Simd<bool, 4>& v);

    int operator[](int idx) const {
        alignas(16) int tmp[4];
        _mm_store_si128((__m128i*)tmp, value);
        return tmp[idx];
    }

    int& operator[](int idx) {
        static thread_local int tmp_int;
        alignas(16) int tmp[4];
        _mm_store_si128((__m128i*)tmp, value);
        tmp_int = tmp[idx];
        return tmp_int;
    }
};

// Define 8-element int SIMD
template <>
struct Simd<int, 8> {
    static constexpr int size = 8;
    __m256i value;

    Simd() {}
    Simd(int v) : value(_mm256_set1_epi32(v)) {}
    Simd(__m256i v) : value(v) {}
    
    // Constructor from two half-size vectors now correctly uses defined type
    Simd(const Simd<int, 4>& lo, const Simd<int, 4>& hi) {
        value = _mm256_insertf128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1);
    }

    // Explicitly declare conversion constructors
    Simd(const Simd<float, 8>& v);
    Simd(const Simd<uint32_t, 8>& v);
    Simd(const Simd<bool, 8>& v);

    int operator[](int idx) const {
        alignas(32) int tmp[8];
        _mm256_store_si256((__m256i*)tmp, value);
        return tmp[idx];
    }

    int& operator[](int idx) {
        static thread_local int tmp_int;
        alignas(32) int tmp[8];
        _mm256_store_si256((__m256i*)tmp, value);
        tmp_int = tmp[idx];
        return tmp_int;
    }
};

// Define 4-element uint32_t SIMD
template <>
struct Simd<uint32_t, 4> {
    static constexpr int size = 4;
    __m128i value;

    Simd() {}
    Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
    Simd(__m128i v) : value(v) {}
    Simd(int v) : value(_mm_set1_epi32(v < 0 ? 0 : v)) {}

    Simd(const Simd<bool, 4>& v);

    uint32_t operator[](int idx) const {
        alignas(16) uint32_t tmp[4];
        _mm_store_si128((__m128i*)tmp, value);
        return tmp[idx];
    }

    uint32_t& operator[](int idx) {
        static thread_local uint32_t tmp_uint;
        alignas(16) uint32_t tmp[4];
        _mm_store_si128((__m128i*)tmp, value);
        tmp_uint = tmp[idx];
        return tmp_uint;
    }
};

// Define 8-element uint32_t SIMD
template <>
struct Simd<uint32_t, 8> {
    static constexpr int size = 8;
    __m256i value;

    Simd() {}
    Simd(uint32_t v) : value(_mm256_set1_epi32(v)) {}
    Simd(__m256i v) : value(v) {}
    Simd(int v) : value(_mm256_set1_epi32(v < 0 ? 0 : v)) {}
    
    // Constructor from two half-size vectors
    Simd(const Simd<uint32_t, 4>& lo, const Simd<uint32_t, 4>& hi) {
        value = _mm256_insertf128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1);
    }

    // Explicitly declare conversion constructors
    Simd(const Simd<float, 8>& v);
    Simd(const Simd<int, 8>& v);
    Simd(const Simd<bool, 8>& v);

    uint32_t operator[](int idx) const {
        alignas(32) uint32_t tmp[8];
        _mm256_store_si256((__m256i*)tmp, value);
        return tmp[idx];
    }

    uint32_t& operator[](int idx) {
        static thread_local uint32_t tmp_uint;
        alignas(32) uint32_t tmp[8];
        _mm256_store_si256((__m256i*)tmp, value);
        tmp_uint = tmp[idx];
        return tmp_uint;
    }
};

// Define 4-element double SIMD
template <>
struct Simd<double, 4> {
    static constexpr int size = 4;
    __m256d value;

    Simd() {}
    Simd(double v) : value(_mm256_set1_pd(v)) {}
    Simd(__m256d v) : value(v) {}

    template <typename T>
    Simd(T v, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr) 
        : value(_mm256_set1_pd(static_cast<double>(v))) {}

    // Explicitly declare conversion constructors
    Simd(const Simd<int64_t, 4>& v);
    Simd(const Simd<uint64_t, 4>& v);
    Simd(const Simd<bool, 4>& v);

    double operator[](int idx) const {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, value);
        return tmp[idx];
    }

    double& operator[](int idx) {
        static thread_local double tmp_double;
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, value);
        tmp_double = tmp[idx];
        return tmp_double;
    }
};

// Define 4-element int64_t SIMD
template <>
struct Simd<int64_t, 4> {
    static constexpr int size = 4;
    __m256i value;

    Simd() {}
    Simd(int64_t v) : value(_mm256_set1_epi64x(v)) {}
    Simd(__m256i v) : value(v) {}

    // Explicitly declare conversion constructors
    Simd(const Simd<uint64_t, 4>& v);
    Simd(const Simd<double, 4>& v);
    Simd(const Simd<bool, 4>& v);

    int64_t operator[](int idx) const {
        alignas(32) int64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, value);
        return tmp[idx];
    }

    int64_t& operator[](int idx) {
        static thread_local int64_t tmp_int64;
        alignas(32) int64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, value);
        tmp_int64 = tmp[idx];
        return tmp_int64;
    }
};

// Define 4-element uint64_t SIMD
template <>
struct Simd<uint64_t, 4> {
    static constexpr int size = 4;
    __m256i value;

    Simd() {}
    Simd(uint64_t v) : value(_mm256_set1_epi64x(v)) {}
    Simd(__m256i v) : value(v) {}
    Simd(int v) : value(_mm256_set1_epi64x(v < 0 ? 0 : v)) {}

    // Explicitly declare conversion constructors
    Simd(const Simd<int64_t, 4>& v);
    Simd(const Simd<double, 4>& v);
    Simd(const Simd<bool, 4>& v);

    uint64_t operator[](int idx) const {
        alignas(32) uint64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, value);
        return tmp[idx];
    }

    uint64_t& operator[](int idx) {
        static thread_local uint64_t tmp_uint64;
        alignas(32) uint64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, value);
        tmp_uint64 = tmp[idx];
        return tmp_uint64;
    }
};

// Define 4-element bool SIMD
template <>
struct Simd<bool, 4> {
    static constexpr int size = 4;
    __m128 value; // Represent booleans as floats where true = all 1s (-1.0f), false = all 0s

    Simd() {}
    Simd(bool v) : value(_mm_set1_ps(v ? -1.0f : 0.0f)) {}
    Simd(__m128 v) : value(v) {}
    Simd(int v) : value(_mm_set1_ps(v != 0 ? -1.0f : 0.0f)) {}

    Simd(const Simd<int64_t, 4>& v);
    Simd(const Simd<uint64_t, 4>& v);
    Simd(const Simd<double, 4>& v);

    bool operator[](int idx) const {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, value);
        return tmp[idx] != 0.0f;
    }

    bool& operator[](int idx) {
        static thread_local bool tmp_bool;
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, value);
        tmp_bool = tmp[idx] != 0.0f;
        return tmp_bool;
    }
};

// Define 8-element bool SIMD
template <>
struct Simd<bool, 8> {
    static constexpr int size = 8;
    __m256 value; // Represent booleans as floats where true = all 1s (-1.0f), false = all 0s

    Simd() {}
    Simd(bool v) : value(_mm256_set1_ps(v ? -1.0f : 0.0f)) {}
    Simd(__m256 v) : value(v) {}
    Simd(int v) : value(_mm256_set1_ps(v != 0 ? -1.0f : 0.0f)) {}
    
    // Constructor from two half-size vectors
    Simd(const Simd<bool, 4>& lo, const Simd<bool, 4>& hi) {
        value = _mm256_insertf128_ps(_mm256_castps128_ps256(lo.value), hi.value, 1);
    }

    // Explicitly declare conversion constructors
    Simd(const Simd<float, 8>& v);
    Simd(const Simd<int, 8>& v);
    Simd(const Simd<uint32_t, 8>& v);

    bool operator[](int idx) const {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, value);
        return tmp[idx] != 0.0f;
    }

    bool& operator[](int idx) {
        static thread_local bool tmp_bool;
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, value);
        tmp_bool = tmp[idx] != 0.0f;
        return tmp_bool;
    }
};

// ====== CONVERSION CONSTRUCTORS ======

// float <-> int conversions
inline Simd<float, 8>::Simd(const Simd<int, 8>& v) : value(_mm256_cvtepi32_ps(v.value)) {}

inline Simd<int, 8>::Simd(const Simd<float, 8>& v) : value(_mm256_cvtps_epi32(v.value)) {}

// float <-> uint32_t conversions
inline Simd<float, 8>::Simd(const Simd<uint32_t, 8>& v) : value(_mm256_cvtepi32_ps(v.value)) {}

inline Simd<uint32_t, 8>::Simd(const Simd<float, 8>& v) : value(_mm256_cvtps_epi32(v.value)) {}

// float <-> bool conversions
inline Simd<float, 8>::Simd(const Simd<bool, 8>& v) : value(v.value) {}

inline Simd<bool, 8>::Simd(const Simd<float, 8>& v) : value(_mm256_cmp_ps(v.value, _mm256_setzero_ps(), _CMP_NEQ_OQ)) {}

// int <-> uint32_t conversions (simple cast, no conversion needed)
inline Simd<int, 8>::Simd(const Simd<uint32_t, 8>& v) : value(v.value) {}

inline Simd<uint32_t, 8>::Simd(const Simd<int, 8>& v) : value(v.value) {}

// int <-> bool conversions
inline Simd<int, 8>::Simd(const Simd<bool, 8>& v) : value(_mm256_cvtps_epi32(v.value)) {}

inline Simd<bool, 8>::Simd(const Simd<int, 8>& v) : value(_mm256_cmp_ps(_mm256_cvtepi32_ps(v.value), _mm256_setzero_ps(), _CMP_NEQ_OQ)) {}

// uint32_t <-> bool conversions
inline Simd<uint32_t, 8>::Simd(const Simd<bool, 8>& v) : value(_mm256_cvtps_epi32(v.value)) {}

inline Simd<bool, 8>::Simd(const Simd<uint32_t, 8>& v) : value(_mm256_cmp_ps(_mm256_cvtepi32_ps(v.value), _mm256_setzero_ps(), _CMP_NEQ_OQ)) {}

// 64-bit type conversions
inline Simd<int64_t, 4>::Simd(const Simd<uint64_t, 4>& v) : value(v.value) {}

inline Simd<uint64_t, 4>::Simd(const Simd<int64_t, 4>& v) : value(v.value) {}

// Double conversions
inline Simd<double, 4>::Simd(const Simd<int64_t, 4>& v) : value(_mm256_cvtepi64_pd(v.value)) {}

inline Simd<int64_t, 4>::Simd(const Simd<double, 4>& v) : value(_mm256_cvtpd_epi64(v.value)) {}

inline Simd<double, 4>::Simd(const Simd<uint64_t, 4>& v) : value(_mm256_cvtepu64_pd(v.value)) {}

inline Simd<uint64_t, 4>::Simd(const Simd<double, 4>& v) : value(_mm256_cvtpd_epu64(v.value)) {}

// Conversion from bool to int (4-wide) - already defined but including for completeness
inline Simd<int, 4>::Simd(const Simd<bool, 4>& v) : 
    value(_mm_cvtps_epi32(v.value)) {}

// Conversion from bool to uint32_t (4-wide)
inline Simd<uint32_t, 4>::Simd(const Simd<bool, 4>& v) : 
    value(_mm_cvtps_epi32(v.value)) {}

// Conversion from int64_t to bool (4-wide)
inline Simd<bool, 4>::Simd(const Simd<int64_t, 4>& v) {
    alignas(32) int64_t values[4];
    alignas(16) float result[4];
    _mm256_store_si256((__m256i*)values, v.value);
    for (int i = 0; i < 4; i++) {
        result[i] = (values[i] != 0) ? -1.0f : 0.0f;
    }
    value = _mm_loadu_ps(result);
}

// Conversion from uint64_t to bool (4-wide)
inline Simd<bool, 4>::Simd(const Simd<uint64_t, 4>& v) {
    alignas(32) uint64_t values[4];
    alignas(16) float result[4];
    _mm256_store_si256((__m256i*)values, v.value);
    for (int i = 0; i < 4; i++) {
        result[i] = (values[i] != 0) ? -1.0f : 0.0f;
    }
    value = _mm_loadu_ps(result);
}

// Conversion from bool to int64_t (4-wide)
inline Simd<int64_t, 4>::Simd(const Simd<bool, 4>& v) {
    alignas(16) float values[4];
    alignas(32) int64_t result[4];
    _mm_store_ps(values, v.value);
    for (int i = 0; i < 4; i++) {
        result[i] = (values[i] != 0.0f) ? 1 : 0;
    }
    value = _mm256_loadu_si256((__m256i*)result);
}

// Conversion from bool to uint64_t (4-wide)
inline Simd<uint64_t, 4>::Simd(const Simd<bool, 4>& v) {
    alignas(16) float values[4];
    alignas(32) uint64_t result[4];
    _mm_store_ps(values, v.value);
    for (int i = 0; i < 4; i++) {
        result[i] = (values[i] != 0.0f) ? 1 : 0;
    }
    value = _mm256_loadu_si256((__m256i*)result);
}

// Conversion from double to bool (4-wide)
inline Simd<bool, 4>::Simd(const Simd<double, 4>& v) {
    // Convert double to float first to get the result in the right format for bool
    __m128 tmp = _mm_cvtpd_ps(_mm256_extractf128_pd(v.value, 0));
    __m128 tmp2 = _mm_cvtpd_ps(_mm256_extractf128_pd(v.value, 1));
    __m128 combined = _mm_shuffle_ps(tmp, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    value = _mm_cmpneq_ps(combined, _mm_setzero_ps());
}

// Conversion from bool to double (4-wide)
inline Simd<double, 4>::Simd(const Simd<bool, 4>& v) {
    alignas(16) float values[4];
    alignas(32) double result[4];
    _mm_store_ps(values, v.value);
    for (int i = 0; i < 4; i++) {
        result[i] = (values[i] != 0.0f) ? 1.0 : 0.0;
    }
    value = _mm256_loadu_pd(result);
}

// ====== LOAD/STORE OPERATIONS ======

// For float/double operations (no casting)
template <>
inline Simd<float, 8> load<float, 8>(const float* x) {
    return Simd<float, 8>(_mm256_loadu_ps(x));
}

template <>
inline void store<float, 8>(float* dst, Simd<float, 8> x) {
    _mm256_storeu_ps(dst, x.value);
}

template <>
inline Simd<float, 4> load<float, 4>(const float* x) {
    return Simd<float, 4>(_mm_loadu_ps(x));
}

template <>
inline void store<float, 4>(float* dst, Simd<float, 4> x) {
    _mm_storeu_ps(dst, x.value);
}

template <>
inline Simd<double, 4> load<double, 4>(const double* x) {
    return Simd<double, 4>(_mm256_loadu_pd(x));
}

template <>
inline void store<double, 4>(double* dst, Simd<double, 4> x) {
    _mm256_storeu_pd(dst, x.value);
}

// For int/uint32_t with __m256i
template <>
inline Simd<int, 8> load<int, 8>(const int* x) {
    return Simd<int, 8>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)));
}

template <>
inline void store<int, 8>(int* dst, Simd<int, 8> x) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), x.value);
}

template <>
inline Simd<int, 4> load<int, 4>(const int* x) {
    return Simd<int, 4>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x)));
}

template <>
inline void store<int, 4>(int* dst, Simd<int, 4> x) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), x.value);
}

template <>
inline Simd<int64_t, 4> load<int64_t, 4>(const int64_t* x) {
    return Simd<int64_t, 4>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)));
}

template <>
inline void store<int64_t, 4>(int64_t* dst, Simd<int64_t, 4> x) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), x.value);
}

template <>
inline Simd<uint32_t, 8> load<uint32_t, 8>(const uint32_t* x) {
    return Simd<uint32_t, 8>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)));
}

template <>
inline void store<uint32_t, 8>(uint32_t* dst, Simd<uint32_t, 8> x) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), x.value);
}

template <>
inline Simd<uint32_t, 4> load<uint32_t, 4>(const uint32_t* x) {
    return Simd<uint32_t, 4>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x)));
}

template <>
inline void store<uint32_t, 4>(uint32_t* dst, Simd<uint32_t, 4> x) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), x.value);
}

template <>
inline Simd<uint64_t, 4> load<uint64_t, 4>(const uint64_t* x) {
    return Simd<uint64_t, 4>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)));
}

template <>
inline void store<uint64_t, 4>(uint64_t* dst, Simd<uint64_t, 4> x) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), x.value);
}

// Load/store for bool (special case)
template <>
inline Simd<bool, 8> load<bool, 8>(const bool* x) {
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; i++) {
        tmp[i] = x[i] ? -1.0f : 0.0f;
    }
    return Simd<bool, 8>(_mm256_loadu_ps(tmp));
}

template <>
inline void store<bool, 8>(bool* dst, Simd<bool, 8> x) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, x.value);
    for (int i = 0; i < 8; i++) {
        dst[i] = tmp[i] != 0.0f;
    }
}

template <>
inline Simd<bool, 4> load<bool, 4>(const bool* x) {
    alignas(16) float tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = x[i] ? -1.0f : 0.0f;
    }
    return Simd<bool, 4>(_mm_loadu_ps(tmp));
}

template <>
inline void store<bool, 4>(bool* dst, Simd<bool, 4> x) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, x.value);
    for (int i = 0; i < 4; i++) {
        dst[i] = tmp[i] != 0.0f;
    }
}

// ====== ARITHMETIC OPERATIONS ======

// Define macros for scalar arithmetic operations to reduce repetition
#define DEFINE_SCALAR_ARITHMETIC_OPS(TYPE, SIZE, OP) \
inline Simd<TYPE, SIZE> operator OP(Simd<TYPE, SIZE> a, TYPE b) { \
    return a OP Simd<TYPE, SIZE>(b); \
} \
\
inline Simd<TYPE, SIZE> operator OP(TYPE a, Simd<TYPE, SIZE> b) { \
    return Simd<TYPE, SIZE>(a) OP b; \
}

// Arithmetic for float 8-wide
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

// Unary minus for float 8-wide
inline Simd<float, 8> operator-(Simd<float, 8> a) {
    return Simd<float, 8>(_mm256_xor_ps(a.value, _mm256_set1_ps(-0.0f)));
}

// Scalar arithmetic for float 8-wide
DEFINE_SCALAR_ARITHMETIC_OPS(float, 8, +)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 8, -)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 8, *)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 8, /)

// Arithmetic for float 4-wide
inline Simd<float, 4> operator+(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_add_ps(a.value, b.value));
}

inline Simd<float, 4> operator-(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_sub_ps(a.value, b.value));
}

inline Simd<float, 4> operator*(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_mul_ps(a.value, b.value));
}

inline Simd<float, 4> operator/(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_div_ps(a.value, b.value));
}

// Unary minus for float 4-wide
inline Simd<float, 4> operator-(Simd<float, 4> a) {
    return Simd<float, 4>(_mm_xor_ps(a.value, _mm_set1_ps(-0.0f)));
}

// Scalar arithmetic for float 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(float, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(float, 4, /)

// Arithmetic for int 8-wide
inline Simd<int, 8> operator+(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<int, 8>(_mm256_add_epi32(a.value, b.value));
}

inline Simd<int, 8> operator-(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<int, 8>(_mm256_sub_epi32(a.value, b.value));
}

inline Simd<int, 8> operator*(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<int, 8>(_mm256_mullo_epi32(a.value, b.value));
}

// Division for int 8-wide (no direct AVX instruction, use scalar)
inline Simd<int, 8> operator/(Simd<int, 8> a, Simd<int, 8> b) {
    alignas(32) int a_array[8], b_array[8], result[8];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 8; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<int, 8>(result);
}

// Unary minus for int 8-wide
inline Simd<int, 8> operator-(Simd<int, 8> a) {
    return Simd<int, 8>(_mm256_sub_epi32(_mm256_setzero_si256(), a.value));
}

// Scalar arithmetic for int 8-wide
DEFINE_SCALAR_ARITHMETIC_OPS(int, 8, +)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 8, -)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 8, *)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 8, /)

// Arithmetic for int 4-wide
inline Simd<int, 4> operator+(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<int, 4>(_mm_add_epi32(a.value, b.value));
}

inline Simd<int, 4> operator-(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<int, 4>(_mm_sub_epi32(a.value, b.value));
}

inline Simd<int, 4> operator*(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<int, 4>(_mm_mullo_epi32(a.value, b.value));
}

// Division for int 4-wide (use scalar)
inline Simd<int, 4> operator/(Simd<int, 4> a, Simd<int, 4> b) {
    alignas(16) int a_array[4], b_array[4], result[4];
    _mm_store_si128((__m128i*)a_array, a.value);
    _mm_store_si128((__m128i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<int, 4>(result);
}

// Unary minus for int 4-wide
inline Simd<int, 4> operator-(Simd<int, 4> a) {
    return Simd<int, 4>(_mm_sub_epi32(_mm_setzero_si128(), a.value));
}

// Scalar arithmetic for int 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(int, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(int, 4, /)

// Arithmetic for uint32_t 8-wide
inline Simd<uint32_t, 8> operator+(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    return Simd<uint32_t, 8>(_mm256_add_epi32(a.value, b.value));
}

inline Simd<uint32_t, 8> operator-(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    return Simd<uint32_t, 8>(_mm256_sub_epi32(a.value, b.value));
}

inline Simd<uint32_t, 8> operator*(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    return Simd<uint32_t, 8>(_mm256_mullo_epi32(a.value, b.value));
}

// Division for uint32_t 8-wide (use scalar implementation)
inline Simd<uint32_t, 8> operator/(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    alignas(32) uint32_t a_array[8], b_array[8], result[8];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 8; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<uint32_t, 8>(result);
}

// Unary minus for uint32_t 8-wide
inline Simd<uint32_t, 8> operator-(Simd<uint32_t, 8> a) {
    return Simd<uint32_t, 8>(_mm256_sub_epi32(_mm256_setzero_si256(), a.value));
}

// Scalar arithmetic for uint32_t 8-wide
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 8, +)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 8, -)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 8, *)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 8, /)

// Arithmetic for uint32_t 4-wide
inline Simd<uint32_t, 4> operator+(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    return Simd<uint32_t, 4>(_mm_add_epi32(a.value, b.value));
}

inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    return Simd<uint32_t, 4>(_mm_sub_epi32(a.value, b.value));
}

inline Simd<uint32_t, 4> operator*(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    return Simd<uint32_t, 4>(_mm_mullo_epi32(a.value, b.value));
}

// Division for uint32_t 4-wide (use scalar)
inline Simd<uint32_t, 4> operator/(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    alignas(16) uint32_t a_array[4], b_array[4], result[4];
    _mm_store_si128((__m128i*)a_array, a.value);
    _mm_store_si128((__m128i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<uint32_t, 4>(result);
}

// Unary minus for uint32_t 4-wide
inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a) {
    return Simd<uint32_t, 4>(_mm_sub_epi32(_mm_setzero_si128(), a.value));
}

// Scalar arithmetic for uint32_t 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(uint32_t, 4, /)

// Arithmetic for double 4-wide
inline Simd<double, 4> operator+(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_add_pd(a.value, b.value));
}

inline Simd<double, 4> operator-(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_sub_pd(a.value, b.value));
}

inline Simd<double, 4> operator*(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_mul_pd(a.value, b.value));
}

inline Simd<double, 4> operator/(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_div_pd(a.value, b.value));
}

// Unary minus for double 4-wide
inline Simd<double, 4> operator-(Simd<double, 4> a) {
    return Simd<double, 4>(_mm256_xor_pd(a.value, _mm256_set1_pd(-0.0)));
}

// Scalar arithmetic for double 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(double, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(double, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(double, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(double, 4, /)

// Arithmetic for int64_t 4-wide
inline Simd<int64_t, 4> operator+(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
    return Simd<int64_t, 4>(_mm256_add_epi64(a.value, b.value));
}

inline Simd<int64_t, 4> operator-(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
    return Simd<int64_t, 4>(_mm256_sub_epi64(a.value, b.value));
}

// Multiplication for int64_t 4-wide (no direct AVX instruction)
inline Simd<int64_t, 4> operator*(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
    alignas(32) int64_t a_array[4], b_array[4], result[4];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] * b_array[i];
    }
    return load<int64_t, 4>(result);
}

// Division for int64_t 4-wide (use scalar)
inline Simd<int64_t, 4> operator/(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
    alignas(32) int64_t a_array[4], b_array[4], result[4];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<int64_t, 4>(result);
}

// Unary minus for int64_t 4-wide
inline Simd<int64_t, 4> operator-(Simd<int64_t, 4> a) {
    return Simd<int64_t, 4>(_mm256_sub_epi64(_mm256_setzero_si256(), a.value));
}

// Scalar arithmetic for int64_t 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(int64_t, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(int64_t, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(int64_t, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(int64_t, 4, /)

// Arithmetic for uint64_t 4-wide
inline Simd<uint64_t, 4> operator+(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
    return Simd<uint64_t, 4>(_mm256_add_epi64(a.value, b.value));
}

inline Simd<uint64_t, 4> operator-(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
    return Simd<uint64_t, 4>(_mm256_sub_epi64(a.value, b.value));
}

// Multiplication for uint64_t 4-wide (no direct AVX instruction)
inline Simd<uint64_t, 4> operator*(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
    alignas(32) uint64_t a_array[4], b_array[4], result[4];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] * b_array[i];
    }
    return load<uint64_t, 4>(result);
}

// Division for uint64_t 4-wide (use scalar)
inline Simd<uint64_t, 4> operator/(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
    alignas(32) uint64_t a_array[4], b_array[4], result[4];
    _mm256_store_si256((__m256i*)a_array, a.value);
    _mm256_store_si256((__m256i*)b_array, b.value);
    for (int i = 0; i < 4; i++) {
        result[i] = a_array[i] / b_array[i];
    }
    return load<uint64_t, 4>(result);
}

// Unary minus for uint64_t 4-wide
inline Simd<uint64_t, 4> operator-(Simd<uint64_t, 4> a) {
    return Simd<uint64_t, 4>(_mm256_sub_epi64(_mm256_setzero_si256(), a.value));
}

// Scalar arithmetic for uint64_t 4-wide
DEFINE_SCALAR_ARITHMETIC_OPS(uint64_t, 4, +)
DEFINE_SCALAR_ARITHMETIC_OPS(uint64_t, 4, -)
DEFINE_SCALAR_ARITHMETIC_OPS(uint64_t, 4, *)
DEFINE_SCALAR_ARITHMETIC_OPS(uint64_t, 4, /)

// Boolean operations (special case)
inline Simd<bool, 8> operator+(Simd<bool, 8> a, Simd<bool, 8> b) {
    return Simd<bool, 8>(_mm256_or_ps(a.value, b.value));
}

inline Simd<bool, 8> operator-(Simd<bool, 8> a, Simd<bool, 8> b) {
    return Simd<bool, 8>(_mm256_andnot_ps(b.value, a.value));
}

inline Simd<bool, 8> operator*(Simd<bool, 8> a, Simd<bool, 8> b) {
    return Simd<bool, 8>(_mm256_and_ps(a.value, b.value));
}

inline Simd<bool, 8> operator/(Simd<bool, 8> a, Simd<bool, 8> b) {
    return Simd<bool, 8>(_mm256_andnot_ps(b.value, a.value)); // Same as -
}

inline Simd<bool, 8> operator-(Simd<bool, 8> a) {
    return Simd<bool, 8>(_mm256_xor_ps(a.value, _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF))));
}

// Boolean operations for 4-wide
inline Simd<bool, 4> operator+(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_or_ps(a.value, b.value));
}

inline Simd<bool, 4> operator-(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_andnot_ps(b.value, a.value));
}

inline Simd<bool, 4> operator*(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_and_ps(a.value, b.value));
}

inline Simd<bool, 4> operator/(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_andnot_ps(b.value, a.value)); // Same as -
}

inline Simd<bool, 4> operator-(Simd<bool, 4> a) {
    return Simd<bool, 4>(_mm_xor_ps(a.value, _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFF))));
}

// ====== BITWISE OPERATIONS ======

// Define macro for bitwise operations
#define DEFINE_BITWISE_OPS(TYPE, SIZE, OP, INSTR) \
inline Simd<TYPE, SIZE> operator OP(Simd<TYPE, SIZE> a, Simd<TYPE, SIZE> b) { \
    return Simd<TYPE, SIZE>(INSTR(a.value, b.value)); \
}

// Bitwise operations for int 8-wide
DEFINE_BITWISE_OPS(int, 8, &, _mm256_and_si256)
DEFINE_BITWISE_OPS(int, 8, |, _mm256_or_si256)
DEFINE_BITWISE_OPS(int, 8, ^, _mm256_xor_si256)

// Bitwise operations for int 4-wide
DEFINE_BITWISE_OPS(int, 4, &, _mm_and_si128)
DEFINE_BITWISE_OPS(int, 4, |, _mm_or_si128)
DEFINE_BITWISE_OPS(int, 4, ^, _mm_xor_si128)

// Bitwise operations for uint32_t 8-wide
DEFINE_BITWISE_OPS(uint32_t, 8, &, _mm256_and_si256)
DEFINE_BITWISE_OPS(uint32_t, 8, |, _mm256_or_si256)
DEFINE_BITWISE_OPS(uint32_t, 8, ^, _mm256_xor_si256)

// Bitwise operations for uint32_t 4-wide
DEFINE_BITWISE_OPS(uint32_t, 4, &, _mm_and_si128)
DEFINE_BITWISE_OPS(uint32_t, 4, |, _mm_or_si128)
DEFINE_BITWISE_OPS(uint32_t, 4, ^, _mm_xor_si128)

// Bitwise operations for int64_t 4-wide
DEFINE_BITWISE_OPS(int64_t, 4, &, _mm256_and_si256)
DEFINE_BITWISE_OPS(int64_t, 4, |, _mm256_or_si256)
DEFINE_BITWISE_OPS(int64_t, 4, ^, _mm256_xor_si256)

// Bitwise operations for uint64_t 4-wide
DEFINE_BITWISE_OPS(uint64_t, 4, &, _mm256_and_si256)
DEFINE_BITWISE_OPS(uint64_t, 4, |, _mm256_or_si256)
DEFINE_BITWISE_OPS(uint64_t, 4, ^, _mm256_xor_si256)

// Bitwise operations for float 8-wide (using cast)
inline Simd<float, 8> operator&(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_and_ps(a.value, b.value));
}

inline Simd<float, 8> operator|(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_or_ps(a.value, b.value));
}

inline Simd<float, 8> operator^(Simd<float, 8> a, Simd<float, 8> b) {
    return Simd<float, 8>(_mm256_xor_ps(a.value, b.value));
}

// Bitwise operations for float 4-wide
inline Simd<float, 4> operator&(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_and_ps(a.value, b.value));
}

inline Simd<float, 4> operator|(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_or_ps(a.value, b.value));
}

inline Simd<float, 4> operator^(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<float, 4>(_mm_xor_ps(a.value, b.value));
}

// Bitwise operations for double 4-wide
inline Simd<double, 4> operator&(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_and_pd(a.value, b.value));
}

inline Simd<double, 4> operator|(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_or_pd(a.value, b.value));
}

inline Simd<double, 4> operator^(Simd<double, 4> a, Simd<double, 4> b) {
    return Simd<double, 4>(_mm256_xor_pd(a.value, b.value));
}

// Bitwise operations for bool 8-wide
DEFINE_BITWISE_OPS(bool, 8, &, _mm256_and_ps)
DEFINE_BITWISE_OPS(bool, 8, |, _mm256_or_ps)
DEFINE_BITWISE_OPS(bool, 8, ^, _mm256_xor_ps)

// Bitwise operations for bool 4-wide
DEFINE_BITWISE_OPS(bool, 4, &, _mm_and_ps)
DEFINE_BITWISE_OPS(bool, 4, |, _mm_or_ps)
DEFINE_BITWISE_OPS(bool, 4, ^, _mm_xor_ps)

// Bitwise NOT operators
inline Simd<int, 8> operator~(Simd<int, 8> a) {
    return Simd<int, 8>(_mm256_xor_si256(a.value, _mm256_set1_epi32(-1)));
}

inline Simd<int, 4> operator~(Simd<int, 4> a) {
    return Simd<int, 4>(_mm_xor_si128(a.value, _mm_set1_epi32(-1)));
}

inline Simd<uint32_t, 8> operator~(Simd<uint32_t, 8> a) {
    return Simd<uint32_t, 8>(_mm256_xor_si256(a.value, _mm256_set1_epi32(-1)));
}

inline Simd<uint32_t, 4> operator~(Simd<uint32_t, 4> a) {
    return Simd<uint32_t, 4>(_mm_xor_si128(a.value, _mm_set1_epi32(-1)));
}

inline Simd<int64_t, 4> operator~(Simd<int64_t, 4> a) {
    return Simd<int64_t, 4>(_mm256_xor_si256(a.value, _mm256_set1_epi64x(-1)));
}

inline Simd<uint64_t, 4> operator~(Simd<uint64_t, 4> a) {
    return Simd<uint64_t, 4>(_mm256_xor_si256(a.value, _mm256_set1_epi64x(-1)));
}

inline Simd<float, 8> operator~(Simd<float, 8> a) {
    return Simd<float, 8>(_mm256_xor_ps(a.value, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
}

inline Simd<float, 4> operator~(Simd<float, 4> a) {
    return Simd<float, 4>(_mm_xor_ps(a.value, _mm_castsi128_ps(_mm_set1_epi32(-1))));
}

inline Simd<double, 4> operator~(Simd<double, 4> a) {
    return Simd<double, 4>(_mm256_xor_pd(a.value, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
}

inline Simd<bool, 8> operator~(Simd<bool, 8> a) {
    return Simd<bool, 8>(_mm256_xor_ps(a.value, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
}

inline Simd<bool, 4> operator~(Simd<bool, 4> a) {
    return Simd<bool, 4>(_mm_xor_ps(a.value, _mm_castsi128_ps(_mm_set1_epi32(-1))));
}

// ====== BITSHIFT OPERATIONS ======

// Define macro for bitshift operations (scalar implementation)
#define DEFINE_BITSHIFT_OP(TYPE, SIZE, OP) \
inline Simd<TYPE, SIZE> operator OP(Simd<TYPE, SIZE> a, int shift) { \
    alignas(32) TYPE a_array[SIZE], result[SIZE]; \
    store(a_array, a); \
    for (int i = 0; i < SIZE; i++) { \
        result[i] = a_array[i] OP shift; \
    } \
    return load<TYPE, SIZE>(result); \
}

// Bitshift for all integer types
DEFINE_BITSHIFT_OP(int, 8, <<)
DEFINE_BITSHIFT_OP(int, 8, >>)
DEFINE_BITSHIFT_OP(int, 4, <<)
DEFINE_BITSHIFT_OP(int, 4, >>)
DEFINE_BITSHIFT_OP(uint32_t, 8, <<)
DEFINE_BITSHIFT_OP(uint32_t, 8, >>)
DEFINE_BITSHIFT_OP(uint32_t, 4, <<)
DEFINE_BITSHIFT_OP(uint32_t, 4, >>)
DEFINE_BITSHIFT_OP(int64_t, 4, <<)
DEFINE_BITSHIFT_OP(int64_t, 4, >>)
DEFINE_BITSHIFT_OP(uint64_t, 4, <<)
DEFINE_BITSHIFT_OP(uint64_t, 4, >>)

// Vector-vector bitshift (allows for variable shifts per lane)
#define DEFINE_VECTOR_BITSHIFT_OP(TYPE, SIZE, OP) \
inline Simd<TYPE, SIZE> operator OP(Simd<TYPE, SIZE> a, Simd<TYPE, SIZE> shift) { \
    alignas(32) TYPE a_array[SIZE], shift_array[SIZE], result[SIZE]; \
    store(a_array, a); \
    store(shift_array, shift); \
    for (int i = 0; i < SIZE; i++) { \
        result[i] = a_array[i] OP shift_array[i]; \
    } \
    return load<TYPE, SIZE>(result); \
}

// Vector-vector bitshift for all integer types
DEFINE_VECTOR_BITSHIFT_OP(int, 8, <<)
DEFINE_VECTOR_BITSHIFT_OP(int, 8, >>)
DEFINE_VECTOR_BITSHIFT_OP(int, 4, <<)
DEFINE_VECTOR_BITSHIFT_OP(int, 4, >>)
DEFINE_VECTOR_BITSHIFT_OP(uint32_t, 8, <<)
DEFINE_VECTOR_BITSHIFT_OP(uint32_t, 8, >>)
DEFINE_VECTOR_BITSHIFT_OP(uint32_t, 4, <<)
DEFINE_VECTOR_BITSHIFT_OP(uint32_t, 4, >>)
DEFINE_VECTOR_BITSHIFT_OP(int64_t, 4, <<)
DEFINE_VECTOR_BITSHIFT_OP(int64_t, 4, >>)
DEFINE_VECTOR_BITSHIFT_OP(uint64_t, 4, <<)
DEFINE_VECTOR_BITSHIFT_OP(uint64_t, 4, >>)

// ====== COMPARISON OPERATIONS ======

// Define macro for comparison operations with floats
#define DEFINE_FLOAT_CMP_OPS(TYPE, SIZE, OP, INSTR, PARAM) \
inline Simd<bool, SIZE> operator OP(Simd<TYPE, SIZE> a, Simd<TYPE, SIZE> b) { \
    return Simd<bool, SIZE>(INSTR(a.value, b.value, PARAM)); \
}

// Comparison for float 8-wide
DEFINE_FLOAT_CMP_OPS(float, 8, ==, _mm256_cmp_ps, _CMP_EQ_OQ)
DEFINE_FLOAT_CMP_OPS(float, 8, !=, _mm256_cmp_ps, _CMP_NEQ_OQ)
DEFINE_FLOAT_CMP_OPS(float, 8, <, _mm256_cmp_ps, _CMP_LT_OQ)
DEFINE_FLOAT_CMP_OPS(float, 8, <=, _mm256_cmp_ps, _CMP_LE_OQ)
DEFINE_FLOAT_CMP_OPS(float, 8, >, _mm256_cmp_ps, _CMP_GT_OQ)
DEFINE_FLOAT_CMP_OPS(float, 8, >=, _mm256_cmp_ps, _CMP_GE_OQ)

// Comparison for float 4-wide
inline Simd<bool, 4> operator==(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmpeq_ps(a.value, b.value));
}

inline Simd<bool, 4> operator!=(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmpneq_ps(a.value, b.value));
}

inline Simd<bool, 4> operator<(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmplt_ps(a.value, b.value));
}

inline Simd<bool, 4> operator<=(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmple_ps(a.value, b.value));
}

inline Simd<bool, 4> operator>(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmpgt_ps(a.value, b.value));
}

inline Simd<bool, 4> operator>=(Simd<float, 4> a, Simd<float, 4> b) {
    return Simd<bool, 4>(_mm_cmpge_ps(a.value, b.value));
}

// Comparison for double 4-wide
inline Simd<bool, 4> operator==(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_EQ_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

inline Simd<bool, 4> operator!=(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_NEQ_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

inline Simd<bool, 4> operator<(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_LT_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

inline Simd<bool, 4> operator<=(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_LE_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

inline Simd<bool, 4> operator>(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_GT_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

inline Simd<bool, 4> operator>=(Simd<double, 4> a, Simd<double, 4> b) {
    __m256d cmp = _mm256_cmp_pd(a.value, b.value, _CMP_GE_OQ);
    __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
    __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
    return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
}

// Comparison for int 8-wide
inline Simd<bool, 8> operator==(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.value, b.value)));
}

inline Simd<bool, 8> operator!=(Simd<int, 8> a, Simd<int, 8> b) {
    __m256i eq = _mm256_cmpeq_epi32(a.value, b.value);
    __m256i neq = _mm256_xor_si256(eq, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(neq));
}

inline Simd<bool, 8> operator<(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpgt_epi32(b.value, a.value)));
}

inline Simd<bool, 8> operator<=(Simd<int, 8> a, Simd<int, 8> b) {
    __m256i gt = _mm256_cmpgt_epi32(a.value, b.value);
    __m256i le = _mm256_xor_si256(gt, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(le));
}

inline Simd<bool, 8> operator>(Simd<int, 8> a, Simd<int, 8> b) {
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.value, b.value)));
}

inline Simd<bool, 8> operator>=(Simd<int, 8> a, Simd<int, 8> b) {
    __m256i lt = _mm256_cmpgt_epi32(b.value, a.value);
    __m256i ge = _mm256_xor_si256(lt, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(ge));
}

// Comparison for int 4-wide
inline Simd<bool, 4> operator==(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<bool, 4>(_mm_castsi128_ps(_mm_cmpeq_epi32(a.value, b.value)));
}

inline Simd<bool, 4> operator!=(Simd<int, 4> a, Simd<int, 4> b) {
    __m128i eq = _mm_cmpeq_epi32(a.value, b.value);
    __m128i neq = _mm_xor_si128(eq, _mm_set1_epi32(-1));
    return Simd<bool, 4>(_mm_castsi128_ps(neq));
}

inline Simd<bool, 4> operator<(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<bool, 4>(_mm_castsi128_ps(_mm_cmplt_epi32(a.value, b.value)));
}

inline Simd<bool, 4> operator<=(Simd<int, 4> a, Simd<int, 4> b) {
    __m128i gt = _mm_cmpgt_epi32(a.value, b.value);
    __m128i le = _mm_xor_si128(gt, _mm_set1_epi32(-1));
    return Simd<bool, 4>(_mm_castsi128_ps(le));
}

inline Simd<bool, 4> operator>(Simd<int, 4> a, Simd<int, 4> b) {
    return Simd<bool, 4>(_mm_castsi128_ps(_mm_cmpgt_epi32(a.value, b.value)));
}

inline Simd<bool, 4> operator>=(Simd<int, 4> a, Simd<int, 4> b) {
    __m128i lt = _mm_cmplt_epi32(a.value, b.value);
    __m128i ge = _mm_xor_si128(lt, _mm_set1_epi32(-1));
    return Simd<bool, 4>(_mm_castsi128_ps(ge));
}

// Comparison for uint32_t 8-wide (need special handling for unsigned)
inline Simd<bool, 8> operator==(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.value, b.value)));
}

inline Simd<bool, 8> operator!=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    __m256i eq = _mm256_cmpeq_epi32(a.value, b.value);
    __m256i neq = _mm256_xor_si256(eq, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(neq));
}

inline Simd<bool, 8> operator<(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    // For unsigned comparisons, handle sign bit differently
    const __m256i sign_bit = _mm256_set1_epi32(INT32_MIN);
    __m256i a_adj = _mm256_xor_si256(a.value, sign_bit);
    __m256i b_adj = _mm256_xor_si256(b.value, sign_bit);
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpgt_epi32(b_adj, a_adj)));
}

inline Simd<bool, 8> operator<=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    // First check if a > b
    const __m256i sign_bit = _mm256_set1_epi32(INT32_MIN);
    __m256i a_adj = _mm256_xor_si256(a.value, sign_bit);
    __m256i b_adj = _mm256_xor_si256(b.value, sign_bit);
    __m256i gt = _mm256_cmpgt_epi32(a_adj, b_adj);
    // Then negate to get a <= b
    __m256i le = _mm256_xor_si256(gt, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(le));
}

inline Simd<bool, 8> operator>(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    const __m256i sign_bit = _mm256_set1_epi32(INT32_MIN);
    __m256i a_adj = _mm256_xor_si256(a.value, sign_bit);
    __m256i b_adj = _mm256_xor_si256(b.value, sign_bit);
    return Simd<bool, 8>(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a_adj, b_adj)));
}

inline Simd<bool, 8> operator>=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    // First check if a < b
    const __m256i sign_bit = _mm256_set1_epi32(INT32_MIN);
    __m256i a_adj = _mm256_xor_si256(a.value, sign_bit);
    __m256i b_adj = _mm256_xor_si256(b.value, sign_bit);
    __m256i lt = _mm256_cmpgt_epi32(b_adj, a_adj);
    // Then negate to get a >= b
    __m256i ge = _mm256_xor_si256(lt, _mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_castsi256_ps(ge));
}

// Comparison for uint32_t 4-wide
inline Simd<bool, 4> operator==(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    return Simd<bool, 4>(_mm_castsi128_ps(_mm_cmpeq_epi32(a.value, b.value)));
}

inline Simd<bool, 4> operator!=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    __m128i eq = _mm_cmpeq_epi32(a.value, b.value);
    __m128i neq = _mm_xor_si128(eq, _mm_set1_epi32(-1));
    return Simd<bool, 4>(_mm_castsi128_ps(neq));
}

// Comparison for int64_t/uint64_t 4-wide (handled specially earlier)

// Comparison for bool types
inline Simd<bool, 8> operator==(Simd<bool, 8> a, Simd<bool, 8> b) {
    // XNOR = NOT (XOR) = ~(a^b)
    __m256 xor_result = _mm256_xor_ps(a.value, b.value);
    __m256 all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    return Simd<bool, 8>(_mm256_xor_ps(xor_result, all_ones));
}

inline Simd<bool, 8> operator!=(Simd<bool, 8> a, Simd<bool, 8> b) {
    return Simd<bool, 8>(_mm256_xor_ps(a.value, b.value));
}

inline Simd<bool, 8> operator<(Simd<bool, 8> a, Simd<bool, 8> b) {
    // For bool, a < b is true when a is false and b is true
    return Simd<bool, 8>(_mm256_andnot_ps(a.value, b.value));
}

inline Simd<bool, 8> operator<=(Simd<bool, 8> a, Simd<bool, 8> b) {
    // a <= b is equivalent to !a || b
    return Simd<bool, 8>(_mm256_or_ps(_mm256_xor_ps(a.value, _mm256_castsi256_ps(_mm256_set1_epi32(-1))), b.value));
}

inline Simd<bool, 8> operator>(Simd<bool, 8> a, Simd<bool, 8> b) {
    // a > b is true when a is true and b is false
    return Simd<bool, 8>(_mm256_andnot_ps(b.value, a.value));
}

inline Simd<bool, 8> operator>=(Simd<bool, 8> a, Simd<bool, 8> b) {
    // a >= b is equivalent to a || !b
    return Simd<bool, 8>(_mm256_or_ps(a.value, _mm256_xor_ps(b.value, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))));
}

// Similar operators for 4-wide bool
inline Simd<bool, 4> operator==(Simd<bool, 4> a, Simd<bool, 4> b) {
    // XNOR = NOT (XOR) = ~(a^b)
    __m128 xor_result = _mm_xor_ps(a.value, b.value);
    __m128 all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
    return Simd<bool, 4>(_mm_xor_ps(xor_result, all_ones));
}

inline Simd<bool, 4> operator!=(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_xor_ps(a.value, b.value));
}

inline Simd<bool, 4> operator<(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_andnot_ps(a.value, b.value));
}

inline Simd<bool, 4> operator<=(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_or_ps(_mm_xor_ps(a.value, _mm_castsi128_ps(_mm_set1_epi32(-1))), b.value));
}

inline Simd<bool, 4> operator>(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_andnot_ps(b.value, a.value));
}

inline Simd<bool, 4> operator>=(Simd<bool, 4> a, Simd<bool, 4> b) {
    return Simd<bool, 4>(_mm_or_ps(a.value, _mm_xor_ps(b.value, _mm_castsi128_ps(_mm_set1_epi32(-1)))));
}

// ====== LOGICAL OPERATIONS ======

// Define logical operations for float 8-wide
inline Simd<bool, 8> operator&&(Simd<float, 8> a, Simd<float, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(a.value, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(b.value, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_and_ps(a_bool, b_bool));
}

inline Simd<bool, 8> operator||(Simd<float, 8> a, Simd<float, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(a.value, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(b.value, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_or_ps(a_bool, b_bool));
}

// Logical operations for float 4-wide
inline Simd<bool, 4> operator&&(Simd<float, 4> a, Simd<float, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(a.value, _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(b.value, _mm_setzero_ps());
    return Simd<bool, 4>(_mm_and_ps(a_bool, b_bool));
}

inline Simd<bool, 4> operator||(Simd<float, 4> a, Simd<float, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(a.value, _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(b.value, _mm_setzero_ps());
    return Simd<bool, 4>(_mm_or_ps(a_bool, b_bool));
}

// Logical operations for int 8-wide
inline Simd<bool, 8> operator&&(Simd<int, 8> a, Simd<int, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(a.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(b.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_and_ps(a_bool, b_bool));
}

inline Simd<bool, 8> operator||(Simd<int, 8> a, Simd<int, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(a.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(b.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_or_ps(a_bool, b_bool));
}

// Logical operations for int 4-wide
inline Simd<bool, 4> operator&&(Simd<int, 4> a, Simd<int, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(a.value), _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(b.value), _mm_setzero_ps());
    return Simd<bool, 4>(_mm_and_ps(a_bool, b_bool));
}

inline Simd<bool, 4> operator||(Simd<int, 4> a, Simd<int, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(a.value), _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(b.value), _mm_setzero_ps());
    return Simd<bool, 4>(_mm_or_ps(a_bool, b_bool));
}

// Logical operations for uint32_t 8-wide
inline Simd<bool, 8> operator&&(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(a.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(b.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_and_ps(a_bool, b_bool));
}

inline Simd<bool, 8> operator||(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
    __m256 a_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(a.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    __m256 b_bool = _mm256_cmp_ps(_mm256_cvtepi32_ps(b.value), _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return Simd<bool, 8>(_mm256_or_ps(a_bool, b_bool));
}

// Logical operations for uint32_t 4-wide
inline Simd<bool, 4> operator&&(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(a.value), _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(b.value), _mm_setzero_ps());
    return Simd<bool, 4>(_mm_and_ps(a_bool, b_bool));
}

inline Simd<bool, 4> operator||(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
    __m128 a_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(a.value), _mm_setzero_ps());
    __m128 b_bool = _mm_cmpneq_ps(_mm_cvtepi32_ps(b.value), _mm_setzero_ps());
    return Simd<bool, 4>(_mm_or_ps(a_bool, b_bool));
}

// Continue from previous code...
    // Logical operations for double 4-wide
    inline Simd<bool, 4> operator&&(Simd<double, 4> a, Simd<double, 4> b) {
        __m256d a_bool = _mm256_cmp_pd(a.value, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        __m256d b_bool = _mm256_cmp_pd(b.value, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        
        // Convert to bool format
        __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(a_bool, 0));
        __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(a_bool, 1));
        __m128 a_ps = _mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0));
        
        low = _mm_castpd_ps(_mm256_extractf128_pd(b_bool, 0));
        high = _mm_castpd_ps(_mm256_extractf128_pd(b_bool, 1));
        __m128 b_ps = _mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0));
        
        return Simd<bool, 4>(_mm_and_ps(a_ps, b_ps));
    }

    inline Simd<bool, 4> operator||(Simd<double, 4> a, Simd<double, 4> b) {
        __m256d a_bool = _mm256_cmp_pd(a.value, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        __m256d b_bool = _mm256_cmp_pd(b.value, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        
        // Convert to bool format
        __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(a_bool, 0));
        __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(a_bool, 1));
        __m128 a_ps = _mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0));
        
        low = _mm_castpd_ps(_mm256_extractf128_pd(b_bool, 0));
        high = _mm_castpd_ps(_mm256_extractf128_pd(b_bool, 1));
        __m128 b_ps = _mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0));
        
        return Simd<bool, 4>(_mm_or_ps(a_ps, b_ps));
    }

    // Logical operations for int64_t/uint64_t 4-wide (scalar implementation due to missing SIMD ops)
    inline Simd<bool, 4> operator&&(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result_array[i] = (a_array[i] != 0 && b_array[i] != 0) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result_array));
    }

    inline Simd<bool, 4> operator||(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result_array[i] = (a_array[i] != 0 || b_array[i] != 0) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result_array));
    }

    inline Simd<bool, 4> operator&&(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result_array[i] = (a_array[i] != 0 && b_array[i] != 0) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result_array));
    }

    inline Simd<bool, 4> operator||(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result_array[i] = (a_array[i] != 0 || b_array[i] != 0) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result_array));
    }

    // Logical operations for bool types
    inline Simd<bool, 8> operator&&(Simd<bool, 8> a, Simd<bool, 8> b) {
        return Simd<bool, 8>(_mm256_and_ps(a.value, b.value));
    }

    inline Simd<bool, 8> operator||(Simd<bool, 8> a, Simd<bool, 8> b) {
        return Simd<bool, 8>(_mm256_or_ps(a.value, b.value));
    }

    inline Simd<bool, 8> operator!(Simd<bool, 8> a) {
        return Simd<bool, 8>(_mm256_xor_ps(a.value, _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF))));
    }

    inline Simd<bool, 4> operator&&(Simd<bool, 4> a, Simd<bool, 4> b) {
        return Simd<bool, 4>(_mm_and_ps(a.value, b.value));
    }

    inline Simd<bool, 4> operator||(Simd<bool, 4> a, Simd<bool, 4> b) {
        return Simd<bool, 4>(_mm_or_ps(a.value, b.value));
    }

    inline Simd<bool, 4> operator!(Simd<bool, 4> a) {
        return Simd<bool, 4>(_mm_xor_ps(a.value, _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFF))));
    }

    // Logical NOT for int (8-wide)
    inline Simd<int, 8> operator!(Simd<int, 8> a) {
        // Compare with zero to get a boolean result, then convert to int
        auto is_zero = (a == Simd<int, 8>(0));
        // Convert bool to int (true -> 1, false -> 0)
        alignas(32) float bool_vals[8];
        alignas(32) int result[8];
        _mm256_store_ps(bool_vals, is_zero.value);
        for (int i = 0; i < 8; i++) {
            result[i] = (bool_vals[i] != 0.0f) ? 1 : 0;
        }
        return load<int, 8>(result);
    }

    // Logical NOT for int (4-wide)
    inline Simd<int, 4> operator!(Simd<int, 4> a) {
        auto is_zero = (a == Simd<int, 4>(0));
        alignas(16) float bool_vals[4];
        alignas(16) int result[4];
        _mm_store_ps(bool_vals, is_zero.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (bool_vals[i] != 0.0f) ? 1 : 0;
        }
        return load<int, 4>(result);
    }

    // Logical NOT for uint32_t (8-wide)
    inline Simd<uint32_t, 8> operator!(Simd<uint32_t, 8> a) {
        auto is_zero = (a == Simd<uint32_t, 8>(0));
        alignas(32) float bool_vals[8];
        alignas(32) uint32_t result[8];
        _mm256_store_ps(bool_vals, is_zero.value);
        for (int i = 0; i < 8; i++) {
            result[i] = (bool_vals[i] != 0.0f) ? 1 : 0;
        }
        return load<uint32_t, 8>(result);
    }

    // Logical NOT for uint32_t (4-wide)
    inline Simd<uint32_t, 4> operator!(Simd<uint32_t, 4> a) {
        auto is_zero = (a == Simd<uint32_t, 4>(0));
        alignas(16) float bool_vals[4];
        alignas(16) uint32_t result[4];
        _mm_store_ps(bool_vals, is_zero.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (bool_vals[i] != 0.0f) ? 1 : 0;
        }
        return load<uint32_t, 4>(result);
    }

    // Logical NOT for int64_t (4-wide)
    inline Simd<int64_t, 4> operator!(Simd<int64_t, 4> a) {
        alignas(32) int64_t values[4], result[4];
        _mm256_store_si256((__m256i*)values, a.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (values[i] == 0) ? 1 : 0;
        }
        return load<int64_t, 4>(result);
    }

    // Logical NOT for uint64_t (4-wide)
    inline Simd<uint64_t, 4> operator!(Simd<uint64_t, 4> a) {
        alignas(32) uint64_t values[4], result[4];
        _mm256_store_si256((__m256i*)values, a.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (values[i] == 0) ? 1 : 0;
        }
        return load<uint64_t, 4>(result);
    }

    // Logical NOT for double (4-wide)
    inline Simd<double, 4> operator!(Simd<double, 4> a) {
        alignas(32) double values[4], result[4];
        _mm256_store_pd(values, a.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (values[i] == 0.0) ? 1.0 : 0.0;
        }
        return load<double, 4>(result);
    }

    // ====== MATH FUNCTIONS ======

    // Unary math functions for float 8-wide - some can use AVX instructions directly
    inline Simd<float, 8> sqrt(Simd<float, 8> a) {
        return Simd<float, 8>(_mm256_sqrt_ps(a.value));
    }

    inline Simd<float, 8> abs(Simd<float, 8> a) {
        // Clear the sign bit
        __m256 sign_mask = _mm256_set1_ps(-0.0f);  // 0x80000000
        return Simd<float, 8>(_mm256_andnot_ps(sign_mask, a.value));
    }

    inline Simd<float, 8> floor(Simd<float, 8> a) {
        return Simd<float, 8>(_mm256_floor_ps(a.value));
    }

    inline Simd<float, 8> ceil(Simd<float, 8> a) {
        return Simd<float, 8>(_mm256_ceil_ps(a.value));
    }

    inline Simd<float, 8> round(Simd<float, 8> a) {
        return Simd<float, 8>(_mm256_round_ps(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // Helper macro for scalar math functions without direct AVX implementation
    #define DEFINE_SCALAR_MATH_FUNC(FUNC, TYPE, SIZE) \
    inline Simd<TYPE, SIZE> FUNC(Simd<TYPE, SIZE> a) { \
        alignas(32) TYPE a_array[SIZE], result[SIZE]; \
        store(a_array, a); \
        for (int i = 0; i < SIZE; i++) { \
            result[i] = std::FUNC(a_array[i]); \
        } \
        return load<TYPE, SIZE>(result); \
    }

    // Common scalar math functions for float 8-wide
    DEFINE_SCALAR_MATH_FUNC(exp, float, 8)
    DEFINE_SCALAR_MATH_FUNC(log, float, 8)
    DEFINE_SCALAR_MATH_FUNC(log10, float, 8)
    DEFINE_SCALAR_MATH_FUNC(log2, float, 8)
    DEFINE_SCALAR_MATH_FUNC(log1p, float, 8)
    DEFINE_SCALAR_MATH_FUNC(sin, float, 8)
    DEFINE_SCALAR_MATH_FUNC(cos, float, 8)
    DEFINE_SCALAR_MATH_FUNC(tan, float, 8)
    DEFINE_SCALAR_MATH_FUNC(asin, float, 8)
    DEFINE_SCALAR_MATH_FUNC(acos, float, 8)
    DEFINE_SCALAR_MATH_FUNC(atan, float, 8)
    DEFINE_SCALAR_MATH_FUNC(sinh, float, 8)
    DEFINE_SCALAR_MATH_FUNC(cosh, float, 8)
    DEFINE_SCALAR_MATH_FUNC(tanh, float, 8)
    DEFINE_SCALAR_MATH_FUNC(expm1, float, 8)
    DEFINE_SCALAR_MATH_FUNC(asinh, float, 8)
    DEFINE_SCALAR_MATH_FUNC(acosh, float, 8)
    DEFINE_SCALAR_MATH_FUNC(atanh, float, 8)

    // Unary math functions for float 4-wide
    inline Simd<float, 4> sqrt(Simd<float, 4> a) {
        return Simd<float, 4>(_mm_sqrt_ps(a.value));
    }

    inline Simd<float, 4> abs(Simd<float, 4> a) {
        __m128 sign_mask = _mm_set1_ps(-0.0f);
        return Simd<float, 4>(_mm_andnot_ps(sign_mask, a.value));
    }

    // Math functions for float 4-wide
    DEFINE_SCALAR_MATH_FUNC(floor, float, 4)
    DEFINE_SCALAR_MATH_FUNC(ceil, float, 4)
    DEFINE_SCALAR_MATH_FUNC(round, float, 4)
    DEFINE_SCALAR_MATH_FUNC(exp, float, 4)
    DEFINE_SCALAR_MATH_FUNC(log, float, 4)
    DEFINE_SCALAR_MATH_FUNC(log10, float, 4)
    DEFINE_SCALAR_MATH_FUNC(log2, float, 4)
    DEFINE_SCALAR_MATH_FUNC(log1p, float, 4)
    DEFINE_SCALAR_MATH_FUNC(sin, float, 4)
    DEFINE_SCALAR_MATH_FUNC(cos, float, 4)
    DEFINE_SCALAR_MATH_FUNC(tan, float, 4)
    DEFINE_SCALAR_MATH_FUNC(asin, float, 4)
    DEFINE_SCALAR_MATH_FUNC(acos, float, 4)
    DEFINE_SCALAR_MATH_FUNC(atan, float, 4)
    DEFINE_SCALAR_MATH_FUNC(sinh, float, 4)
    DEFINE_SCALAR_MATH_FUNC(cosh, float, 4)
    DEFINE_SCALAR_MATH_FUNC(tanh, float, 4)
    DEFINE_SCALAR_MATH_FUNC(expm1, float, 4)
    DEFINE_SCALAR_MATH_FUNC(asinh, float, 4)
    DEFINE_SCALAR_MATH_FUNC(acosh, float, 4)
    DEFINE_SCALAR_MATH_FUNC(atanh, float, 4)

    // Math functions for double 4-wide - some can use AVX instructions directly
    inline Simd<double, 4> sqrt(Simd<double, 4> a) {
        return Simd<double, 4>(_mm256_sqrt_pd(a.value));
    }

    inline Simd<double, 4> abs(Simd<double, 4> a) {
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        return Simd<double, 4>(_mm256_andnot_pd(sign_mask, a.value));
    }

    inline Simd<double, 4> floor(Simd<double, 4> a) {
        return Simd<double, 4>(_mm256_floor_pd(a.value));
    }

    inline Simd<double, 4> ceil(Simd<double, 4> a) {
        return Simd<double, 4>(_mm256_ceil_pd(a.value));
    }

    inline Simd<double, 4> round(Simd<double, 4> a) {
        return Simd<double, 4>(_mm256_round_pd(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // Common scalar math functions for double 4-wide
    DEFINE_SCALAR_MATH_FUNC(exp, double, 4)
    DEFINE_SCALAR_MATH_FUNC(log, double, 4)
    DEFINE_SCALAR_MATH_FUNC(log10, double, 4)
    DEFINE_SCALAR_MATH_FUNC(log2, double, 4)
    DEFINE_SCALAR_MATH_FUNC(log1p, double, 4)
    DEFINE_SCALAR_MATH_FUNC(sin, double, 4)
    DEFINE_SCALAR_MATH_FUNC(cos, double, 4)
    DEFINE_SCALAR_MATH_FUNC(tan, double, 4)
    DEFINE_SCALAR_MATH_FUNC(asin, double, 4)
    DEFINE_SCALAR_MATH_FUNC(acos, double, 4)
    DEFINE_SCALAR_MATH_FUNC(atan, double, 4)
    DEFINE_SCALAR_MATH_FUNC(sinh, double, 4)
    DEFINE_SCALAR_MATH_FUNC(cosh, double, 4)
    DEFINE_SCALAR_MATH_FUNC(tanh, double, 4)
    DEFINE_SCALAR_MATH_FUNC(expm1, double, 4)
    DEFINE_SCALAR_MATH_FUNC(asinh, double, 4)
    DEFINE_SCALAR_MATH_FUNC(acosh, double, 4)
    DEFINE_SCALAR_MATH_FUNC(atanh, double, 4)

    // Math functions for integer types
    inline Simd<int, 8> abs(Simd<int, 8> a) {
        return Simd<int, 8>(_mm256_abs_epi32(a.value));
    }

    inline Simd<int, 4> abs(Simd<int, 4> a) {
        return Simd<int, 4>(_mm_abs_epi32(a.value));
    }

    // Abs for uint32_t is a no-op (already positive)
    inline Simd<uint32_t, 8> abs(Simd<uint32_t, 8> a) {
        return a;
    }

    inline Simd<uint32_t, 4> abs(Simd<uint32_t, 4> a) {
        return a;
    }

    // Abs for int64_t and uint64_t
    inline Simd<int64_t, 4> abs(Simd<int64_t, 4> a) {
        alignas(32) int64_t a_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::abs(a_array[i]);
        }
        return load<int64_t, 4>(result);
    }

    inline Simd<uint64_t, 4> abs(Simd<uint64_t, 4> a) {
        return a; // No-op for unsigned
    }

    // isnan implementation
    inline Simd<bool, 8> isnan(Simd<float, 8> a) {
        return Simd<bool, 8>(_mm256_cmp_ps(a.value, a.value, _CMP_UNORD_Q));
    }

    inline Simd<bool, 4> isnan(Simd<float, 4> a) {
        return Simd<bool, 4>(_mm_cmpunord_ps(a.value, a.value));
    }

    inline Simd<bool, 4> isnan(Simd<double, 4> a) {
        __m256d cmp = _mm256_cmp_pd(a.value, a.value, _CMP_UNORD_Q);
        __m128 low = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 0));
        __m128 high = _mm_castpd_ps(_mm256_extractf128_pd(cmp, 1));
        return Simd<bool, 4>(_mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)));
    }

    // Binary math functions with two vector arguments
    inline Simd<float, 8> atan2(Simd<float, 8> y, Simd<float, 8> x) {
        alignas(32) float y_array[8], x_array[8], result[8];
        _mm256_store_ps(y_array, y.value);
        _mm256_store_ps(x_array, x.value);
        for (int i = 0; i < 8; i++) {
            result[i] = std::atan2(y_array[i], x_array[i]);
        }
        return load<float, 8>(result);
    }

    inline Simd<double, 4> atan2(Simd<double, 4> y, Simd<double, 4> x) {
        alignas(32) double y_array[4], x_array[4], result[4];
        _mm256_store_pd(y_array, y.value);
        _mm256_store_pd(x_array, x.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::atan2(y_array[i], x_array[i]);
        }
        return load<double, 4>(result);
    }

    inline Simd<float, 8> pow(Simd<float, 8> a, Simd<float, 8> b) {
        alignas(32) float a_array[8], b_array[8], result[8];
        _mm256_store_ps(a_array, a.value);
        _mm256_store_ps(b_array, b.value);
        for (int i = 0; i < 8; i++) {
            result[i] = std::pow(a_array[i], b_array[i]);
        }
        return load<float, 8>(result);
    }

    inline Simd<double, 4> pow(Simd<double, 4> a, Simd<double, 4> b) {
        alignas(32) double a_array[4], b_array[4], result[4];
        _mm256_store_pd(a_array, a.value);
        _mm256_store_pd(b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::pow(a_array[i], b_array[i]);
        }
        return load<double, 4>(result);
    }

    // Integer pow implementations
    inline Simd<int, 8> pow(Simd<int, 8> a, Simd<int, 8> b) {
        alignas(32) int a_array[8], b_array[8], result[8];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 8; i++) {
            int base = a_array[i];
            int exp = b_array[i];
            int res = 1;
            while (exp > 0) {
                if (exp & 1) res *= base;
                exp >>= 1;
                base *= base;
            }
            result[i] = res;
        }
        return load<int, 8>(result);
    }

    // Reciprocal square root
    inline Simd<float, 8> rsqrt(Simd<float, 8> x) {
        return Simd<float, 8>(_mm256_rsqrt_ps(x.value));
    }

    inline Simd<float, 4> rsqrt(Simd<float, 4> x) {
        return Simd<float, 4>(_mm_rsqrt_ps(x.value));
    }

    inline Simd<double, 4> rsqrt(Simd<double, 4> x) {
        // No direct AVX instruction for double rsqrt
        return Simd<double, 4>(1.0) / sqrt(x);
    }

    // Reciprocal
    inline Simd<float, 8> recip(Simd<float, 8> x) {
        return Simd<float, 8>(_mm256_rcp_ps(x.value));
    }

    inline Simd<float, 4> recip(Simd<float, 4> x) {
        return Simd<float, 4>(_mm_rcp_ps(x.value));
    }

    inline Simd<double, 4> recip(Simd<double, 4> x) {
        // No direct AVX instruction for double recip
        return Simd<double, 4>(1.0) / x;
    }

    // Round to nearest integer
    inline Simd<float, 8> rint(Simd<float, 8> x) {
        return round(x);  // Using your existing round implementation
    }

    inline Simd<float, 4> rint(Simd<float, 4> x) {
        return round(x);  // Using your existing round implementation
    }

    inline Simd<double, 4> rint(Simd<double, 4> x) {
        return round(x);  // Using your existing round implementation
    }

    // Remainder operation
    inline Simd<float, 8> remainder(Simd<float, 8> a, Simd<float, 8> b) {
        alignas(32) float a_array[8], b_array[8], result[8];
        _mm256_store_ps(a_array, a.value);
        _mm256_store_ps(b_array, b.value);
        for (int i = 0; i < 8; i++) {
            result[i] = std::remainder(a_array[i], b_array[i]);
        }
        return load<float, 8>(result);
    }

    inline Simd<float, 4> remainder(Simd<float, 4> a, Simd<float, 4> b) {
        alignas(16) float a_array[4], b_array[4], result[4];
        _mm_store_ps(a_array, a.value);
        _mm_store_ps(b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::remainder(a_array[i], b_array[i]);
        }
        return load<float, 4>(result);
    }

    inline Simd<double, 4> remainder(Simd<double, 4> a, Simd<double, 4> b) {
        alignas(32) double a_array[4], b_array[4], result[4];
        _mm256_store_pd(a_array, a.value);
        _mm256_store_pd(b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::remainder(a_array[i], b_array[i]);
        }
        return load<double, 4>(result);
    }

    inline Simd<int, 8> remainder(Simd<int, 8> a, Simd<int, 8> b) {
        alignas(32) int a_array[8], b_array[8], result[8];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 8; i++) {
            // Integer remainder is modulo
            result[i] = a_array[i] % b_array[i];
        }
        return load<int, 8>(result);
    }

    inline Simd<int, 4> remainder(Simd<int, 4> a, Simd<int, 4> b) {
        alignas(16) int a_array[4], b_array[4], result[4];
        _mm_store_si128((__m128i*)a_array, a.value);
        _mm_store_si128((__m128i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = a_array[i] % b_array[i];
        }
        return load<int, 4>(result);
    }

    // Remainder implementation for uint32_t (8-wide)
    inline Simd<uint32_t, 8> remainder(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
        alignas(32) uint32_t a_array[8], b_array[8], result[8];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 8; i++) {
            // For unsigned integers, remainder is just modulo
            result[i] = (b_array[i] != 0) ? a_array[i] % b_array[i] : 0;
        }
        return load<uint32_t, 8>(result);
    }

    // Remainder implementation for uint32_t (4-wide)
    inline Simd<uint32_t, 4> remainder(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
        alignas(16) uint32_t a_array[4], b_array[4], result[4];
        _mm_store_si128((__m128i*)a_array, a.value);
        _mm_store_si128((__m128i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (b_array[i] != 0) ? a_array[i] % b_array[i] : 0;
        }
        return load<uint32_t, 4>(result);
    }

    // Remainder implementation for int64_t (4-wide)
    inline Simd<int64_t, 4> remainder(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            // Check for division by zero
            result[i] = (b_array[i] != 0) ? a_array[i] % b_array[i] : 0;
        }
        return load<int64_t, 4>(result);
    }

    // Remainder implementation for uint64_t (4-wide)
    inline Simd<uint64_t, 4> remainder(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (b_array[i] != 0) ? a_array[i] % b_array[i] : 0;
        }
        return load<uint64_t, 4>(result);
    }

    // Remainder implementation for bool (8-wide)
    inline Simd<bool, 8> remainder(Simd<bool, 8> a, Simd<bool, 8> b) {
        // For booleans, remainder doesn't make much sense, but we can define
        // it as: a % b = a when b is true, 0 when b is false
        return a & b;
    }

    // Remainder implementation for bool (4-wide)
    inline Simd<bool, 4> remainder(Simd<bool, 4> a, Simd<bool, 4> b) {
        return a & b;
    }

    // Comparison operators for int64_t (4-wide)
    inline Simd<bool, 4> operator==(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] == b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator!=(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] != b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator<(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] < b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator<=(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] <= b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator>(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] > b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator>=(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] >= b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    // Comparison operators for uint64_t (4-wide)
    inline Simd<bool, 4> operator==(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] == b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator!=(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] != b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator<(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] < b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator<=(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] <= b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator>(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] > b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    inline Simd<bool, 4> operator>=(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4];
        alignas(16) float result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (a_array[i] >= b_array[i]) ? -1.0f : 0.0f;
        }
        return Simd<bool, 4>(_mm_loadu_ps(result));
    }

    // ====== Min/Max Functions ======

    // min/max for float 8-wide
    inline Simd<float, 8> minimum(Simd<float, 8> a, Simd<float, 8> b) {
        return Simd<float, 8>(_mm256_min_ps(a.value, b.value));
    }

    inline Simd<float, 8> maximum(Simd<float, 8> a, Simd<float, 8> b) {
        return Simd<float, 8>(_mm256_max_ps(a.value, b.value));
    }

    // min/max for float 4-wide
    inline Simd<float, 4> minimum(Simd<float, 4> a, Simd<float, 4> b) {
        return Simd<float, 4>(_mm_min_ps(a.value, b.value));
    }

    inline Simd<float, 4> maximum(Simd<float, 4> a, Simd<float, 4> b) {
        return Simd<float, 4>(_mm_max_ps(a.value, b.value));
    }

    // min/max for double 4-wide
    inline Simd<double, 4> minimum(Simd<double, 4> a, Simd<double, 4> b) {
        return Simd<double, 4>(_mm256_min_pd(a.value, b.value));
    }

    inline Simd<double, 4> maximum(Simd<double, 4> a, Simd<double, 4> b) {
        return Simd<double, 4>(_mm256_max_pd(a.value, b.value));
    }

    // min/max for int 8-wide
    inline Simd<int, 8> minimum(Simd<int, 8> a, Simd<int, 8> b) {
        return Simd<int, 8>(_mm256_min_epi32(a.value, b.value));
    }

    inline Simd<int, 8> maximum(Simd<int, 8> a, Simd<int, 8> b) {
        return Simd<int, 8>(_mm256_max_epi32(a.value, b.value));
    }

    // min/max for int 4-wide
    inline Simd<int, 4> minimum(Simd<int, 4> a, Simd<int, 4> b) {
        return Simd<int, 4>(_mm_min_epi32(a.value, b.value));
    }

    inline Simd<int, 4> maximum(Simd<int, 4> a, Simd<int, 4> b) {
        return Simd<int, 4>(_mm_max_epi32(a.value, b.value));
    }

    // min/max for uint32_t 8-wide
    inline Simd<uint32_t, 8> minimum(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
        return Simd<uint32_t, 8>(_mm256_min_epu32(a.value, b.value));
    }

    inline Simd<uint32_t, 8> maximum(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
        return Simd<uint32_t, 8>(_mm256_max_epu32(a.value, b.value));
    }

    // min/max for uint32_t 4-wide
    inline Simd<uint32_t, 4> minimum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
        return Simd<uint32_t, 4>(_mm_min_epu32(a.value, b.value));
    }

    inline Simd<uint32_t, 4> maximum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
        return Simd<uint32_t, 4>(_mm_max_epu32(a.value, b.value));
    }

    // min/max for bool
    inline Simd<bool, 8> maximum(Simd<bool, 8> a, Simd<bool, 8> b) {
        // For bool, maximum is equivalent to OR
        return Simd<bool, 8>(_mm256_or_ps(a.value, b.value));
    }

    inline Simd<bool, 4> maximum(Simd<bool, 4> a, Simd<bool, 4> b) {
        return Simd<bool, 4>(_mm_or_ps(a.value, b.value));
    }

    inline Simd<bool, 8> minimum(Simd<bool, 8> a, Simd<bool, 8> b) {
        // For bool, minimum is equivalent to AND
        return Simd<bool, 8>(_mm256_and_ps(a.value, b.value));
    }

    inline Simd<bool, 4> minimum(Simd<bool, 4> a, Simd<bool, 4> b) {
        return Simd<bool, 4>(_mm_and_ps(a.value, b.value));
    }

    // Maximum and minimum for int64_t (4-wide)
    inline Simd<int64_t, 4> maximum(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::max(a_array[i], b_array[i]);
        }
        return load<int64_t, 4>(result);
    }

    inline Simd<int64_t, 4> minimum(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::min(a_array[i], b_array[i]);
        }
        return load<int64_t, 4>(result);
    }

    // Maximum and minimum for uint64_t (4-wide)
    inline Simd<uint64_t, 4> maximum(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::max(a_array[i], b_array[i]);
        }
        return load<uint64_t, 4>(result);
    }

    inline Simd<uint64_t, 4> minimum(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::min(a_array[i], b_array[i]);
        }
        return load<uint64_t, 4>(result);
    }

    // ====== UTILITY FUNCTIONS ======

    // Add a special template specialization for unsigned types
    template <typename T, int N, typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
    inline Simd<T, N> sign(Simd<T, N> x) {
        // For unsigned types, sign is 0 if x is 0, otherwise 1
        auto zero = Simd<T, N>{0};
        auto one = Simd<T, N>{1};
        return select(x == zero, zero, one);
    }

    // Special case for bool
    template <int N>
    inline Simd<bool, N> sign(Simd<bool, N> x) {
        // For bool, sign is the same as the value
        return x;
    }

    // Add specialized versions for bool
    // template <int N>
    // inline Simd<bool, N> operator<<(Simd<bool, N> a, Simd<bool, N> b) {
    //     // Convert to int, perform shift, then convert back
    //     auto int_a = Simd<int, N>(a);
    //     auto int_b = Simd<int, N>(b);
    //     auto result = int_a << int_b;
    //     return Simd<bool, N>(result != 0);
    // }

    // template <int N>
    // inline Simd<bool, N> operator>>(Simd<bool, N> a, Simd<bool, N> b) {
    //     auto int_a = Simd<int, N>(a);
    //     auto int_b = Simd<int, N>(b);
    //     auto result = int_a >> int_b;
    //     return Simd<bool, N>(result != 0);
    // }

    // Left shift (<<) for bool 4-wide
    inline Simd<bool, 4> operator<<(Simd<bool, 4> a, Simd<bool, 4> b) {
        // Convert to integer first
        Simd<int, 4> a_int(a);
        Simd<int, 4> b_int(b);
        // Perform the shift
        Simd<int, 4> result = a_int << b_int;
        // Convert back to bool (non-zero becomes true)
        return Simd<bool, 4>(result != Simd<int, 4>(0));
    }

    // Right shift (>>) for bool 4-wide
    inline Simd<bool, 4> operator>>(Simd<bool, 4> a, Simd<bool, 4> b) {
        Simd<int, 4> a_int(a);
        Simd<int, 4> b_int(b);
        Simd<int, 4> result = a_int >> b_int;
        return Simd<bool, 4>(result != Simd<int, 4>(0));
    }

    // Left shift (<<) for bool 8-wide
    inline Simd<bool, 8> operator<<(Simd<bool, 8> a, Simd<bool, 8> b) {
        Simd<int, 8> a_int(a);
        Simd<int, 8> b_int(b);
        Simd<int, 8> result = a_int << b_int;
        return Simd<bool, 8>(result != Simd<int, 8>(0));
    }

    // Right shift (>>) for bool 8-wide
    inline Simd<bool, 8> operator>>(Simd<bool, 8> a, Simd<bool, 8> b) {
        Simd<int, 8> a_int(a);
        Simd<int, 8> b_int(b);
        Simd<int, 8> result = a_int >> b_int;
        return Simd<bool, 8>(result != Simd<int, 8>(0));
    }

    // Logical NOT for float (8-wide)
    inline Simd<bool, 8> operator!(Simd<float, 8> a) {
        return Simd<bool, 8>(_mm256_cmp_ps(a.value, _mm256_setzero_ps(), _CMP_EQ_OQ));
    }

    // Logical NOT for float (4-wide)
    inline Simd<bool, 4> operator!(Simd<float, 4> a) {
        return Simd<bool, 4>(_mm_cmpeq_ps(a.value, _mm_setzero_ps()));
    }

    // Add explicit specialization for bool
    // For pow with bool (explicitly for sizes 4 and 8 only)
    inline Simd<bool, 4> pow(Simd<bool, 4> a, Simd<bool, 4> b) {
        // Implementation that uses a & b
        return a & b;
    }

    inline Simd<bool, 8> pow(Simd<bool, 8> a, Simd<bool, 8> b) {
        // Implementation that uses a & b
        return a & b;
    }

    // For pow with uint32_t (explicitly for sizes 4 and 8 only)
    inline Simd<uint32_t, 4> pow(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
        // Implementation from your current code
        alignas(16) uint32_t a_array[4], b_array[4], result[4];
        store(a_array, a);
        store(b_array, b);
        for (int i = 0; i < 4; i++) {
            uint32_t base = a_array[i];
            uint32_t exp = b_array[i];
            uint32_t res = 1;
            while (exp > 0) {
                if (exp & 1) res *= base;
                exp >>= 1;
                base *= base;
            }
            result[i] = res;
        }
        return load<uint32_t, 4>(result);
    }

    inline Simd<uint32_t, 8> pow(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
        // Same implementation but for 8-wide
        alignas(32) uint32_t a_array[8], b_array[8], result[8];
        store(a_array, a);
        store(b_array, b);
        for (int i = 0; i < 8; i++) {
            uint32_t base = a_array[i];
            uint32_t exp = b_array[i];
            uint32_t res = 1;
            while (exp > 0) {
                if (exp & 1) res *= base;
                exp >>= 1;
                base *= base;
            }
            result[i] = res;
        }
        return load<uint32_t, 8>(result);
    }

    // Power implementation for int64_t (4-wide)
    inline Simd<int64_t, 4> pow(Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            int64_t base = a_array[i];
            int64_t exp = b_array[i];
            int64_t res = 1;
            // Handle negative exponents for int by setting to 0
            if (exp < 0) {
                result[i] = 0;
                continue;
            }
            while (exp > 0) {
                if (exp & 1) res *= base;
                exp >>= 1;
                if (exp > 0) base *= base; // Only multiply base if we'll use it again
            }
            result[i] = res;
        }
        return load<int64_t, 4>(result);
    }

    // Power implementation for uint64_t (4-wide)
    inline Simd<uint64_t, 4> pow(Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4], result[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        for (int i = 0; i < 4; i++) {
            uint64_t base = a_array[i];
            uint64_t exp = b_array[i];
            uint64_t res = 1;
            while (exp > 0) {
                if (exp & 1) res *= base;
                exp >>= 1;
                if (exp > 0) base *= base; // Only multiply base if we'll use it again
            }
            result[i] = res;
        }
        return load<uint64_t, 4>(result);
    }

    // Select implementation - conditional function based on mask
    template <typename T>
    inline Simd<T, 8> select(Simd<bool, 8> mask, Simd<T, 8> a, Simd<T, 8> b);

    // Specialization for float 8-wide
    template <>
    inline Simd<float, 8> select(Simd<bool, 8> mask, Simd<float, 8> a, Simd<float, 8> b) {
        return Simd<float, 8>(_mm256_blendv_ps(b.value, a.value, mask.value));
    }

    // Specialization for int 8-wide
    template <>
    inline Simd<int, 8> select(Simd<bool, 8> mask, Simd<int, 8> a, Simd<int, 8> b) {
        return Simd<int, 8>(_mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(b.value), _mm256_castsi256_ps(a.value), mask.value)
        ));
    }

    // Specialization for uint32_t 8-wide
    template <>
    inline Simd<uint32_t, 8> select(Simd<bool, 8> mask, Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
        return Simd<uint32_t, 8>(_mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(b.value), _mm256_castsi256_ps(a.value), mask.value)
        ));
    }

    // Specialization for bool 8-wide
    template <>
    inline Simd<bool, 8> select(Simd<bool, 8> mask, Simd<bool, 8> a, Simd<bool, 8> b) {
        return Simd<bool, 8>(_mm256_blendv_ps(b.value, a.value, mask.value));
    }

    // Select for 4-wide types
    template <typename T>
    inline Simd<T, 4> select(Simd<bool, 4> mask, Simd<T, 4> a, Simd<T, 4> b);

    // Specialization for float 4-wide
    template <>
    inline Simd<float, 4> select(Simd<bool, 4> mask, Simd<float, 4> a, Simd<float, 4> b) {
        return Simd<float, 4>(_mm_blendv_ps(b.value, a.value, mask.value));
    }

    // Specialization for int 4-wide
    template <>
    inline Simd<int, 4> select(Simd<bool, 4> mask, Simd<int, 4> a, Simd<int, 4> b) {
        return Simd<int, 4>(_mm_castps_si128(
            _mm_blendv_ps(_mm_castsi128_ps(b.value), _mm_castsi128_ps(a.value), mask.value)
        ));
    }

    // Specialization for uint32_t 4-wide
    template <>
    inline Simd<uint32_t, 4> select(Simd<bool, 4> mask, Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
        return Simd<uint32_t, 4>(_mm_castps_si128(
            _mm_blendv_ps(_mm_castsi128_ps(b.value), _mm_castsi128_ps(a.value), mask.value)
        ));
    }

    // Specialization for double 4-wide
    template <>
    inline Simd<double, 4> select(Simd<bool, 4> mask, Simd<double, 4> a, Simd<double, 4> b) {
        __m256d mask_pd = _mm256_castps_pd(_mm256_insertf128_ps(_mm256_castps128_ps256(mask.value), mask.value, 1));
        return Simd<double, 4>(_mm256_blendv_pd(b.value, a.value, mask_pd));
    }

    // Specialization for int64_t 4-wide (no direct AVX blend for 64-bit integers)
    template <>
    inline Simd<int64_t, 4> select(Simd<bool, 4> mask, Simd<int64_t, 4> a, Simd<int64_t, 4> b) {
        alignas(32) int64_t a_array[4], b_array[4], result[4];
        alignas(16) float mask_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        _mm_store_ps(mask_array, mask.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (mask_array[i] != 0.0f) ? a_array[i] : b_array[i];
        }
        return load<int64_t, 4>(result);
    }

    // Specialization for uint64_t 4-wide
    template <>
    inline Simd<uint64_t, 4> select(Simd<bool, 4> mask, Simd<uint64_t, 4> a, Simd<uint64_t, 4> b) {
        alignas(32) uint64_t a_array[4], b_array[4], result[4];
        alignas(16) float mask_array[4];
        _mm256_store_si256((__m256i*)a_array, a.value);
        _mm256_store_si256((__m256i*)b_array, b.value);
        _mm_store_ps(mask_array, mask.value);
        for (int i = 0; i < 4; i++) {
            result[i] = (mask_array[i] != 0.0f) ? a_array[i] : b_array[i];
        }
        return load<uint64_t, 4>(result);
    }

    // Specialization for bool 4-wide
    template <>
    inline Simd<bool, 4> select(Simd<bool, 4> mask, Simd<bool, 4> a, Simd<bool, 4> b) {
        return Simd<bool, 4>(_mm_blendv_ps(b.value, a.value, mask.value));
    }

    // FMA implementation for AVX
    inline Simd<float, 8> fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
        #ifdef __FMA__
        return Simd<float, 8>(_mm256_fmadd_ps(a.value, b.value, c.value));
        #else
        return a * b + c;
        #endif
    }

    inline Simd<float, 4> fma(Simd<float, 4> a, Simd<float, 4> b, Simd<float, 4> c) {
        #ifdef __FMA__
        return Simd<float, 4>(_mm_fmadd_ps(a.value, b.value, c.value));
        #else
        return a * b + c;
        #endif
    }

    inline Simd<double, 4> fma(Simd<double, 4> a, Simd<double, 4> b, Simd<double, 4> c) {
        #ifdef __FMA__
        return Simd<double, 4>(_mm256_fmadd_pd(a.value, b.value, c.value));
        #else
        return a * b + c;
        #endif
    }

    // Clamp function (restricting a value to a range)
    template <typename T, int N>
    inline Simd<T, N> clamp(Simd<T, N> v, Simd<T, N> min_val, Simd<T, N> max_val) {
        return minimum(maximum(v, min_val), max_val);
    }



    // ====== REDUCTION OPERATIONS ======

    // Sum reduction for float 8-wide
    template <>
    inline float sum<float, 8>(Simd<float, 8> x) {
        __m256 sum1 = _mm256_hadd_ps(x.value, x.value);
        __m256 sum2 = _mm256_hadd_ps(sum1, sum1);
        __m128 lo = _mm256_castps256_ps128(sum2);
        __m128 hi = _mm256_extractf128_ps(sum2, 1);
        __m128 sum_ps = _mm_add_ps(lo, hi);
        alignas(16) float result[4];
        _mm_store_ps(result, sum_ps);
        return result[0];
    }

    // Sum reduction for float 4-wide
    template <>
    inline float sum<float, 4>(Simd<float, 4> x) {
        __m128 sum1 = _mm_hadd_ps(x.value, x.value);
        __m128 sum2 = _mm_hadd_ps(sum1, sum1);
        alignas(16) float result[4];
        _mm_store_ps(result, sum2);
        return result[0];
    }

    // Sum reduction for double 4-wide
    template <>
    inline double sum<double, 4>(Simd<double, 4> x) {
        __m256d sum1 = _mm256_hadd_pd(x.value, x.value);
        __m128d lo = _mm256_castpd256_pd128(sum1);
        __m128d hi = _mm256_extractf128_pd(sum1, 1);
        __m128d sum_pd = _mm_add_pd(lo, hi);
        alignas(16) double result[2];
        _mm_store_pd(result, sum_pd);
        return result[0];
    }

    // Sum reduction for int 8-wide
    template <>
    inline int sum<int, 8>(Simd<int, 8> x) {
        __m256i sum1 = _mm256_hadd_epi32(x.value, x.value);
        __m256i sum2 = _mm256_hadd_epi32(sum1, sum1);
        __m128i lo = _mm256_castsi256_si128(sum2);
        __m128i hi = _mm256_extractf128_si256(sum2, 1);
        __m128i sum_epi32 = _mm_add_epi32(lo, hi);
        alignas(16) int result[4];
        _mm_store_si128((__m128i*)result, sum_epi32);
        return result[0];
    }

    // Max reduction for common types
    template <>
    inline float max<float, 8>(Simd<float, 8> x) {
        __m128 hi = _mm256_extractf128_ps(x.value, 1);
        __m128 lo = _mm256_castps256_ps128(x.value);
        __m128 max1 = _mm_max_ps(lo, hi);
        __m128 max2 = _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 max3 = _mm_max_ps(max1, max2);
        __m128 max4 = _mm_shuffle_ps(max3, max3, _MM_SHUFFLE(0, 1, 0, 1));
        __m128 max5 = _mm_max_ps(max3, max4);
        alignas(16) float result[4];
        _mm_store_ps(result, max5);
        return result[0];
    }

    template <>
    inline int max<int, 8>(Simd<int, 8> x) {
        __m128i hi = _mm256_extractf128_si256(x.value, 1);
        __m128i lo = _mm256_castsi256_si128(x.value);
        __m128i max1 = _mm_max_epi32(lo, hi);
        __m128i max2 = _mm_shuffle_epi32(max1, _MM_SHUFFLE(1, 0, 3, 2));
        __m128i max3 = _mm_max_epi32(max1, max2);
        __m128i max4 = _mm_shuffle_epi32(max3, _MM_SHUFFLE(0, 1, 0, 1));
        __m128i max5 = _mm_max_epi32(max3, max4);
        alignas(16) int result[4];
        _mm_store_si128((__m128i*)result, max5);
        return result[0];
    }

    // Min reduction for common types
    template <>
    inline float min<float, 8>(Simd<float, 8> x) {
        __m128 hi = _mm256_extractf128_ps(x.value, 1);
        __m128 lo = _mm256_castps256_ps128(x.value);
        __m128 min1 = _mm_min_ps(lo, hi);
        __m128 min2 = _mm_shuffle_ps(min1, min1, _MM_SHUFFLE(1, 0, 3, 2));
        __m128 min3 = _mm_min_ps(min1, min2);
        __m128 min4 = _mm_shuffle_ps(min3, min3, _MM_SHUFFLE(0, 1, 0, 1));
        __m128 min5 = _mm_min_ps(min3, min4);
        alignas(16) float result[4];
        _mm_store_ps(result, min5);
        return result[0];
    }

    template <>
    inline int min<int, 8>(Simd<int, 8> x) {
        __m128i hi = _mm256_extractf128_si256(x.value, 1);
        __m128i lo = _mm256_castsi256_si128(x.value);
        __m128i min1 = _mm_min_epi32(lo, hi);
        __m128i min2 = _mm_shuffle_epi32(min1, _MM_SHUFFLE(1, 0, 3, 2));
        __m128i min3 = _mm_min_epi32(min1, min2);
        __m128i min4 = _mm_shuffle_epi32(min3, _MM_SHUFFLE(0, 1, 0, 1));
        __m128i min5 = _mm_min_epi32(min3, min4);
        alignas(16) int result[4];
        _mm_store_si128((__m128i*)result, min5);
        return result[0];
    }

    // Add this specialization for max with bool 8-wide
    template <>
    inline bool max<bool, 8>(Simd<bool, 8> x) {
        // For boolean values, max is true if any element is true
        // This is equivalent to 'any' operation
        return _mm256_movemask_ps(x.value) != 0;
    }

    // And you might need min as well
    template <>
    inline bool min<bool, 8>(Simd<bool, 8> x) {
        // For boolean values, min is false if any element is false
        // This is equivalent to 'all' operation
        return _mm256_movemask_ps(x.value) == 0xFF;
    }

    // And you may need the 4-wide versions too:
    template <>
    inline bool max<bool, 4>(Simd<bool, 4> x) {
        return _mm_movemask_ps(x.value) != 0;
    }

    template <>
    inline bool min<bool, 4>(Simd<bool, 4> x) {
        return _mm_movemask_ps(x.value) == 0xF;
    }

    // Specialization for min reduction with unsigned int (uint32_t) 8-wide
    template <>
    inline uint32_t min<uint32_t, 8>(Simd<uint32_t, 8> x) {
        alignas(32) uint32_t values[8];
        _mm256_store_si256((__m256i*)values, x.value);
        uint32_t min_val = values[0];
        for (int i = 1; i < 8; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
            }
        }
        return min_val;
    }

    // Specialization for max reduction with unsigned int (uint32_t) 8-wide
    template <>
    inline uint32_t max<uint32_t, 8>(Simd<uint32_t, 8> x) {
        alignas(32) uint32_t values[8];
        _mm256_store_si256((__m256i*)values, x.value);
        uint32_t max_val = values[0];
        for (int i = 1; i < 8; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }

    // You might also need sum and prod
    template <>
    inline uint32_t sum<uint32_t, 8>(Simd<uint32_t, 8> x) {
        alignas(32) uint32_t values[8];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] + values[1] + values[2] + values[3] + 
            values[4] + values[5] + values[6] + values[7];
    }

    template <>
    inline uint32_t prod<uint32_t, 8>(Simd<uint32_t, 8> x) {
        alignas(32) uint32_t values[8];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] * values[1] * values[2] * values[3] * 
            values[4] * values[5] * values[6] * values[7];
    }

    // 4-wide versions
    template <>
    inline uint32_t min<uint32_t, 4>(Simd<uint32_t, 4> x) {
        alignas(16) uint32_t values[4];
        _mm_store_si128((__m128i*)values, x.value);
        uint32_t min_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
            }
        }
        return min_val;
    }

    template <>
    inline uint32_t max<uint32_t, 4>(Simd<uint32_t, 4> x) {
        alignas(16) uint32_t values[4];
        _mm_store_si128((__m128i*)values, x.value);
        uint32_t max_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }

    // Boolean reductions
    // Add concrete implementations for specific sizes
    // For 'all' function
    inline bool all(Simd<bool, 4> x) {
        return _mm_movemask_ps(x.value) == 0xF;
    }

    inline bool all(Simd<bool, 8> x) {
        return _mm256_movemask_ps(x.value) == 0xFF;
    }

    // For 'any' function
    inline bool any(Simd<bool, 4> x) {
        return _mm_movemask_ps(x.value) != 0;
    }

    inline bool any(Simd<bool, 8> x) {
        return _mm256_movemask_ps(x.value) != 0;
    }

    // General templates for other sizes (using SFINAE to avoid conflict)
    template <int N>
    inline typename std::enable_if<(N > 1 && N != 4 && N != 8), bool>::type
    all(Simd<bool, N> x) {
        bool result = true;
        for (int i = 0; i < N; ++i) {
            result = result && x[i];
        }
        return result;
    }

    template <int N>
    inline typename std::enable_if<(N > 1 && N != 4 && N != 8), bool>::type
    any(Simd<bool, N> x) {
        bool result = false;
        for (int i = 0; i < N; ++i) {
            result = result || x[i];
        }
        return result;
    }

    // Product reductions 
    template <>
    inline float prod<float, 8>(Simd<float, 8> x) {
        alignas(32) float values[8];
        _mm256_store_ps(values, x.value);
        float product = 1.0f;
        for (int i = 0; i < 8; i++) {
            product *= values[i];
        }
        return product;
    }

    // template <>
    // inline double prod<double, 4>(Simd<double, 4> x) {
    //     alignas(32) double values[4];
    //     _mm256_store_pd(values, x.value);
    //     double product = 1.0;
    //     for (int i = 0; i < 4; i++) {
    //         product *= values[i];
    //     }
    //     return product;
    // }

    template <>
    inline int prod<int, 8>(Simd<int, 8> x) {
        alignas(32) int values[8];
        _mm256_store_si256((__m256i*)values, x.value);
        int product = 1;
        for (int i = 0; i < 8; i++) {
            product *= values[i];
        }
        return product;
    }

    // Add this helper function somewhere in your namespace
    inline double erfinv_impl(double x) {
        // Handle edge cases
        if (x >= 1.0) return std::numeric_limits<double>::infinity();
        if (x <= -1.0) return -std::numeric_limits<double>::infinity();
        if (x == 0.0) return 0.0;

        // Implementation with different approximations based on range
        bool neg = (x < 0);
        if (neg) x = -x;

        double w, p;
        
        // Central region: |x| <= 0.7
        if (x <= 0.7) {
            w = x * x - 0.56249;
            p = (((2.81022636e-08 * w + 3.43273939e-07) * w + -3.5233877e-06) * w +
                -4.39150654e-06) * w + 0.00021858087;
            p = (((1.00950558e-04 * w + 0.00134934322) * w + -0.00367342844) * w +
                0.00573950773) * w + -0.0076224613;
            p = (((1.75966091e-02 * w + -0.0200214257) * w + 0.0223223464) * w +
                -0.0165562398) * w + p;
            p = (((-0.0199463912 * w + -0.0128209635) * w + 0.0094049351) * w + 0.0736418409) * w + 0.888593956;
            p = p * x;
        }
        // Tail region: 0.7 < |x| < 1.0
        else {
            w = std::sqrt(-std::log((1.0 - x) / 2.0));
            p = ((-0.000200214257 * w + 0.000100950558) * w + 0.00134934322) * w + -0.00367342844;
            p = (((0.00573950773 * w + -0.0076224613) * w + 0.0175966091) * w + -0.0199463912) * w + p;
            p = (((-0.0128209635 * w + 0.0223223464) * w + -0.0165562398) * w + 0.0094049351) * w + p;
            p = ((0.0736418409 * w + -0.0200214257) * w + 0.888593956) * w + p;
        }

        return neg ? -p : p;
    }

    // Then use this helper in your SIMD implementation
    inline Simd<double, 4> erfinv(Simd<double, 4> x) {
        alignas(32) double values[4], result[4];
        _mm256_store_pd(values, x.value);
        for (int i = 0; i < 4; i++) {
            result[i] = erfinv_impl(values[i]);
        }
        return load<double, 4>(result);
    }

    // And similarly for the erf function
    inline Simd<double, 4> erf(Simd<double, 4> x) {
        alignas(32) double values[4], result[4];
        _mm256_store_pd(values, x.value);
        for (int i = 0; i < 4; i++) {
            result[i] = std::erf(values[i]);
        }
        return load<double, 4>(result);
    }

    // Add this specialization for min with long (int64_t)
    template <>
    inline long min<long, 4>(Simd<long, 4> x) {
        alignas(32) long values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        long min_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
            }
        }
        return min_val;
    }

    // You'll likely need the max function too
    template <>
    inline long max<long, 4>(Simd<long, 4> x) {
        alignas(32) long values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        long max_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }

    // And while you're at it, you might need the sum function too
    template <>
    inline long sum<long, 4>(Simd<long, 4> x) {
        alignas(32) long values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] + values[1] + values[2] + values[3];
    }

    // And potentially prod
    template <>
    inline long prod<long, 4>(Simd<long, 4> x) {
        alignas(32) long values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] * values[1] * values[2] * values[3];
    }

    // Specialization for max reduction with unsigned long (uint64_t) 4-wide
    template <>
    inline uint64_t max<uint64_t, 4>(Simd<uint64_t, 4> x) {
        alignas(32) uint64_t values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        uint64_t max_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }

    // You'll probably need these other reduction operations for uint64_t too
    template <>
    inline uint64_t min<uint64_t, 4>(Simd<uint64_t, 4> x) {
        alignas(32) uint64_t values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        uint64_t min_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
            }
        }
        return min_val;
    }

    template <>
    inline uint64_t sum<uint64_t, 4>(Simd<uint64_t, 4> x) {
        alignas(32) uint64_t values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] + values[1] + values[2] + values[3];
    }

    template <>
    inline uint64_t prod<uint64_t, 4>(Simd<uint64_t, 4> x) {
        alignas(32) uint64_t values[4];
        _mm256_store_si256((__m256i*)values, x.value);
        return values[0] * values[1] * values[2] * values[3];
    }

    // Specialization for min reduction with double 4-wide
    template <>
    inline double min<double, 4>(Simd<double, 4> x) {
        alignas(32) double values[4];
        _mm256_store_pd(values, x.value);
        double min_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] < min_val) {
                min_val = values[i];
            }
        }
        return min_val;
    }

    // And you might need these other reduction operations for double too
    template <>
    inline double max<double, 4>(Simd<double, 4> x) {
        alignas(32) double values[4];
        _mm256_store_pd(values, x.value);
        double max_val = values[0];
        for (int i = 1; i < 4; i++) {
            if (values[i] > max_val) {
                max_val = values[i];
            }
        }
        return max_val;
    }

    // template <>
    // inline double sum<double, 4>(Simd<double, 4> x) {
    //     alignas(32) double values[4];
    //     _mm256_store_pd(values, x.value);
    //     return values[0] + values[1] + values[2] + values[3];
    // }

    template <>
    inline double prod<double, 4>(Simd<double, 4> x) {
        alignas(32) double values[4];
        _mm256_store_pd(values, x.value);
        return values[0] * values[1] * values[2] * values[3];
    }

} // namespace mlx::core::simd