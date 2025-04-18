// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h> // AVX, AVX2, FMA, F16C intrinsics
#include <cstdint>     // For uint16_t, uint32_t
#include <cstring>     // For memcpy
#include <type_traits> // For std::is_same_v
#include <cmath>       // For std::fmaf
#include <limits>      // For numeric_limits
#include <algorithm>   // For std::min/max used in saturation? (Handled by T conversion usually)

// Assume these types are defined elsewhere, matching MLX's definitions
#include "mlx/backend/cpu/simd/base_simd.h" // Should define float16_t, bfloat16_t and conversion operators/methods

namespace mlx::core::simd {

// Forward declarations
template <typename T, int N> struct Simd;
template <typename T, int N> inline Simd<T, N> load(const T* ptr);
template <typename T, int N> inline void store(T* ptr, Simd<T, N> x);
template <typename T, int N> inline Simd<T, N> broadcast(const T* x);
template <typename T, int N> inline Simd<T, N> fma(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c);
template <typename T> inline void scale_accumulate_store(
    T* c_ptr,
    Simd<float, 8> acc,    // Accumulation result for the *current* panel
    Simd<float, 8> alpha_vec,
    Simd<float, 8> beta_vec); // The *effective* beta for this panel update


// ==========================================================================
// Specialization for float using AVX (__m256, N=8)
// ==========================================================================
template <> constexpr int max_size<float> = 8;
using float8 = Simd<float, 8>; // Alias

template <>
struct Simd<float, 8> {
    static constexpr int size = 8;
    __m256 value;

    Simd() : value(_mm256_setzero_ps()) {}
    Simd(float v) : value(_mm256_set1_ps(v)) {}
    explicit Simd(__m256 v) : value(v) {}
    Simd(const Simd& other) = default;
    Simd& operator=(const Simd& other) = default;
    operator __m256() const { return value; }
};

// --- Load/Store (float) ---
template <> inline float8 load<float, 8>(const float* x) {
    return float8(_mm256_loadu_ps(x));
}
template <> inline void store<float, 8>(float* dst, float8 x) {
    _mm256_storeu_ps(dst, x.value);
}
template <> inline float8 broadcast<float, 8>(const float* x) {
    // Ensure x is valid pointer. Behavior is undefined if x is null.
    // Consider adding null check if necessary, though typically not in perf kernels.
    return float8(_mm256_broadcast_ss(x));
}

// --- Arithmetic (float) ---
inline float8 operator+(float8 a, float8 b) { return float8(_mm256_add_ps(a, b)); }
inline float8 operator-(float8 a, float8 b) { return float8(_mm256_sub_ps(a, b)); }
inline float8 operator*(float8 a, float8 b) { return float8(_mm256_mul_ps(a, b)); }
inline float8 operator/(float8 a, float8 b) { return float8(_mm256_div_ps(a, b)); }

// --- FMA (float) ---
template <> inline float8 fma<float, 8>(float8 a, float8 b, float8 c) {
#ifdef __AVX2__
    return float8(_mm256_fmadd_ps(a, b, c)); // c = a * b + c
#else
    // Fallback without FMA (lower precision)
    return float8(_mm256_add_ps(_mm256_mul_ps(a, b), c));
#endif
}

// --- Horizontal Sum (float) ---
// Not strictly needed for GEMM microkernel, but can be useful
inline float sum(float8 x) {
    __m256 val = x.value;
    __m128 vlow = _mm256_castps256_ps128(val);
    __m128 vhigh = _mm256_extractf128_ps(val, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);              // add the low 128
    __m128 shuf = _mm_movehdup_ps(vlow);         // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);           // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
    // Alternative using hadd (often slower than shuffle approach)
    // __m128 sum128 = _mm_add_ps(hi, lo);
    // __m128 hadd1 = _mm_hadd_ps(sum128, sum128);
    // __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
    // return _mm_cvtss_f32(hadd2);
}


// ==========================================================================
// Conversion and Combined Operations (T -> float -> T)
// T = float16_t or bfloat16_t
// ==========================================================================

// Load 8 x T values, convert to float8 (__m256)
template <typename T>
inline float8 load_convert_to_float(const T* src) {
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                  "load_convert_to_float requires float16_t or bfloat16_t input for this specialization.");
    static_assert(sizeof(T) == 2, "Input type T must be 2 bytes.");

    if constexpr (std::is_same_v<T, float16_t>) {
#ifdef __F16C__
        // Load 8 float16 values (128 bits)
        __m128i f16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        // Convert to 8 float values (__m256)
        return float8(_mm256_cvtph_ps(f16_vals));
#else
        float buffer[8];
        for (int i = 0; i < 8; ++i) buffer[i] = static_cast<float>(src[i]); // Requires T::operator float()
        return load<float, 8>(buffer);
#endif
    } else { // bfloat16_t
        // Load 8 bfloat16 values (as uint16_t)
        __m128i bf16_vals_u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));

#ifdef __AVX512BF16__ // Use AVX512-BF16 if available (very recent CPUs)
        // Need AVX512VL for 256-bit version? Check intrinsics guide.
        // Example assuming __m256bh exists and vcvtneebf16_ps is the intrinsic:
        // return float8(_mm256_cvtneebf16_ps(reinterpret_cast<__m256bh>(bf16_vals_u16)));
        // If not directly available, fall through to manual or AVX2 approach.
        // Let's assume AVX512BF16 isn't the primary target for now.
#endif

#ifdef __AVX2__ // Use AVX2 shifts and combines
        // Expand uint16 vector to uint32 vector by shifting left by 16
        // Zero-extend bf16_vals_u16 to 256 bits (__m256i)
        __m256i bf16_vals_u32 = _mm256_cvtepu16_epi32(bf16_vals_u16);
        // Shift left by 16 (equivalent to multiplying by 2^16)
        __m256i fp32_bits = _mm256_slli_epi32(bf16_vals_u32, 16);
        // Cast integer bits to float vector
        return float8(_mm256_castsi256_ps(fp32_bits));
#else // Scalar fallback (or if F16C/AVX2 missing entirely)
        float buffer[8];
        alignas(16) uint16_t raw_bits[8];
        std::memcpy(raw_bits, src, 8 * sizeof(bfloat16_t));
        for (int i = 0; i < 8; ++i) {
            uint32_t val_int = static_cast<uint32_t>(raw_bits[i]) << 16;
            std::memcpy(&buffer[i], &val_int, sizeof(float));
        }
        return load<float, 8>(buffer);
#endif
    }
}

// Store float8 (__m256), converting back to 8 x T values
template <typename T>
inline void store_convert_from_float(T* dst, float8 src) {
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                   "store_convert_from_float requires float16_t or bfloat16_t output for this specialization.");
    static_assert(sizeof(T) == 2, "Output type T must be 2 bytes.");

    if constexpr (std::is_same_v<T, float16_t>) {
#ifdef __F16C__
        // Convert 8 float values (__m256) to 8 float16 values (__m128i)
        // Rounding mode: TO_NEAREST_INT is round-nearest-ties-to-even
        __m128i f16_result = _mm256_cvtps_ph(src.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Store 128 bits (8 * float16)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), f16_result);
#else
        float buffer[8];
        store<float, 8>(buffer, src);
        for(int i=0; i<8; ++i) dst[i] = static_cast<float16_t>(buffer[i]); // Requires float16_t::operator=(float)
#endif
    } else { // bfloat16_t
// #if defined(__AVX512_BF16__) && defined(__AVX512VL__)
//         // Use AVX512-BF16 conversion intrinsic if available
//         // Assumes vcvtneeps2bf16 intrinsic exists for 256-bit regs (_MM_FROUND_TO_NEAREST_INT default?)
//         // __m128i bf16_result = _mm256_cvtneeps_pbh(src.value, _MM_FROUND_TO_NEAREST_INT); // Check actual intrinsic name/signature
//         // _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), bf16_result);
//         // Fall through if not using AVX512BF16
// #endif
        // AVX2 / SSE approach (Round-to-nearest-even is preferred for BF16)
        // Based on Eigen's implementation idea or similar hardware instructions behavior

        __m256i val_int = _mm256_castps_si256(src.value); // Get float bits

        // Add rounding bias (0x7FFF add causes round-to-nearest, ties to even implicitly)
        __m256i bias = _mm256_set1_epi32(0x7FFF);
        __m256i rounded_val = _mm256_add_epi32(val_int, bias);

        // Shift right by 16 (extract upper 16 bits)
        __m256i bf16_bits_32 = _mm256_srli_epi32(rounded_val, 16);

        // Pack 32-bit integers down to 16-bit integers.
        // AVX2 pack instructions _mm256_packus_epi32 or _mm256_packs_epi32 saturate.
        // We need truncation after shift, so permute/shuffle approach:
        // Shuffle pairs of 32-bit integers, keeping the lower 16 bits (which contain our shifted result)
        // Example using permutevar8x32: indices select which 32-bit lanes go where.
        // We want lanes 0,2,4,6 from lower 128 and 0,2,4,6 from upper 128 (lanes 8,10,12,14 relative to 256 reg)
        // Indices: 0, 2, 4, 6, 8, 10, 12, 14
        __m256i permute_indices = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        __m256i packed_bf16_32 = _mm256_permutevar8x32_epi32(bf16_bits_32, permute_indices);

        // Extract the lower 128 bits which now contain the 8 packed uint16 results
        __m128i bf16_result = _mm256_castsi256_si128(packed_bf16_32);

        // Handle NaN specifically? BF16 conversion often maps SNaN->QNaN, preserves payload MSB.
        // The RNE approach might handle NaNs implicitly, but verification needed.
        // Check for NaN inputs (exponent=0xFF, mantissa!=0) and output bf16 NaN (0x7FC0 | payload_msb).
        // This adds complexity, common ML frameworks might rely on the RNE behavior directly.

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), bf16_result);

        // Scalar Fallback (RNE logic)
        // float buffer[8];
        // store<float, 8>(buffer, src);
        // alignas(16) uint16_t bf16_bits_arr[8];
        // for (int i = 0; i < 8; ++i) {
        //     uint32_t val_int;
        //     std::memcpy(&val_int, &buffer[i], sizeof(float));
        //     if ((val_int & 0x7F800000) == 0x7F800000 && (val_int & 0x007FFFFF) != 0) { // Check for NaN
        //         bf16_bits_arr[i] = 0x7FC0 | static_cast<uint16_t>((val_int >> 16) & 0x003F); // Convert to QNaN bf16
        //     } else {
        //         uint32_t rounding_bias = ((val_int >> 16) & 1) + 0x7FFF;
        //         val_int += rounding_bias;
        //         bf16_bits_arr[i] = static_cast<uint16_t>(val_int >> 16);
        //     }
        // }
        // std::memcpy(dst, bf16_bits_arr, 8 * sizeof(bfloat16_t));
    }
}


// --- Combined Scale, Accumulate, Store ---
// Performs: C = alpha * acc + beta * C_old
// acc: Result of A*B for the current K-panel (float8)
// alpha_vec: Broadcasted alpha (float8)
// beta_vec: Broadcasted *effective* beta for this panel (float8)
// c_ptr: Pointer to output matrix elements (type T)
template <typename T> // T = float16_t or bfloat16_t
inline void scale_accumulate_store(
    T* c_ptr,
    float8 acc,
    float8 alpha_vec,
    float8 beta_vec)
{
    // Calculate alpha * accumulation (Result is float8)
    float8 result = alpha_vec * acc;

    // Check effective_beta. Extract first element.
    float beta_scalar;
    _mm_store_ss(&beta_scalar, _mm256_castps256_ps128(beta_vec.value));

    // If effective beta is exactly 0.0f, we just store alpha * acc.
    // Otherwise, load C_old, convert to float, compute beta * C_old + (alpha * acc).
    if (beta_scalar != 0.0f) {
        // Load existing C values (type T) and convert them to float8 vector
        float8 c_old_vec = load_convert_to_float<T>(c_ptr);

        // result = beta_vec * c_old_vec + result; (using FMA for float precision)
        result = fma<float, 8>(beta_vec, c_old_vec, result);
    }

    // Store final result (float8 vector) back to C (converting float -> T)
    store_convert_from_float<T>(c_ptr, result);
}

} // namespace mlx::core::simd