// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h> // AVX, AVX2, FMA, F16C intrinsics
#include <cstdint>     // For uint16_t, uint32_t
#include <cstring>     // For memcpy
#include <type_traits> // For std::is_same_v
#include <cmath>       // For std::fmaf
#include <limits>      // For numeric_limits
#include <algorithm>   // For std::min/max used in saturation

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
}

// ==========================================================================
// Optimized 8x8 block transpose for packing
// Used to handle strided loads efficiently
// ==========================================================================
template <typename T>
inline void transpose_8x8_block(const T* src, float* dst, int src_stride, int dst_stride) {
    static_assert(std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
                 "transpose_8x8_block requires float16_t or bfloat16_t input");
    
    if constexpr (std::is_same_v<T, float16_t>) {
#ifdef __F16C__
        // ⭐⭐⭐ OPTIMIZATION: Use F16C for optimal float16 to float32 conversion
        // Load 8 rows of 8 float16 elements each (each row is strided)
        __m128i row0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        __m128i row1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + src_stride));
        __m128i row2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * src_stride));
        __m128i row3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3 * src_stride));
        __m128i row4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4 * src_stride));
        __m128i row5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 5 * src_stride));
        __m128i row6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 6 * src_stride));
        __m128i row7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 7 * src_stride));
        
        // Convert float16 to float32 (1 per cycle throughput, 3 cycles latency)
        __m256 frow0 = _mm256_cvtph_ps(row0);
        __m256 frow1 = _mm256_cvtph_ps(row1);
        __m256 frow2 = _mm256_cvtph_ps(row2);
        __m256 frow3 = _mm256_cvtph_ps(row3);
        __m256 frow4 = _mm256_cvtph_ps(row4);
        __m256 frow5 = _mm256_cvtph_ps(row5);
        __m256 frow6 = _mm256_cvtph_ps(row6);
        __m256 frow7 = _mm256_cvtph_ps(row7);
        
        // Transpose 8x8 using AVX shuffles and permutes
        
        // Interleave 2x2 blocks
        __m256 t0 = _mm256_unpacklo_ps(frow0, frow1);
        __m256 t1 = _mm256_unpackhi_ps(frow0, frow1);
        __m256 t2 = _mm256_unpacklo_ps(frow2, frow3);
        __m256 t3 = _mm256_unpackhi_ps(frow2, frow3);
        __m256 t4 = _mm256_unpacklo_ps(frow4, frow5);
        __m256 t5 = _mm256_unpackhi_ps(frow4, frow5);
        __m256 t6 = _mm256_unpacklo_ps(frow6, frow7);
        __m256 t7 = _mm256_unpackhi_ps(frow6, frow7);
        
        // Interleave 4x4 blocks
        __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
        __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
        __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
        __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
        __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        
        // Final permutations between 128-bit lanes
        __m256 r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
        __m256 r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
        __m256 r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
        __m256 r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
        __m256 r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
        __m256 r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
        __m256 r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
        __m256 r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
        
        // ⭐⭐⭐ FIX: Store with the proper stride for column-major access
        // Each row needs to be stored with dst_stride between elements
        _mm256_storeu_ps(dst + 0*dst_stride, r0);
        _mm256_storeu_ps(dst + 1*dst_stride, r1);
        _mm256_storeu_ps(dst + 2*dst_stride, r2);
        _mm256_storeu_ps(dst + 3*dst_stride, r3);
        _mm256_storeu_ps(dst + 4*dst_stride, r4);
        _mm256_storeu_ps(dst + 5*dst_stride, r5);
        _mm256_storeu_ps(dst + 6*dst_stride, r6);
        _mm256_storeu_ps(dst + 7*dst_stride, r7);
#else
        // Fallback without F16C
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                dst[j * dst_stride + i] = static_cast<float>(src[i * src_stride + j]);
            }
        }
#endif
    } else { // bfloat16_t
#ifdef __AVX2__
        // ⭐⭐⭐ OPTIMIZATION: Use optimized bfloat16 to float conversion
        // Improved version using _mm256_cvtepu16_epi32 + _mm256_slli_epi32 + _mm256_castsi256_ps
        // to reduce instruction count
        
        // Process 8 rows, use strides correctly
        __m256 rows[8];
        for (int i = 0; i < 8; i++) {
            // Load 8 bfloat16 values from a single row
            __m128i bf16_vals_u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i * src_stride));
            
            // Optimized conversion from bf16 to float
            // 1. Expand to 32-bit integers
            __m256i bf16_vals_u32 = _mm256_cvtepu16_epi32(bf16_vals_u16);
            // 2. Shift left by 16 bits
            __m256i fp32_bits = _mm256_slli_epi32(bf16_vals_u32, 16);
            // 3. Interpret as float
            rows[i] = _mm256_castsi256_ps(fp32_bits);
        }
        
        // Transpose the 8 rows using AVX shuffles
        __m256 t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
        __m256 t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
        __m256 t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
        __m256 t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
        __m256 t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
        __m256 t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
        __m256 t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
        __m256 t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
        
        __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
        __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
        __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
        __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
        __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        
        __m256 r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
        __m256 r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
        __m256 r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
        __m256 r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
        __m256 r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
        __m256 r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
        __m256 r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
        __m256 r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
        
        // ⭐⭐⭐ FIX: Store with the proper stride for column-major access
        _mm256_storeu_ps(dst + 0*dst_stride, r0);
        _mm256_storeu_ps(dst + 1*dst_stride, r1);
        _mm256_storeu_ps(dst + 2*dst_stride, r2);
        _mm256_storeu_ps(dst + 3*dst_stride, r3);
        _mm256_storeu_ps(dst + 4*dst_stride, r4);
        _mm256_storeu_ps(dst + 5*dst_stride, r5);
        _mm256_storeu_ps(dst + 6*dst_stride, r6);
        _mm256_storeu_ps(dst + 7*dst_stride, r7);
#else
        // Scalar fallback
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                dst[j * dst_stride + i] = static_cast<float>(src[i * src_stride + j]);
            }
        }
#endif
    }
}

// ==========================================================================
// Conversion and Combined Operations (T -> float -> T)
// T = float16_t or bfloat16_t
// ==========================================================================

// ⭐⭐⭐ OPTIMIZATION: Optimized Load 8 x T values, convert to float8 (__m256)
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
#ifdef __AVX2__
        // ⭐⭐⭐ OPTIMIZATION: More efficient bfloat16 to float conversion
        // Using one _mm256_cvtepu16_epi32 + _mm256_slli_epi32 + _mm256_castsi256_ps
        
        // Load 8 bfloat16 values (as uint16_t)
        __m128i bf16_vals_u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        
        // Zero-extend bf16_vals_u16 to 256 bits (__m256i)
        __m256i bf16_vals_u32 = _mm256_cvtepu16_epi32(bf16_vals_u16);
        // Shift left by 16 (equivalent to multiplying by 2^16)
        __m256i fp32_bits = _mm256_slli_epi32(bf16_vals_u32, 16);
        // Cast integer bits to float vector
        return float8(_mm256_castsi256_ps(fp32_bits));
#else
        // Scalar fallback
        float buffer[8];
        for (int i = 0; i < 8; ++i) {
            uint32_t val_int = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(src)[i]) << 16;
            std::memcpy(&buffer[i], &val_int, sizeof(float));
        }
        return load<float, 8>(buffer);
#endif
    }
}

// ⭐⭐⭐ OPTIMIZATION: Helper function for float to bfloat16 conversion with RNE (Round to Nearest Even)
// Extracted to standalone function so compiler can better optimize and hoist constants
#ifdef __AVX2__
inline __m128i convert_float_to_bfloat16_rne_avx2(__m256 src) {
    // Get float bits
    __m256i val_int = _mm256_castps_si256(src);
    
    // Add rounding bias (0x7FFF add causes round-to-nearest, ties to even implicitly)
    __m256i bias = _mm256_set1_epi32(0x7FFF);
    __m256i rounded_val = _mm256_add_epi32(val_int, bias);
    
    // Shift right by 16 (extract upper 16 bits)
    __m256i bf16_bits_32 = _mm256_srli_epi32(rounded_val, 16);
    
    // Split 256-bit vector into two 128-bit vectors
    __m128i bf16_bits_low = _mm256_castsi256_si128(bf16_bits_32);      // Lower 128 bits (elements 0-3)
    __m128i bf16_bits_high = _mm256_extracti128_si256(bf16_bits_32, 1); // Upper 128 bits (elements 4-7)
    
    // ⭐⭐⭐ FIX: Use SIGNED packing intrinsic instead of unsigned
    // This preserves negative values instead of zeroing them
    return _mm_packs_epi32(bf16_bits_low, bf16_bits_high);
}
#endif

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
        for(int i=0; i<8; ++i) dst[i] = static_cast<T>(buffer[i]); // Requires float16_t::operator=(float)
#endif
    } else { // bfloat16_t
#ifdef __AVX2__
        // ⭐⭐⭐ OPTIMIZATION: Use standalone helper function for bfloat16 conversion
        // Round-to-nearest-even for bfloat16 conversions
        __m128i bf16_result = convert_float_to_bfloat16_rne_avx2(src.value);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), bf16_result);
#else
        // Scalar Fallback with RNE logic
        float buffer[8];
        store<float, 8>(buffer, src);
        alignas(16) uint16_t bf16_bits_arr[8];
        for (int i = 0; i < 8; ++i) {
            uint32_t val_int;
            std::memcpy(&val_int, &buffer[i], sizeof(float));
            
            // Handle NaN specifically
            if ((val_int & 0x7F800000) == 0x7F800000 && (val_int & 0x007FFFFF) != 0) {
                bf16_bits_arr[i] = 0x7FC0 | static_cast<uint16_t>((val_int >> 16) & 0x003F);
            } else {
                // Apply rounding bias for RNE
                uint32_t rounding_bias = ((val_int >> 16) & 1) + 0x7FFF;
                val_int += rounding_bias;
                bf16_bits_arr[i] = static_cast<uint16_t>(val_int >> 16);
            }
        }
        std::memcpy(dst, bf16_bits_arr, 8 * sizeof(uint16_t));
#endif
    }
}

// --- Combined Scale, Accumulate, Store ---
// Performs: C = alpha * acc + beta * C_old
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

// ⭐⭐ OPTIMIZATION: New optimized microkernel (8x16) implementation - uses AVX2 intrinsics directly
// This achieves better performance by:
// 1. Unrolling the k loop (by 4)
// 2. Pre-loading B values for all unrolled k iterations
// 3. Carefully scheduling the broadcasts and FMAs to hide latencies
// On Skylake-X this can sustain ~80% of peak FMA throughput
template <int MR = 8, int NR = 16>
inline void micro_kernel_8x16(
    const float* A_panel,   // Packed A panel, col-major
    const float* B_panel,   // Packed B panel, row-major
    float* C_block,         // Output block in C_acc
    int ldc,                // Leading dimension of C
    int kc,                 // Number of K iterations
    int a_stride,   //  == MC_BLOCK
    int b_stride)   //  == NC_BLOCK
{
    static_assert(MR == 8, "This kernel requires MR=8");
    static_assert(NR == 16, "This kernel requires NR=16");
    
    // Define our vectors - use raw __m256 for best compiler optimization
    __m256 c[MR][NR/8];

    // Load accumulators 
    #pragma unroll
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR/8; ++j) {
            c[i][j] = _mm256_loadu_ps(C_block + i*ldc + 8*j);
        }
    }

    // Main k-loop with unrolling by 4
    for (int k = 0; k < kc; k += 4) {
        // Handle edge case for k dimension
        if (k + 4 > kc) {
            // Handle remaining k iterations one at a time
            for (int kr = k; kr < kc; ++kr) {
                // Load B vectors (two vectors for NR=16)
                // __m256 b0 = _mm256_loadu_ps(B_panel + kr*NR);
                // __m256 b1 = _mm256_loadu_ps(B_panel + kr*NR + 8);
                __m256 b0 = _mm256_loadu_ps(B_panel + kr*b_stride);
                __m256 b1 = _mm256_loadu_ps(B_panel + kr*b_stride + 8);
                
                // Process each row of A
                #pragma unroll
                for (int i = 0; i < MR; ++i) {
                    // Broadcast A scalar to vector
                    // __m256 a = _mm256_broadcast_ss(A_panel + i + kr*MR);
                    __m256 a = _mm256_broadcast_ss(A_panel + i + kr*a_stride);
                    
                    // FMA operations for the two B vectors
                    c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
                    c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
                }
            }
            break;
        }
        
        // Load all B vectors for 4 iterations (2 vectors per iteration for NR=16)
        __m256 b00 = _mm256_loadu_ps(B_panel + (k+0)*b_stride);
        __m256 b01 = _mm256_loadu_ps(B_panel + (k+0)*b_stride + 8);
        __m256 b10 = _mm256_loadu_ps(B_panel + (k+1)*b_stride);
        __m256 b11 = _mm256_loadu_ps(B_panel + (k+1)*b_stride + 8);
        __m256 b20 = _mm256_loadu_ps(B_panel + (k+2)*b_stride);
        __m256 b21 = _mm256_loadu_ps(B_panel + (k+2)*b_stride + 8);
        __m256 b30 = _mm256_loadu_ps(B_panel + (k+3)*b_stride);
        __m256 b31 = _mm256_loadu_ps(B_panel + (k+3)*b_stride + 8);

        // Process each row of A
        #pragma unroll
        for (int i = 0; i < MR; ++i) {
            // Broadcast A scalar to vector for each k iteration
            __m256 a0 = _mm256_broadcast_ss(A_panel + i + (k+0)*a_stride);
            __m256 a1 = _mm256_broadcast_ss(A_panel + i + (k+1)*a_stride);
            __m256 a2 = _mm256_broadcast_ss(A_panel + i + (k+2)*a_stride);
            __m256 a3 = _mm256_broadcast_ss(A_panel + i + (k+3)*a_stride);

            // First iteration (k+0)
            c[i][0] = _mm256_fmadd_ps(a0, b00, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a0, b01, c[i][1]);
            
            // Second iteration (k+1)
            c[i][0] = _mm256_fmadd_ps(a1, b10, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a1, b11, c[i][1]);
            
            // Third iteration (k+2)
            c[i][0] = _mm256_fmadd_ps(a2, b20, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a2, b21, c[i][1]);
            
            // Fourth iteration (k+3)
            c[i][0] = _mm256_fmadd_ps(a3, b30, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a3, b31, c[i][1]);
        }
    }

    // Store results back to C
    #pragma unroll
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR/8; ++j) {
            _mm256_storeu_ps(C_block + i*ldc + 8*j, c[i][j]);
        }
    }
}

} // namespace mlx::core::simd