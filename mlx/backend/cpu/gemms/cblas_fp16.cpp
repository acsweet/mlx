#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/lapack.h"

namespace mlx::core {

template <>
void matmul<float16_t>(
    const float16_t* a,
    const float16_t* b,
    float16_t* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];
  
  // Use thread-local storage for temporary buffers
  thread_local std::vector<float> a_float;
  thread_local std::vector<float> b_float;
  thread_local std::vector<float> out_float;
  
  // Resize only if needed
  if (a_float.size() < M * K) a_float.resize(M * K);
  if (b_float.size() < K * N) b_float.resize(K * N);
  if (out_float.size() < M * N) out_float.resize(M * N);
  
  for (int i = 0; i < batch_size; ++i) {
    // Get pointers with correct offsets
    const float16_t* a_ptr = a + elem_to_loc(M * K * i, a_shape, a_strides);
    const float16_t* b_ptr = b + elem_to_loc(K * N * i, b_shape, b_strides);
    float16_t* out_ptr = out + M * N * i;
    
    // Vectorized conversion from fp16 to float32
    constexpr int simd_size = simd::max_size<float>;
    for (size_t j = 0; j < M * K; j += simd_size) {
      size_t remaining = std::min(simd_size, static_cast<int>(M * K - j));
      if (remaining == simd_size) {
        auto simd_val = simd::load<float16_t, simd_size>(a_ptr + j);
        simd::store(a_float.data() + j, simd::Simd<float, simd_size>(simd_val));
      } else {
        for (size_t k = 0; k < remaining; ++k) {
          a_float[j + k] = static_cast<float>(a_ptr[j + k]);
        }
      }
    }
    
    for (size_t j = 0; j < K * N; j += simd_size) {
      size_t remaining = std::min(simd_size, static_cast<int>(K * N - j));
      if (remaining == simd_size) {
        auto simd_val = simd::load<float16_t, simd_size>(b_ptr + j);
        simd::store(b_float.data() + j, simd::Simd<float, simd_size>(simd_val));
      } else {
        for (size_t k = 0; k < remaining; ++k) {
          b_float[j + k] = static_cast<float>(b_ptr[j + k]);
        }
      }
    }
    
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans,
        b_transposed ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        alpha,
        a_float.data(),
        lda,
        b_float.data(),
        ldb,
        0.0f,  // Beta is applied afterwards
        out_float.data(),
        ldc);
    
    // Convert back to fp16 with beta application
    for (size_t j = 0; j < M * N; j += simd_size) {
      size_t remaining = std::min(simd_size, static_cast<int>(M * N - j));
      
      if (beta != 0.0f) {
        // Apply beta to original output values
        if (remaining == simd_size) {
          auto out_vals = simd::load<float16_t, simd_size>(out_ptr + j);
          auto float_out = simd::Simd<float, simd_size>(out_vals);
          auto result = simd::Simd<float, simd_size>(
              simd::load<float, simd_size>(out_float.data() + j));
          
          result = result + float_out * beta;
          simd::store(out_ptr + j, simd::Simd<float16_t, simd_size>(result));
        } else {
          for (size_t k = 0; k < remaining; ++k) {
            out_ptr[j + k] = static_cast<float16_t>(
                out_float[j + k] + beta * static_cast<float>(out_ptr[j + k]));
          }
        }
      } else {
        // Direct conversion without beta
        if (remaining == simd_size) {
          auto result = simd::load<float, simd_size>(out_float.data() + j);
          simd::store(out_ptr + j, simd::Simd<float16_t, simd_size>(result));
        } else {
          for (size_t k = 0; k < remaining; ++k) {
            out_ptr[j + k] = static_cast<float16_t>(out_float[j + k]);
          }
        }
      }
    }
  }
}

} // namespace mlx::core