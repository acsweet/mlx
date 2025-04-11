// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"

#ifdef HAVE_AVX
#include "mlx/backend/cpu/gemms/avx_simd_gemm.h"
#else
#include "mlx/backend/cpu/gemms/simd_gemm.h"
#endif

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
  for (int i = 0; i < batch_size; ++i) {
    simd_gemm<float16_t, float>(
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        out + M * N * i,
        a_transposed,
        b_transposed,
        M,
        N,
        K,
        alpha,
        beta);
  }
}

} // namespace mlx::core
