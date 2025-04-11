#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#ifdef MLX_USE_ACCELERATE
#include "mlx/backend/cpu/simd/accelerate_simd.h"
#endif

#ifdef HAVE_AVX
#include "mlx/backend/cpu/simd/avx_simd.h"
#endif