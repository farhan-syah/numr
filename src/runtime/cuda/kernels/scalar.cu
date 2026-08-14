// Scalar CUDA kernels (tensor-scalar operations)
// Supports: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64
//
// FP8 operations compute in FP32 and convert back to FP8 for storage.
// Hopper (SM 8.9+) uses native PTX intrinsics for FP8 conversion.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 Scalar Operations
// ============================================================================

__global__ void add_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_pow_safe(a[idx], scalar);
    }
}

__global__ void rsub_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = scalar - a[idx];
    }
}

// ============================================================================
// F64 Scalar Operations
// ============================================================================

__global__ void add_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_pow_safe(a[idx], scalar);
    }
}

__global__ void rsub_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = scalar - a[idx];
    }
}

// ============================================================================
// F16 Scalar Operations (half precision)
// Note: Scalar is passed as float and converted to half for compatibility
// ============================================================================

__global__ void add_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hadd(a[idx], s);
    }
}

__global__ void sub_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hsub(a[idx], s);
    }
}

__global__ void mul_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hmul(a[idx], s);
    }
}

__global__ void div_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hdiv(a[idx], s);
    }
}

__global__ void pow_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __half2float(a[idx]);
        out[idx] = __float2half(numr_pow_safe(af, scalar));
    }
}

__global__ void rsub_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hsub(s, a[idx]);
    }
}

// ============================================================================
// BF16 Scalar Operations (bfloat16)
// Note: Scalar is passed as float and converted to bfloat16
// ============================================================================

__global__ void add_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hadd(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + scalar);
        #endif
    }
}

__global__ void sub_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hsub(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) - scalar);
        #endif
    }
}

__global__ void mul_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hmul(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * scalar);
        #endif
    }
}

__global__ void div_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hdiv(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) / scalar);
        #endif
    }
}

__global__ void pow_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(numr_pow_safe(af, scalar));
    }
}

__global__ void rsub_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hsub(s, a[idx]);
        #else
        out[idx] = __float2bfloat16(scalar - __bfloat162float(a[idx]));
        #endif
    }
}

// ============================================================================
// I32 Scalar Operations
// ============================================================================

__global__ void add_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void rsub_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = scalar - a[idx];
    }
}

// ============================================================================
// I64 Scalar Operations
// ============================================================================

__global__ void add_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void rsub_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = scalar - a[idx];
    }
}

// ============================================================================
// FP8 E4M3 Scalar Operations
// Scalar is passed as float, operations compute in FP32
// Hopper (SM 8.9+) uses native PTX intrinsics for conversion
// ============================================================================

__global__ void add_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af + scalar));
    }
}

__global__ void sub_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af - scalar));
    }
}

__global__ void mul_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af * scalar));
    }
}

__global__ void div_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af / scalar));
    }
}

__global__ void pow_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(numr_pow_safe(af, scalar)));
    }
}

__global__ void rsub_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(scalar - af));
    }
}

// ============================================================================
// FP8 E5M2 Scalar Operations
// Same pattern as E4M3 but with E5M2 conversion functions
// ============================================================================

__global__ void add_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af + scalar));
    }
}

__global__ void sub_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af - scalar));
    }
}

__global__ void mul_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af * scalar));
    }
}

__global__ void div_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af / scalar));
    }
}

__global__ void pow_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(numr_pow_safe(af, scalar)));
    }
}

__global__ void rsub_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(scalar - af));
    }
}

// ============================================================================
// Complex64 (float2) Scalar Operations
// Scalar is a real float that operates on complex numbers
// ============================================================================

// Add real scalar to complex: (a+bi) + s = (a+s) + bi
__global__ void add_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x + scalar, a[idx].y);
    }
}

// Subtract real scalar from complex: (a+bi) - s = (a-s) + bi
__global__ void sub_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x - scalar, a[idx].y);
    }
}

// Reverse subtract: s - (a+bi) = (s-a) + (-b)i
__global__ void rsub_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(scalar - a[idx].x, -a[idx].y);
    }
}

// Multiply complex by real scalar: s(a+bi) = sa + sbi
__global__ void mul_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x * scalar, a[idx].y * scalar);
    }
}

// Divide complex by real scalar: (a+bi)/s = a/s + (b/s)i
__global__ void div_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x / scalar, a[idx].y / scalar);
    }
}

// Complex power with real exponent: z^p
// Edge cases:
//   - 0^p where p < 0: returns (Inf, Inf) - division by zero
//   - 0^0: returns (1, 0) - mathematical convention
//   - 0^p where p > 0: returns (0, 0)
__global__ void pow_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float2 z = a[idx];
        float r = complex64_abs(z);

        // Handle edge cases for zero magnitude
        if (r == 0.0f) {
            if (scalar < 0.0f) {
                // 0^(-p) = Inf (division by zero)
                out[idx] = make_float2(NUMR_INF_F, NUMR_INF_F);
            } else if (scalar == 0.0f) {
                // 0^0 = 1 (mathematical convention)
                out[idx] = make_float2(1.0f, 0.0f);
            } else {
                // 0^p = 0 for p > 0
                out[idx] = make_float2(0.0f, 0.0f);
            }
            return;
        }

        // z^p = |z|^p * e^(i * p * theta)
        float theta = complex64_angle(z);
        float r_pow = powf(r, scalar);
        float new_theta = scalar * theta;
        float sin_t, cos_t;
        sincosf(new_theta, &sin_t, &cos_t);

        out[idx] = make_float2(r_pow * cos_t, r_pow * sin_t);
    }
}

// ============================================================================
// Complex128 (double2) Scalar Operations
// ============================================================================

__global__ void add_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x + scalar, a[idx].y);
    }
}

__global__ void sub_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x - scalar, a[idx].y);
    }
}

// Reverse subtract: s - (a+bi) = (s-a) + (-b)i
__global__ void rsub_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(scalar - a[idx].x, -a[idx].y);
    }
}

__global__ void mul_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x * scalar, a[idx].y * scalar);
    }
}

__global__ void div_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x / scalar, a[idx].y / scalar);
    }
}

// Complex128 power with real exponent: z^p
// Edge cases mirror pow_scalar_c64
__global__ void pow_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double2 z = a[idx];
        double r = complex128_abs(z);

        // Handle edge cases for zero magnitude
        if (r == 0.0) {
            if (scalar < 0.0) {
                // 0^(-p) = Inf (division by zero)
                out[idx] = make_double2(NUMR_INF, NUMR_INF);
            } else if (scalar == 0.0) {
                // 0^0 = 1 (mathematical convention)
                out[idx] = make_double2(1.0, 0.0);
            } else {
                // 0^p = 0 for p > 0
                out[idx] = make_double2(0.0, 0.0);
            }
            return;
        }

        // z^p = |z|^p * e^(i * p * theta)
        double theta = complex128_angle(z);
        double r_pow = pow(r, scalar);
        double new_theta = scalar * theta;
        double sin_t, cos_t;
        sincos(new_theta, &sin_t, &cos_t);

        out[idx] = make_double2(r_pow * cos_t, r_pow * sin_t);
    }
}

} // extern "C"
