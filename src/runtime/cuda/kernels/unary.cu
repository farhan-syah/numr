// Unary element-wise CUDA kernels
//
// Supports:
//   Basic:       neg, abs, sign, logical_not
//   Power/Root:  sqrt, rsqrt, cbrt, square, recip
//   Exponential: exp, exp2, expm1
//   Logarithmic: log, log2, log10, log1p
//   Trigonometric: sin, cos, tan, asin, acos, atan
//   Hyperbolic:  sinh, cosh, tanh, asinh, acosh, atanh
//   Rounding:    floor, ceil, round, round_ties_even, trunc
//
//   round          = roundf/round: ties away from zero (matches Rust f32::round)
//   round_ties_even = rintf/rint under the default round-to-nearest-even mode:
//                     IEEE roundTiesToEven (matches np.round and torch.round)
//   Special:     clamp, isnan, isinf
//
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool)
//        neg, abs and sign also cover i8 and i16
//
// Notes:
//   - Clamp operation: clamp(x, min_scalar, max_scalar) = min(max(x, lo), hi)
//   - Sign operation: returns -1, 0, or +1
//   - IsNaN/IsInf: returns u8 (bool) - 1 if true, 0 if false
//   - Logical NOT: boolean negation (u8 input/output)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Clamp Helper Device Function (Templated - must be outside extern "C")
// ============================================================================

template<typename T>
__device__ __forceinline__ T clamp_impl(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : (val > max_val ? max_val : val);
}

// Specialization for float
template<>
__device__ __forceinline__ float clamp_impl(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

// Specialization for double
template<>
__device__ __forceinline__ double clamp_impl(double val, double min_val, double max_val) {
    return fmin(fmax(val, min_val), max_val);
}

// Specialization for half
template<>
__device__ __forceinline__ __half clamp_impl(__half val, __half min_val, __half max_val) {
    return __hmin(__hmax(val, min_val), max_val);
}

// Specialization for bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 clamp_impl(__nv_bfloat16 val, __nv_bfloat16 min_val, __nv_bfloat16 max_val) {
    #if __CUDA_ARCH__ >= 800
    return __hmin(__hmax(val, min_val), max_val);
    #else
    float vf = __bfloat162float(val);
    float minf = __bfloat162float(min_val);
    float maxf = __bfloat162float(max_val);
    return __float2bfloat16(fminf(fmaxf(vf, minf), maxf));
    #endif
}

// Specialization for FP8E4M3 (clamp in F32)
template<>
__device__ __forceinline__ numr_fp8_e4m3 clamp_impl(numr_fp8_e4m3 val, numr_fp8_e4m3 min_val, numr_fp8_e4m3 max_val) {
    float vf = fp8_e4m3_to_f32(val.data);
    float minf = fp8_e4m3_to_f32(min_val.data);
    float maxf = fp8_e4m3_to_f32(max_val.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fmaxf(vf, minf), maxf)));
}

// Specialization for FP8E5M2 (clamp in F32)
template<>
__device__ __forceinline__ numr_fp8_e5m2 clamp_impl(numr_fp8_e5m2 val, numr_fp8_e5m2 min_val, numr_fp8_e5m2 max_val) {
    float vf = fp8_e5m2_to_f32(val.data);
    float minf = fp8_e5m2_to_f32(min_val.data);
    float maxf = fp8_e5m2_to_f32(max_val.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fmaxf(vf, minf), maxf)));
}

// ============================================================================
// Integer neg/abs/sign Helper Device Functions (Templated - outside extern "C")
// ============================================================================
//
// Element-wise integer ops WRAP. That is the convention documented in
// src/runtime/cpu/kernels/wide_acc.rs and mirrored by binary_ops.cuh, so
// neg and abs of the type's minimum answer the minimum itself rather than
// saturating to the maximum. WebGPU's unary_i32.wgsl answers the same by
// WGSL's own definition.
//
// Signed overflow is undefined behaviour in C++, so both run in the dtype's
// unsigned counterpart and convert back. That conversion is modulo 2^N, which
// is exactly what Rust's wrapping_neg and wrapping_abs produce. A bare `-a` or
// `abs(a)` is UB on the minimum instead.

template<typename T, typename U>
__device__ __forceinline__ T numr_int_neg(T a) {
    return (T)(U)((U)0 - (U)a);
}

template<typename T, typename U>
__device__ __forceinline__ T numr_int_abs(T a) {
    return (a < (T)0) ? (T)(U)((U)0 - (U)a) : a;
}

template<typename T>
__device__ __forceinline__ T numr_int_sign(T a) {
    return (a > (T)0) ? (T)1 : ((a < (T)0) ? (T)(-1) : (T)0);
}

extern "C" {

// ============================================================================
// F32 Unary Operations
// ============================================================================

__global__ void neg_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(a[idx]);
    }
}

__global__ void sqrt_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(a[idx]);
    }
}

__global__ void exp_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(a[idx]);
    }
}

__global__ void log_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(a[idx]);
    }
}

__global__ void sin_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinf(a[idx]);
    }
}

__global__ void cos_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosf(a[idx]);
    }
}

__global__ void tan_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanf(a[idx]);
    }
}

__global__ void tanh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(a[idx]);
    }
}

__global__ void recip_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / a[idx];
    }
}

__global__ void square_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floorf(a[idx]);
    }
}

__global__ void ceil_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceilf(a[idx]);
    }
}

__global__ void round_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = roundf(a[idx]);
    }
}

__global__ void round_ties_even_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rintf(a[idx]);
    }
}

__global__ void trunc_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = truncf(a[idx]);
    }
}

__global__ void rsqrt_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rsqrtf(a[idx]);
    }
}

__global__ void cbrt_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cbrtf(a[idx]);
    }
}

__global__ void exp2_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp2f(a[idx]);
    }
}

__global__ void expm1_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expm1f(a[idx]);
    }
}

__global__ void log2_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log2f(a[idx]);
    }
}

__global__ void log10_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log10f(a[idx]);
    }
}

__global__ void log1p_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log1pf(a[idx]);
    }
}

__global__ void asin_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = asinf(a[idx]);
    }
}

__global__ void acos_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = acosf(a[idx]);
    }
}

__global__ void atan_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atanf(a[idx]);
    }
}

__global__ void sinh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinhf(a[idx]);
    }
}

__global__ void cosh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = coshf(a[idx]);
    }
}

__global__ void asinh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = asinhf(a[idx]);
    }
}

__global__ void acosh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = acoshf(a[idx]);
    }
}

__global__ void atanh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atanhf(a[idx]);
    }
}

// ============================================================================
// F64 Unary Operations
// ============================================================================

__global__ void neg_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabs(a[idx]);
    }
}

__global__ void sqrt_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrt(a[idx]);
    }
}

__global__ void exp_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(a[idx]);
    }
}

__global__ void log_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log(a[idx]);
    }
}

__global__ void sin_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sin(a[idx]);
    }
}

__global__ void cos_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cos(a[idx]);
    }
}

__global__ void tan_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tan(a[idx]);
    }
}

__global__ void tanh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanh(a[idx]);
    }
}

__global__ void recip_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0 / a[idx];
    }
}

__global__ void square_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floor(a[idx]);
    }
}

__global__ void ceil_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceil(a[idx]);
    }
}

__global__ void round_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = round(a[idx]);
    }
}

__global__ void round_ties_even_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rint(a[idx]);
    }
}

__global__ void trunc_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = trunc(a[idx]);
    }
}

__global__ void rsqrt_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = rsqrt(a[idx]);
    }
}

__global__ void cbrt_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cbrt(a[idx]);
    }
}

__global__ void exp2_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp2(a[idx]);
    }
}

__global__ void expm1_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expm1(a[idx]);
    }
}

__global__ void log2_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log2(a[idx]);
    }
}

__global__ void log10_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log10(a[idx]);
    }
}

__global__ void log1p_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log1p(a[idx]);
    }
}

__global__ void asin_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = asin(a[idx]);
    }
}

__global__ void acos_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = acos(a[idx]);
    }
}

__global__ void atan_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan(a[idx]);
    }
}

__global__ void sinh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinh(a[idx]);
    }
}

__global__ void cosh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosh(a[idx]);
    }
}

__global__ void asinh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = asinh(a[idx]);
    }
}

__global__ void acosh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = acosh(a[idx]);
    }
}

__global__ void atanh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atanh(a[idx]);
    }
}

// ============================================================================
// F16 Unary Operations (half precision)
// Transcendental functions use FP32 internally for accuracy
// ============================================================================

__global__ void neg_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hneg(a[idx]);
    }
}

__global__ void abs_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __habs(a[idx]);
    }
}

__global__ void sqrt_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hsqrt(a[idx]);
    }
}

__global__ void exp_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hexp(a[idx]);
    }
}

__global__ void log_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hlog(a[idx]);
    }
}

__global__ void sin_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hsin(a[idx]);
    }
}

__global__ void cos_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hcos(a[idx]);
    }
}

__global__ void tan_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // No native htan, use FP32
        out[idx] = __float2half(tanf(__half2float(a[idx])));
    }
}

__global__ void tanh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for accuracy
        out[idx] = __float2half(tanhf(__half2float(a[idx])));
    }
}

__global__ void recip_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hrcp(a[idx]);
    }
}

__global__ void square_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half val = a[idx];
        out[idx] = __hmul(val, val);
    }
}

__global__ void floor_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = __half2float(a[idx]);
        out[idx] = __float2half(floorf(fa));
    }
}

__global__ void ceil_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = __half2float(a[idx]);
        out[idx] = __float2half(ceilf(fa));
    }
}

__global__ void round_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = __half2float(a[idx]);
        out[idx] = __float2half(roundf(fa));
    }
}

__global__ void round_ties_even_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = __half2float(a[idx]);
        out[idx] = __float2half(rintf(fa));
    }
}

__global__ void trunc_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = __half2float(a[idx]);
        out[idx] = __float2half(truncf(fa));
    }
}

__global__ void rsqrt_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hrsqrt(a[idx]);
    }
}

__global__ void cbrt_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(cbrtf(__half2float(a[idx])));
    }
}

__global__ void exp2_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hexp2(a[idx]);
    }
}

__global__ void expm1_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(expm1f(__half2float(a[idx])));
    }
}

__global__ void log2_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hlog2(a[idx]);
    }
}

__global__ void log10_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hlog10(a[idx]);
    }
}

__global__ void log1p_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(log1pf(__half2float(a[idx])));
    }
}

__global__ void asin_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(asinf(__half2float(a[idx])));
    }
}

__global__ void acos_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(acosf(__half2float(a[idx])));
    }
}

__global__ void atan_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(atanf(__half2float(a[idx])));
    }
}

__global__ void sinh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(sinhf(__half2float(a[idx])));
    }
}

__global__ void cosh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(coshf(__half2float(a[idx])));
    }
}

__global__ void asinh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(asinhf(__half2float(a[idx])));
    }
}

__global__ void acosh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(acoshf(__half2float(a[idx])));
    }
}

__global__ void atanh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(atanhf(__half2float(a[idx])));
    }
}

// ============================================================================
// BF16 Unary Operations (bfloat16)
// All transcendental functions use FP32 internally (no native BF16 math)
// ============================================================================

__global__ void neg_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hneg(a[idx]);
        #else
        out[idx] = __float2bfloat16(-__bfloat162float(a[idx]));
        #endif
    }
}

__global__ void abs_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __habs(a[idx]);
        #else
        out[idx] = __float2bfloat16(fabsf(__bfloat162float(a[idx])));
        #endif
    }
}

__global__ void sqrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(sqrtf(__bfloat162float(a[idx])));
    }
}

__global__ void exp_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(expf(__bfloat162float(a[idx])));
    }
}

__global__ void log_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(logf(__bfloat162float(a[idx])));
    }
}

__global__ void sin_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(sinf(__bfloat162float(a[idx])));
    }
}

__global__ void cos_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(cosf(__bfloat162float(a[idx])));
    }
}

__global__ void tan_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(tanf(__bfloat162float(a[idx])));
    }
}

__global__ void tanh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(tanhf(__bfloat162float(a[idx])));
    }
}

__global__ void recip_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(1.0f / __bfloat162float(a[idx]));
    }
}

__global__ void square_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(val * val);
    }
}

__global__ void floor_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(floorf(__bfloat162float(a[idx])));
    }
}

__global__ void ceil_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(ceilf(__bfloat162float(a[idx])));
    }
}

__global__ void round_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(roundf(__bfloat162float(a[idx])));
    }
}

__global__ void round_ties_even_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(rintf(__bfloat162float(a[idx])));
    }
}

__global__ void trunc_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(truncf(__bfloat162float(a[idx])));
    }
}

__global__ void rsqrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(rsqrtf(__bfloat162float(a[idx])));
    }
}

__global__ void cbrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(cbrtf(__bfloat162float(a[idx])));
    }
}

__global__ void exp2_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(exp2f(__bfloat162float(a[idx])));
    }
}

__global__ void expm1_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(expm1f(__bfloat162float(a[idx])));
    }
}

__global__ void log2_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(log2f(__bfloat162float(a[idx])));
    }
}

__global__ void log10_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(log10f(__bfloat162float(a[idx])));
    }
}

__global__ void log1p_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(log1pf(__bfloat162float(a[idx])));
    }
}

__global__ void asin_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(asinf(__bfloat162float(a[idx])));
    }
}

__global__ void acos_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(acosf(__bfloat162float(a[idx])));
    }
}

__global__ void atan_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(atanf(__bfloat162float(a[idx])));
    }
}

__global__ void sinh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(sinhf(__bfloat162float(a[idx])));
    }
}

__global__ void cosh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(coshf(__bfloat162float(a[idx])));
    }
}

__global__ void asinh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(asinhf(__bfloat162float(a[idx])));
    }
}

__global__ void acosh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(acoshf(__bfloat162float(a[idx])));
    }
}

__global__ void atanh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(atanhf(__bfloat162float(a[idx])));
    }
}

// ============================================================================
// I32 Unary Operations
// ============================================================================

__global__ void neg_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_neg<int, unsigned int>(a[idx]);
    }
}

__global__ void abs_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_abs<int, unsigned int>(a[idx]);
    }
}

__global__ void square_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        out[idx] = val * val;
    }
}

// ============================================================================
// I64 Unary Operations
// ============================================================================

__global__ void neg_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_neg<long long, unsigned long long>(a[idx]);
    }
}

__global__ void abs_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_abs<long long, unsigned long long>(a[idx]);
    }
}

__global__ void square_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long val = a[idx];
        out[idx] = val * val;
    }
}

// ============================================================================
// I8 / I16 Unary Operations
//
// Only neg, abs and sign. The narrow signed dtypes carry no transcendental
// kernels, and these three are the ops backend_parity/unary_signed_int.rs
// sweeps over every signed width.
// ============================================================================

__global__ void neg_i8(const signed char* a, signed char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_neg<signed char, unsigned char>(a[idx]);
    }
}

__global__ void abs_i8(const signed char* a, signed char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_abs<signed char, unsigned char>(a[idx]);
    }
}

__global__ void sign_i8(const signed char* a, signed char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_sign<signed char>(a[idx]);
    }
}

__global__ void neg_i16(const short* a, short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_neg<short, unsigned short>(a[idx]);
    }
}

__global__ void abs_i16(const short* a, short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_abs<short, unsigned short>(a[idx]);
    }
}

__global__ void sign_i16(const short* a, short* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_sign<short>(a[idx]);
    }
}

// ============================================================================
// FP8 E4M3 Unary Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
// ============================================================================

__global__ void neg_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(-fa));
    }
}

__global__ void abs_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fabsf(fa)));
    }
}

__global__ void sqrt_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sqrtf(fa)));
    }
}

__global__ void exp_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(expf(fa)));
    }
}

__global__ void log_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(logf(fa)));
    }
}

__global__ void sin_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sinf(fa)));
    }
}

__global__ void cos_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(cosf(fa)));
    }
}

__global__ void tan_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(tanf(fa)));
    }
}

__global__ void tanh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(tanhf(fa)));
    }
}

__global__ void recip_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(1.0f / fa));
    }
}

__global__ void square_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fa));
    }
}

__global__ void floor_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(floorf(fa)));
    }
}

__global__ void ceil_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(ceilf(fa)));
    }
}

__global__ void round_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(roundf(fa)));
    }
}

__global__ void round_ties_even_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(rintf(fa)));
    }
}

__global__ void trunc_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(truncf(fa)));
    }
}

__global__ void rsqrt_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(rsqrtf(fa)));
    }
}

__global__ void cbrt_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(cbrtf(fa)));
    }
}

__global__ void exp2_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(exp2f(fa)));
    }
}

__global__ void expm1_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(expm1f(fa)));
    }
}

__global__ void log2_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(log2f(fa)));
    }
}

__global__ void log10_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(log10f(fa)));
    }
}

__global__ void log1p_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(log1pf(fa)));
    }
}

__global__ void asin_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(asinf(fa)));
    }
}

__global__ void acos_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(acosf(fa)));
    }
}

__global__ void atan_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(atanf(fa)));
    }
}

__global__ void sinh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sinhf(fa)));
    }
}

__global__ void cosh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(coshf(fa)));
    }
}

__global__ void asinh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(asinhf(fa)));
    }
}

__global__ void acosh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(acoshf(fa)));
    }
}

__global__ void atanh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(atanhf(fa)));
    }
}

// ============================================================================
// FP8 E5M2 Unary Operations
// ============================================================================

__global__ void neg_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(-fa));
    }
}

__global__ void abs_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fabsf(fa)));
    }
}

__global__ void sqrt_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sqrtf(fa)));
    }
}

__global__ void exp_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(expf(fa)));
    }
}

__global__ void log_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(logf(fa)));
    }
}

__global__ void sin_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sinf(fa)));
    }
}

__global__ void cos_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(cosf(fa)));
    }
}

__global__ void tan_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(tanf(fa)));
    }
}

__global__ void tanh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(tanhf(fa)));
    }
}

__global__ void recip_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(1.0f / fa));
    }
}

__global__ void square_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fa));
    }
}

__global__ void floor_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(floorf(fa)));
    }
}

__global__ void ceil_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(ceilf(fa)));
    }
}

__global__ void round_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(roundf(fa)));
    }
}

__global__ void round_ties_even_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(rintf(fa)));
    }
}

__global__ void trunc_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(truncf(fa)));
    }
}

__global__ void rsqrt_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(rsqrtf(fa)));
    }
}

__global__ void cbrt_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(cbrtf(fa)));
    }
}

__global__ void exp2_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(exp2f(fa)));
    }
}

__global__ void expm1_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(expm1f(fa)));
    }
}

__global__ void log2_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(log2f(fa)));
    }
}

__global__ void log10_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(log10f(fa)));
    }
}

__global__ void log1p_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(log1pf(fa)));
    }
}

__global__ void asin_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(asinf(fa)));
    }
}

__global__ void acos_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(acosf(fa)));
    }
}

__global__ void atan_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(atanf(fa)));
    }
}

__global__ void sinh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sinhf(fa)));
    }
}

__global__ void cosh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(coshf(fa)));
    }
}

__global__ void asinh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(asinhf(fa)));
    }
}

__global__ void acosh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(acoshf(fa)));
    }
}

__global__ void atanh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(atanhf(fa)));
    }
}

// ============================================================================
// Clamp Operations (tensor, min_val, max_val) - Kernel Wrappers
// ============================================================================

__global__ void clamp_f32(const float* a, float min_val, float max_val, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<float>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_f64(const double* a, double min_val, double max_val, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<double>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_f16(const __half* a, __half min_val, __half max_val, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<__half>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_bf16(const __nv_bfloat16* a, __nv_bfloat16 min_val, __nv_bfloat16 max_val, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<__nv_bfloat16>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_i32(const int* a, int min_val, int max_val, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<int>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_i64(const long long* a, long long min_val, long long max_val, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<long long>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3 min_val, numr_fp8_e4m3 max_val, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<numr_fp8_e4m3>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2 min_val, numr_fp8_e5m2 max_val, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<numr_fp8_e5m2>(a[idx], min_val, max_val);
    }
}

// ============================================================================
// Sign Operations - Returns -1, 0, or +1
// ============================================================================

__global__ void sign_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        out[idx] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void sign_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        out[idx] = (val > 0.0) ? 1.0 : ((val < 0.0) ? -1.0 : 0.0);
    }
}

__global__ void sign_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(a[idx]);
        out[idx] = __float2half((val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f));
    }
}

__global__ void sign_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16((val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f));
    }
}

__global__ void sign_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_sign<int>(a[idx]);
    }
}

__global__ void sign_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_int_sign<long long>(a[idx]);
    }
}

__global__ void sign_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sign));
    }
}

__global__ void sign_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sign));
    }
}

// ============================================================================
// IsNaN Operations - Returns 1 if NaN, 0 otherwise (output is u8)
// ============================================================================

__global__ void isnan_f32(const float* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_f64(const double* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_f16(const __half* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hisnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_bf16(const __nv_bfloat16* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(__bfloat162float(a[idx])) ? 1 : 0;
    }
}

__global__ void isnan_fp8_e4m3(const numr_fp8_e4m3* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = isnan(val) ? 1 : 0;
    }
}

__global__ void isnan_fp8_e5m2(const numr_fp8_e5m2* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = isnan(val) ? 1 : 0;
    }
}

// ============================================================================
// IsInf Operations - Returns 1 if Inf, 0 otherwise (output is u8)
// ============================================================================

__global__ void isinf_f32(const float* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_f64(const double* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_f16(const __half* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hisinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_bf16(const __nv_bfloat16* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(__bfloat162float(a[idx])) ? 1 : 0;
    }
}

__global__ void isinf_fp8_e4m3(const numr_fp8_e4m3* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = isinf(val) ? 1 : 0;
    }
}

__global__ void isinf_fp8_e5m2(const numr_fp8_e5m2* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = isinf(val) ? 1 : 0;
    }
}

// ============================================================================
// Logical NOT - Boolean negation (input and output are u8)
// ============================================================================

__global__ void logical_not_u8(const unsigned char* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] ? 0 : 1;
    }
}

// ============================================================================
// Complex64 (float2) Unary Operations
// ============================================================================

__global__ void neg_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_neg(a[idx]);
    }
}

__global__ void conj_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_conj(a[idx]);
    }
}

// Returns magnitude as float (real output)
__global__ void abs_c64(const float2* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_abs(a[idx]);
    }
}

// Returns phase angle as float (real output)
__global__ void angle_c64(const float2* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_angle(a[idx]);
    }
}

// Extracts real part
__global__ void real_c64(const float2* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].x;
    }
}

// Extracts imaginary part
__global__ void imag_c64(const float2* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].y;
    }
}

__global__ void square_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_square(a[idx]);
    }
}

__global__ void sqrt_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_sqrt(a[idx]);
    }
}

__global__ void exp_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_exp(a[idx]);
    }
}

__global__ void log_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_log(a[idx]);
    }
}

__global__ void recip_c64(const float2* a, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_recip(a[idx]);
    }
}

// ============================================================================
// Complex128 (double2) Unary Operations
// ============================================================================

__global__ void neg_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_neg(a[idx]);
    }
}

__global__ void conj_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_conj(a[idx]);
    }
}

// Returns magnitude as double (real output)
__global__ void abs_c128(const double2* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_abs(a[idx]);
    }
}

// Returns phase angle as double (real output)
__global__ void angle_c128(const double2* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_angle(a[idx]);
    }
}

// Extracts real part
__global__ void real_c128(const double2* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].x;
    }
}

// Extracts imaginary part
__global__ void imag_c128(const double2* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].y;
    }
}

__global__ void square_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_square(a[idx]);
    }
}

__global__ void sqrt_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_sqrt(a[idx]);
    }
}

__global__ void exp_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_exp(a[idx]);
    }
}

__global__ void log_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_log(a[idx]);
    }
}

__global__ void recip_c128(const double2* a, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_recip(a[idx]);
    }
}

} // extern "C"
