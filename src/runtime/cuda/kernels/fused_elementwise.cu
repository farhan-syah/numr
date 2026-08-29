// Fused elementwise CUDA kernels
// fused_mul_add: out = a * b + c (FMA)
// fused_add_mul: out = (a + b) * c
// fused_mul_add_scalar: out = a * scale + bias
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//        i64, i32, i16, i8, u64, u32, u16, u8
//
// Kernel naming, matching `kernel_name(op, dtype)` in
// src/runtime/cuda/kernels/loader.rs: {op}_{suffix}.
//
// The integer rows compose binop_mul and binop_add from binary_ops.cuh rather
// than writing `a * b + c`. A fused op must answer exactly what the unfused
// sequence answers, and for integers that sequence wraps at every step; the
// float rows contract the two roundings into one FMA instead, which is the
// point of fusing them.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "binary_ops.cuh"
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// fused_mul_add: out = a * b + c
// ============================================================================

__global__ void fused_mul_add_f32(const float* a, const float* b, const float* c, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaf(a[idx], b[idx], c[idx]);
    }
}

__global__ void fused_mul_add_f64(const double* a, const double* b, const double* c, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fma(a[idx], b[idx], c[idx]);
    }
}

__global__ void fused_mul_add_f16(const __half* a, const __half* b, const __half* c, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float vc = __half2float(c[idx]);
        out[idx] = __float2half(fmaf(va, vb, vc));
    }
}

__global__ void fused_mul_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        float vc = __bfloat162float(c[idx]);
        out[idx] = __float2bfloat16(fmaf(va, vb, vc));
    }
}

// ============================================================================
// fused_add_mul: out = (a + b) * c
// ============================================================================

__global__ void fused_add_mul_f32(const float* a, const float* b, const float* c, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] + b[idx]) * c[idx];
    }
}

__global__ void fused_add_mul_f64(const double* a, const double* b, const double* c, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] + b[idx]) * c[idx];
    }
}

__global__ void fused_add_mul_f16(const __half* a, const __half* b, const __half* c, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float vc = __half2float(c[idx]);
        out[idx] = __float2half((va + vb) * vc);
    }
}

__global__ void fused_add_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        float vc = __bfloat162float(c[idx]);
        out[idx] = __float2bfloat16((va + vb) * vc);
    }
}

// ============================================================================
// fused_mul_add_scalar: out = a * scale + bias
// ============================================================================

__global__ void fused_mul_add_scalar_f32(const float* a, float* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaf(a[idx], scale, bias);
    }
}

__global__ void fused_mul_add_scalar_f64(const double* a, double* out, unsigned int n, double scale, double bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fma(a[idx], scale, bias);
    }
}

__global__ void fused_mul_add_scalar_f16(const __half* a, __half* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        out[idx] = __float2half(fmaf(va, scale, bias));
    }
}

__global__ void fused_mul_add_scalar_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(fmaf(va, scale, bias));
    }
}

// ============================================================================
// FP8 fused_mul_add: out = a * b + c
// ============================================================================

__global__ void fused_mul_add_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, const numr_fp8_e4m3* c, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e4m3_to_f32(a[idx].data);
        float vb = fp8_e4m3_to_f32(b[idx].data);
        float vc = fp8_e4m3_to_f32(c[idx].data);
        out[idx].data = f32_to_fp8_e4m3(fmaf(va, vb, vc));
    }
}

__global__ void fused_mul_add_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, const numr_fp8_e5m2* c, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e5m2_to_f32(a[idx].data);
        float vb = fp8_e5m2_to_f32(b[idx].data);
        float vc = fp8_e5m2_to_f32(c[idx].data);
        out[idx].data = f32_to_fp8_e5m2(fmaf(va, vb, vc));
    }
}

// ============================================================================
// FP8 fused_add_mul: out = (a + b) * c
// ============================================================================

__global__ void fused_add_mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, const numr_fp8_e4m3* c, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e4m3_to_f32(a[idx].data);
        float vb = fp8_e4m3_to_f32(b[idx].data);
        float vc = fp8_e4m3_to_f32(c[idx].data);
        out[idx].data = f32_to_fp8_e4m3((va + vb) * vc);
    }
}

__global__ void fused_add_mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, const numr_fp8_e5m2* c, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e5m2_to_f32(a[idx].data);
        float vb = fp8_e5m2_to_f32(b[idx].data);
        float vc = fp8_e5m2_to_f32(c[idx].data);
        out[idx].data = f32_to_fp8_e5m2((va + vb) * vc);
    }
}

// ============================================================================
// FP8 fused_mul_add_scalar: out = a * scale + bias
// ============================================================================

__global__ void fused_mul_add_scalar_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e4m3_to_f32(a[idx].data);
        out[idx].data = f32_to_fp8_e4m3(fmaf(va, scale, bias));
    }
}

__global__ void fused_mul_add_scalar_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = fp8_e5m2_to_f32(a[idx].data);
        out[idx].data = f32_to_fp8_e5m2(fmaf(va, scale, bias));
    }
}


// ============================================================================
// Integer fused operations
//
// Every step wraps, so `fused_mul_add(a, b, c)` equals `add(mul(a, b), c)`
// element for element, including where the intermediate product leaves the
// dtype. `fused_mul_add_scalar` takes scale and bias in the element type,
// already saturated to that range by the host's `as` cast, so it equally
// equals `add_scalar(mul_scalar(a, scale), bias)`.
// ============================================================================

#define NUMR_FUSED_INT_ROW(T, SUF)                                              \
    __global__ void fused_mul_add_##SUF(                                        \
        const T* a, const T* b, const T* c, T* out, unsigned int n) {           \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = binop_add<T>(binop_mul<T>(a[idx], b[idx]), c[idx]);      \
        }                                                                       \
    }                                                                           \
    __global__ void fused_add_mul_##SUF(                                        \
        const T* a, const T* b, const T* c, T* out, unsigned int n) {           \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = binop_mul<T>(binop_add<T>(a[idx], b[idx]), c[idx]);      \
        }                                                                       \
    }                                                                           \
    __global__ void fused_mul_add_scalar_##SUF(                                 \
        const T* a, T* out, unsigned int n, T scale, T bias) {                  \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = binop_add<T>(binop_mul<T>(a[idx], scale), bias);         \
        }                                                                       \
    }

NUMR_FUSED_INT_ROW(int64_t, i64)
NUMR_FUSED_INT_ROW(int32_t, i32)
NUMR_FUSED_INT_ROW(int16_t, i16)
NUMR_FUSED_INT_ROW(int8_t, i8)
NUMR_FUSED_INT_ROW(uint64_t, u64)
NUMR_FUSED_INT_ROW(uint32_t, u32)
NUMR_FUSED_INT_ROW(uint16_t, u16)
NUMR_FUSED_INT_ROW(uint8_t, u8)

} // extern "C"
