// Binary element-wise CUDA kernels
// Supports: add, sub, mul, div, pow, max, min, logical_and, logical_or, logical_xor
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool for logical ops)
// Includes broadcast variants for all arithmetic types

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Broadcast Helper Device Functions (Templated)
// ============================================================================

template<typename T>
__device__ __forceinline__ T broadcast_add(T a, T b) { return a + b; }

template<typename T>
__device__ __forceinline__ T broadcast_sub(T a, T b) { return a - b; }

template<typename T>
__device__ __forceinline__ T broadcast_mul(T a, T b) { return a * b; }

template<typename T>
__device__ __forceinline__ T broadcast_div(T a, T b) { return a / b; }

template<typename T>
__device__ __forceinline__ T broadcast_pow(T a, T b) { return numr_pow_safe((float)a, (float)b); }

template<>
__device__ __forceinline__ double broadcast_pow(double a, double b) { return numr_pow_safe(a, b); }

template<typename T>
__device__ __forceinline__ T broadcast_max(T a, T b) { return a > b ? a : b; }

template<typename T>
__device__ __forceinline__ T broadcast_min(T a, T b) { return a < b ? a : b; }

// Specializations for half precision
template<>
__device__ __forceinline__ __half broadcast_pow(__half a, __half b) {
    return __float2half(numr_pow_safe(__half2float(a), __half2float(b)));
}

// Specializations for bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 broadcast_pow(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(numr_pow_safe(__bfloat162float(a), __bfloat162float(b)));
}

// Specializations for integers
template<>
__device__ __forceinline__ int32_t broadcast_pow(int32_t a, int32_t b) {
    return (int32_t)numr_pow_safe((float)a, (float)b);
}

template<>
__device__ __forceinline__ int64_t broadcast_pow(int64_t a, int64_t b) {
    return (int64_t)numr_pow_safe((float)a, (float)b);
}

// Specializations for FP8E4M3 (compute in F32)
template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_add(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa + fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_sub(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa - fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_mul(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_div(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa / fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_pow(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(numr_pow_safe(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_max(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_min(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fa, fb)));
}

// Specializations for FP8E5M2 (compute in F32)
template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_add(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa + fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_sub(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa - fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_mul(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_div(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa / fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_pow(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(numr_pow_safe(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_max(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_min(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fa, fb)));
}

// Generic broadcast kernel template
template<typename T, typename OpFunc>
__device__ void broadcast_kernel_impl(
    const T* a, const T* b, T* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n,
    OpFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = op(a[a_offset], b[b_offset]);
}

// CUDA-graph-safe inline broadcast kernel template.
//
// Strides and shape are passed as individual scalar arguments baked into the
// kernel-parameter block rather than as device-memory pointers. This eliminates
// the H2D memcpy nodes that the pointer-based variant creates during graph
// capture — those nodes encode the host-side Vec<u32> addresses, which become
// stale (dangling) on graph replay, causing CUDA_ERROR_ILLEGAL_ADDRESS.
//
// Up to 8 dimensions are supported (MAX_BROADCAST_DIMS = 8). Unused trailing
// dimensions must be zero-padded by the caller.
#define MAX_BROADCAST_DIMS 8

// ============================================================================
// Magic-number fast division helpers
//
// For each dimension d with size shape[d], the caller precomputes:
//   magic[d]  — 32-bit multiplier (unsigned)
//   shift[d]  — post-multiply right-shift amount
//
// These satisfy: floor(x / shape[d]) == __umulhi(x, magic[d]) >> shift[d]
// for all 0 <= x < 2^32 when magic/shift are computed correctly.
//
// This replaces the ~20-40 cycle hardware integer division with 1 mulhi + 1
// shift, making the broadcast kernel bandwidth-bound instead of divide-bound.
// ============================================================================
__device__ __forceinline__ unsigned int fast_div(unsigned int x, unsigned int magic, unsigned int shift) {
    // __umulhi returns the high 32 bits of x * magic (64-bit multiply)
    return (__umulhi(x, magic) >> shift);
}

// Fast-path kernel for the common trailing-broadcast pattern:
//   a has the same shape as out (contiguous, stride == natural stride)
//   b is a contiguous tensor with b_numel elements that repeats along the
//   leading dimensions.
//   b_index(idx) = idx % b_numel  =>  b[fast_div + subtraction trick]
//
// This covers:  [M,N] + [1,N] (b_numel=N, b broadcasts over rows)
//               [B,H,S,S] + [B,1,1,S] (b_numel=S, b broadcasts over B*H*S)
//               and any other contiguous trailing broadcast.
//
// Args:
//   b_magic, b_shift  — magic-number for dividing by b_numel
//   b_numel           — size of the repeating b tensor
template<typename T, typename OpFunc>
__device__ __forceinline__ void broadcast_fast_trailing_impl(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    unsigned int b_magic,
    unsigned int b_shift,
    unsigned int b_numel,
    unsigned int n,
    OpFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // b_idx = idx % b_numel  =>  idx - floor(idx/b_numel)*b_numel
    // b_magic==0 is the power-of-2 sentinel: q = idx >> b_shift
    unsigned int q = (b_magic == 0u) ? (idx >> b_shift) : fast_div(idx, b_magic, b_shift);
    unsigned int b_idx = idx - q * b_numel;
    out[idx] = op(a[idx], b[b_idx]);
}

// General magic-number broadcast kernel.
//
// Replaces the shape-based div/mod loop with precomputed magic+shift per dim.
// The shape is still passed (needed for coord = remaining - q*shape[d]).
// This eliminates hardware integer division entirely.
template<typename T, typename OpFunc>
__device__ void broadcast_kernel_impl_inline(
    const T* a, const T* b, T* out,
    // a_strides[0..7]
    unsigned int as0, unsigned int as1, unsigned int as2, unsigned int as3,
    unsigned int as4, unsigned int as5, unsigned int as6, unsigned int as7,
    // b_strides[0..7]
    unsigned int bs0, unsigned int bs1, unsigned int bs2, unsigned int bs3,
    unsigned int bs4, unsigned int bs5, unsigned int bs6, unsigned int bs7,
    // shape[0..7]
    unsigned int sh0, unsigned int sh1, unsigned int sh2, unsigned int sh3,
    unsigned int sh4, unsigned int sh5, unsigned int sh6, unsigned int sh7,
    // magic[0..7]  (precomputed fast-div multipliers for each shape dimension)
    unsigned int mg0, unsigned int mg1, unsigned int mg2, unsigned int mg3,
    unsigned int mg4, unsigned int mg5, unsigned int mg6, unsigned int mg7,
    // post-shift[0..7]
    unsigned int ps0, unsigned int ps1, unsigned int ps2, unsigned int ps3,
    unsigned int ps4, unsigned int ps5, unsigned int ps6, unsigned int ps7,
    unsigned int ndim, unsigned int n,
    OpFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Unpack inline args into local arrays so the loop below can index them.
    const unsigned int a_strides[MAX_BROADCAST_DIMS] = {as0, as1, as2, as3, as4, as5, as6, as7};
    const unsigned int b_strides[MAX_BROADCAST_DIMS] = {bs0, bs1, bs2, bs3, bs4, bs5, bs6, bs7};
    const unsigned int shape[MAX_BROADCAST_DIMS]     = {sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7};
    const unsigned int magic[MAX_BROADCAST_DIMS]     = {mg0, mg1, mg2, mg3, mg4, mg5, mg6, mg7};
    const unsigned int pshift[MAX_BROADCAST_DIMS]    = {ps0, ps1, ps2, ps3, ps4, ps5, ps6, ps7};

    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    // Unrolled for up to 8 dims using precomputed magic-number division.
    // Sentinel: magic[d]==0 means use bit-shift (q = remaining >> pshift[d]).
    //   d==1: magic=0, shift=0 → q = remaining; coord = remaining - remaining*1 = 0. ✓
    //   d==2^k: magic=0, shift=k → q = remaining>>k; coord = remaining - q*d. ✓
    //   general: q = __umulhi(remaining, magic[d]) >> pshift[d]. ✓
    #pragma unroll
    for (int d = MAX_BROADCAST_DIMS - 1; d >= 0; d--) {
        if ((unsigned int)d >= ndim) continue;
        unsigned int q;
        if (magic[d] == 0u) {
            q = remaining >> pshift[d];
        } else {
            q = fast_div(remaining, magic[d], pshift[d]);
        }
        unsigned int coord = remaining - q * shape[d];
        remaining = q;
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = op(a[a_offset], b[b_offset]);
}

extern "C" {

// ============================================================================
// F32 Binary Operations
// ============================================================================

__global__ void add_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_pow_safe(a[idx], b[idx]);
    }
}

__global__ void max_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void min_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void atan2_f32(const float* y, const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2f(y[idx], x[idx]);
    }
}

// ============================================================================
// F16 Binary Operations (half precision)
// Uses half2 vectorization for 2x throughput where possible
// ============================================================================

__global__ void add_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void sub_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void div_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hdiv(a[idx], b[idx]);
    }
}

__global__ void pow_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __half2float(a[idx]);
        float bf = __half2float(b[idx]);
        out[idx] = __float2half(numr_pow_safe(af, bf));
    }
}

__global__ void max_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void atan2_f16(const __half* y, const __half* x, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(atan2f(__half2float(y[idx]), __half2float(x[idx])));
    }
}

// ============================================================================
// BF16 Binary Operations (bfloat16)
// Note: BF16 arithmetic requires SM 8.0+ (Ampere)
// ============================================================================

__global__ void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hadd(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hsub(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) - __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hmul(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void div_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hdiv(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) / __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void pow_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __bfloat162float(a[idx]);
        float bf = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(numr_pow_safe(af, bf));
    }
}

__global__ void max_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) > __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

__global__ void min_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) < __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

__global__ void atan2_bf16(const __nv_bfloat16* y, const __nv_bfloat16* x, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(atan2f(__bfloat162float(y[idx]), __bfloat162float(x[idx])));
    }
}

// ============================================================================
// F64 Binary Operations
// ============================================================================

__global__ void add_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_pow_safe(a[idx], b[idx]);
    }
}

__global__ void max_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmax(a[idx], b[idx]);
    }
}

__global__ void min_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmin(a[idx], b[idx]);
    }
}

__global__ void atan2_f64(const double* y, const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2(y[idx], x[idx]);
    }
}

// ============================================================================
// I32 Binary Operations
// ============================================================================

__global__ void add_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// I64 Binary Operations
// ============================================================================

__global__ void add_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// Broadcasting Binary Operations (strided access)
// ============================================================================

// ============================================================================
// Broadcast Kernels - Templated for all dtypes
// Thin wrappers call templated device functions
// ============================================================================

// F32 broadcast kernels
__global__ void add_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<float>);
}

__global__ void sub_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<float>);
}

__global__ void mul_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<float>);
}

__global__ void div_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<float>);
}

__global__ void pow_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<float>);
}

__global__ void max_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<float>);
}

__global__ void min_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<float>);
}

// F64 broadcast kernels
__global__ void add_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<double>);
}

__global__ void sub_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<double>);
}

__global__ void mul_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<double>);
}

__global__ void div_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<double>);
}

__global__ void pow_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<double>);
}

__global__ void max_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<double>);
}

__global__ void min_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<double>);
}

// F16 broadcast kernels
__global__ void add_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<__half>);
}

__global__ void sub_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<__half>);
}

__global__ void mul_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<__half>);
}

__global__ void div_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<__half>);
}

__global__ void pow_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<__half>);
}

__global__ void max_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<__half>);
}

__global__ void min_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<__half>);
}

// BF16 broadcast kernels
__global__ void add_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<__nv_bfloat16>);
}

__global__ void sub_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<__nv_bfloat16>);
}

__global__ void mul_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<__nv_bfloat16>);
}

__global__ void div_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<__nv_bfloat16>);
}

__global__ void pow_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<__nv_bfloat16>);
}

__global__ void max_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<__nv_bfloat16>);
}

__global__ void min_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<__nv_bfloat16>);
}

// I32 broadcast kernels
__global__ void add_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<int32_t>);
}

__global__ void sub_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<int32_t>);
}

__global__ void mul_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<int32_t>);
}

__global__ void div_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<int32_t>);
}

__global__ void pow_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<int32_t>);
}

__global__ void max_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<int32_t>);
}

__global__ void min_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<int32_t>);
}

// I64 broadcast kernels
__global__ void add_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<int64_t>);
}

__global__ void sub_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<int64_t>);
}

__global__ void mul_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<int64_t>);
}

__global__ void div_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<int64_t>);
}

__global__ void pow_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<int64_t>);
}

__global__ void max_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<int64_t>);
}

__global__ void min_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<int64_t>);
}

// FP8E4M3 broadcast kernels
__global__ void add_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<numr_fp8_e4m3>);
}

__global__ void sub_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<numr_fp8_e4m3>);
}

__global__ void mul_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<numr_fp8_e4m3>);
}

__global__ void div_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<numr_fp8_e4m3>);
}

__global__ void pow_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<numr_fp8_e4m3>);
}

__global__ void max_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<numr_fp8_e4m3>);
}

__global__ void min_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<numr_fp8_e4m3>);
}

// FP8E5M2 broadcast kernels
__global__ void add_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<numr_fp8_e5m2>);
}

__global__ void sub_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<numr_fp8_e5m2>);
}

__global__ void mul_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<numr_fp8_e5m2>);
}

__global__ void div_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<numr_fp8_e5m2>);
}

__global__ void pow_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<numr_fp8_e5m2>);
}

__global__ void max_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<numr_fp8_e5m2>);
}

__global__ void min_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<numr_fp8_e5m2>);
}

// ============================================================================
// FP8 E4M3 Binary Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
// ============================================================================

__global__ void add_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa + fb));
    }
}

__global__ void sub_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa - fb));
    }
}

__global__ void mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fb));
    }
}

__global__ void div_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa / fb));
    }
}

__global__ void pow_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(numr_pow_safe(fa, fb)));
    }
}

__global__ void max_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fa, fb)));
    }
}

__global__ void atan2_fp8_e4m3(const numr_fp8_e4m3* y, const numr_fp8_e4m3* x, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fy = fp8_e4m3_to_f32(y[idx].data);
        float fx = fp8_e4m3_to_f32(x[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(atan2f(fy, fx)));
    }
}

// ============================================================================
// FP8 E5M2 Binary Operations
// ============================================================================

__global__ void add_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa + fb));
    }
}

__global__ void sub_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa - fb));
    }
}

__global__ void mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fb));
    }
}

__global__ void div_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa / fb));
    }
}

__global__ void pow_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(numr_pow_safe(fa, fb)));
    }
}

__global__ void max_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fa, fb)));
    }
}

__global__ void atan2_fp8_e5m2(const numr_fp8_e5m2* y, const numr_fp8_e5m2* x, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fy = fp8_e5m2_to_f32(y[idx].data);
        float fx = fp8_e5m2_to_f32(x[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(atan2f(fy, fx)));
    }
}

// ============================================================================
// Logical AND - Boolean AND operation (input and output are u8)
// ============================================================================

__global__ void logical_and_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] && b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Logical OR - Boolean OR operation (input and output are u8)
// ============================================================================

__global__ void logical_or_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] || b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Logical XOR - Boolean XOR operation (input and output are u8)
// ============================================================================

__global__ void logical_xor_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ((a[idx] != 0) != (b[idx] != 0)) ? 1 : 0;
    }
}

// ============================================================================
// Complex64 (float2) Binary Operations
// ============================================================================

__global__ void add_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_add(a[idx], b[idx]);
    }
}

__global__ void sub_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_sub(a[idx], b[idx]);
    }
}

__global__ void mul_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_mul(a[idx], b[idx]);
    }
}

__global__ void div_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_div(a[idx], b[idx]);
    }
}

// ============================================================================
// Complex128 (double2) Binary Operations
// ============================================================================

__global__ void add_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_add(a[idx], b[idx]);
    }
}

__global__ void sub_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_sub(a[idx], b[idx]);
    }
}

__global__ void mul_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_mul(a[idx], b[idx]);
    }
}

__global__ void div_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_div(a[idx], b[idx]);
    }
}

// ============================================================================
// Inline broadcast kernels (CUDA-graph safe)
//
// These variants pass strides and shape as individual scalar u32 arguments
// baked into the kernel-parameter block rather than as device-memory pointers.
// This makes them safe for use inside CUDA graph capture regions: no H2D memcpy
// nodes are created, so no stale host pointers are baked into the graph.
//
// Naming convention: {op}_broadcast_{dtype}_inline
// ============================================================================

// Shared inline signature for all broadcast kernels.
// Includes precomputed magic/shift for fast division (replaces hardware div/mod).
#define BROADCAST_INLINE_ARGS \
    unsigned int as0, unsigned int as1, unsigned int as2, unsigned int as3, \
    unsigned int as4, unsigned int as5, unsigned int as6, unsigned int as7, \
    unsigned int bs0, unsigned int bs1, unsigned int bs2, unsigned int bs3, \
    unsigned int bs4, unsigned int bs5, unsigned int bs6, unsigned int bs7, \
    unsigned int sh0, unsigned int sh1, unsigned int sh2, unsigned int sh3, \
    unsigned int sh4, unsigned int sh5, unsigned int sh6, unsigned int sh7, \
    unsigned int mg0, unsigned int mg1, unsigned int mg2, unsigned int mg3, \
    unsigned int mg4, unsigned int mg5, unsigned int mg6, unsigned int mg7, \
    unsigned int ps0, unsigned int ps1, unsigned int ps2, unsigned int ps3, \
    unsigned int ps4, unsigned int ps5, unsigned int ps6, unsigned int ps7, \
    unsigned int ndim, unsigned int n

#define BROADCAST_INLINE_CALL \
    as0, as1, as2, as3, as4, as5, as6, as7, \
    bs0, bs1, bs2, bs3, bs4, bs5, bs6, bs7, \
    sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, \
    mg0, mg1, mg2, mg3, mg4, mg5, mg6, mg7, \
    ps0, ps1, ps2, ps3, ps4, ps5, ps6, ps7, \
    ndim, n

// Fast trailing-broadcast kernel signature (contiguous a, contiguous repeating b)
#define BROADCAST_FAST_TRAILING_ARGS \
    unsigned int b_magic, unsigned int b_shift, unsigned int b_numel, unsigned int n

#define BROADCAST_FAST_TRAILING_CALL \
    b_magic, b_shift, b_numel, n

// F32 inline broadcast kernels
__global__ void add_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<float>);
}
__global__ void sub_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<float>);
}
__global__ void mul_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<float>);
}
__global__ void div_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<float>);
}
__global__ void pow_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<float>);
}
__global__ void max_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<float>);
}
__global__ void min_broadcast_f32_inline(const float* a, const float* b, float* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<float>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<float>);
}

// F64 inline broadcast kernels
__global__ void add_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<double>);
}
__global__ void sub_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<double>);
}
__global__ void mul_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<double>);
}
__global__ void div_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<double>);
}
__global__ void pow_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<double>);
}
__global__ void max_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<double>);
}
__global__ void min_broadcast_f64_inline(const double* a, const double* b, double* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<double>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<double>);
}

// F16 inline broadcast kernels
__global__ void add_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<__half>);
}
__global__ void sub_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<__half>);
}
__global__ void mul_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<__half>);
}
__global__ void div_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<__half>);
}
__global__ void pow_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<__half>);
}
__global__ void max_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<__half>);
}
__global__ void min_broadcast_f16_inline(const __half* a, const __half* b, __half* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__half>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<__half>);
}

// BF16 inline broadcast kernels
__global__ void add_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<__nv_bfloat16>);
}
__global__ void sub_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<__nv_bfloat16>);
}
__global__ void mul_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<__nv_bfloat16>);
}
__global__ void div_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<__nv_bfloat16>);
}
__global__ void pow_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<__nv_bfloat16>);
}
__global__ void max_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<__nv_bfloat16>);
}
__global__ void min_broadcast_bf16_inline(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<__nv_bfloat16>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<__nv_bfloat16>);
}

// I32 inline broadcast kernels
__global__ void add_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<int32_t>);
}
__global__ void sub_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<int32_t>);
}
__global__ void mul_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<int32_t>);
}
__global__ void div_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<int32_t>);
}
__global__ void pow_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<int32_t>);
}
__global__ void max_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<int32_t>);
}
__global__ void min_broadcast_i32_inline(const int32_t* a, const int32_t* b, int32_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int32_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<int32_t>);
}

// I64 inline broadcast kernels
__global__ void add_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_add<int64_t>);
}
__global__ void sub_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_sub<int64_t>);
}
__global__ void mul_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_mul<int64_t>);
}
__global__ void div_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_div<int64_t>);
}
__global__ void pow_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_pow<int64_t>);
}
__global__ void max_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_max<int64_t>);
}
__global__ void min_broadcast_i64_inline(const int64_t* a, const int64_t* b, int64_t* out, BROADCAST_INLINE_ARGS) {
    broadcast_kernel_impl_inline<int64_t>(a, b, out, BROADCAST_INLINE_CALL, broadcast_min<int64_t>);
}

// ============================================================================
// Fast trailing-broadcast kernels
//
// Used when a is contiguous with the same shape as out, and b is a contiguous
// tensor that tiles along the leading dimensions (b[idx % b_numel]).
// b_magic + b_shift are precomputed magic-number divisors for b_numel.
// Naming: {op}_broadcast_fast_trailing_{dtype}
// ============================================================================

// F32 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<float>);
}
__global__ void sub_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<float>);
}
__global__ void mul_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<float>);
}
__global__ void div_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<float>);
}
__global__ void pow_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<float>);
}
__global__ void max_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<float>);
}
__global__ void min_broadcast_fast_trailing_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<float>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<float>);
}

// F64 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<double>);
}
__global__ void sub_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<double>);
}
__global__ void mul_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<double>);
}
__global__ void div_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<double>);
}
__global__ void pow_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<double>);
}
__global__ void max_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<double>);
}
__global__ void min_broadcast_fast_trailing_f64(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<double>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<double>);
}

// F16 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<__half>);
}
__global__ void sub_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<__half>);
}
__global__ void mul_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<__half>);
}
__global__ void div_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<__half>);
}
__global__ void pow_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<__half>);
}
__global__ void max_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<__half>);
}
__global__ void min_broadcast_fast_trailing_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__half>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<__half>);
}

// BF16 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<__nv_bfloat16>);
}
__global__ void sub_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<__nv_bfloat16>);
}
__global__ void mul_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<__nv_bfloat16>);
}
__global__ void div_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<__nv_bfloat16>);
}
__global__ void pow_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<__nv_bfloat16>);
}
__global__ void max_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<__nv_bfloat16>);
}
__global__ void min_broadcast_fast_trailing_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<__nv_bfloat16>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<__nv_bfloat16>);
}

// I32 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<int32_t>);
}
__global__ void sub_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<int32_t>);
}
__global__ void mul_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<int32_t>);
}
__global__ void div_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<int32_t>);
}
__global__ void pow_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<int32_t>);
}
__global__ void max_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<int32_t>);
}
__global__ void min_broadcast_fast_trailing_i32(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int32_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int32_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<int32_t>);
}

// I64 fast trailing-broadcast kernels
__global__ void add_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_add<int64_t>);
}
__global__ void sub_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_sub<int64_t>);
}
__global__ void mul_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_mul<int64_t>);
}
__global__ void div_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_div<int64_t>);
}
__global__ void pow_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_pow<int64_t>);
}
__global__ void max_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_max<int64_t>);
}
__global__ void min_broadcast_fast_trailing_i64(const int64_t* __restrict__ a, const int64_t* __restrict__ b, int64_t* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {
    broadcast_fast_trailing_impl<int64_t>(a, b, out, BROADCAST_FAST_TRAILING_CALL, broadcast_min<int64_t>);
}

} // extern "C"
