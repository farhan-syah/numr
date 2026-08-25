//! Cumulative operations CUDA kernels
//!
//! Implements cumulative sum (cumsum), cumulative product (cumprod),
//! and log-sum-exp (logsumexp) operations.
//!
//! Uses block-level parallel scan for efficiency with large arrays.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <climits>
#include "dtype_traits.cuh"

// ============================================================================
// Constants
// ============================================================================

#define BLOCK_SIZE 256

// ============================================================================
// Cumulative Sum (Inclusive Scan) - Device Functions
// ============================================================================

// Simple sequential cumsum for small arrays or when scan dimension is last
template<typename T>
__device__ void cumsum_simple_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    T acc = T(0);
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = acc + input[base + i];
        output[base + i] = acc;
    }
}

// ----------------------------------------------------------------------------
// Integer cumsum with a wider accumulator
//
// A running total held in the element type wraps when it leaves that type's
// range, even when a later element brings it back to a value the output could
// represent. Accumulate in the next wider integer and clamp on store, matching
// the CPU kernels (which accumulate in i128 and saturate on narrowing).
//
// I64 and U64 have no wider integer available here, so they keep their native
// accumulator: `__int128` is not portable across the CUDA versions this builds
// against. That is a deliberate limit, documented in `cumsum_i64`/`cumsum_u64`.
// ----------------------------------------------------------------------------

// `lo` is 0 for the unsigned instantiations, where the low branch folds away.
template<typename T, typename Acc>
__device__ __forceinline__ T cumsum_saturate(Acc v, Acc lo, Acc hi) {
    return (T)(v < lo ? lo : (v > hi ? hi : v));
}

template<typename T, typename Acc>
__device__ void cumsum_simple_wide_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    Acc lo,
    Acc hi
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    Acc acc = (Acc)0;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = acc + (Acc)input[base + i];
        output[base + i] = cumsum_saturate<T, Acc>(acc, lo, hi);
    }
}

template<typename T, typename Acc>
__device__ void cumsum_strided_wide_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size,
    Acc lo,
    Acc hi
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    Acc acc = (Acc)0;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = acc + (Acc)input[offset];
        output[offset] = cumsum_saturate<T, Acc>(acc, lo, hi);
    }
}

// Strided cumsum for non-last dimension
template<typename T>
__device__ void cumsum_strided_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    T acc = T(0);
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = acc + input[offset];
        output[offset] = acc;
    }
}

// ============================================================================
// Cumulative Product (Inclusive Scan) - Device Functions
// ============================================================================

template<typename T>
__device__ void cumprod_simple_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    T acc = T(1);
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = acc * input[base + i];
        output[base + i] = acc;
    }
}

template<typename T>
__device__ void cumprod_strided_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    T acc = T(1);
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = acc * input[offset];
        output[offset] = acc;
    }
}

// ============================================================================
// Log-Sum-Exp (Numerically Stable Reduction) - Device Functions
// ============================================================================

// logsumexp = max(x) + log(sum(exp(x - max(x))))
// This is a reduction operation, not a scan

template<typename T>
__device__ void logsumexp_simple_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * reduce_size;

    // Step 1: Find max
    T max_val = input[base];
    for (unsigned int i = 1; i < reduce_size; i++) {
        T val = input[base + i];
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute sum(exp(x - max))
    T sum = T(0);
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum = sum + exp(float(input[base + i] - max_val));
    }

    // Step 3: Result = max + log(sum)
    output[outer_idx] = max_val + T(log(float(sum)));
}

template<typename T>
__device__ void logsumexp_strided_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    // Step 1: Find max along reduce dimension
    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    T max_val = input[first_offset];
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        T val = input[offset];
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute sum(exp(x - max))
    T sum = T(0);
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum = sum + exp(float(input[offset] - max_val));
    }

    // Step 3: Write result
    output[outer_idx * inner_size + inner_idx] = max_val + T(log(float(sum)));
}

// ============================================================================
// F64 specializations (use double math) - Device Functions
// ============================================================================

__device__ void logsumexp_simple_f64_impl(
    const double* __restrict__ input,
    double* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * reduce_size;

    double max_val = input[base];
    for (unsigned int i = 1; i < reduce_size; i++) {
        double val = input[base + i];
        if (val > max_val) max_val = val;
    }

    double sum = 0.0;
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum += exp(input[base + i] - max_val);
    }

    output[outer_idx] = max_val + log(sum);
}

__device__ void logsumexp_strided_f64_impl(
    const double* __restrict__ input,
    double* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    double max_val = input[first_offset];
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        double val = input[offset];
        if (val > max_val) max_val = val;
    }

    double sum = 0.0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum += exp(input[offset] - max_val);
    }

    output[outer_idx * inner_size + inner_idx] = max_val + log(sum);
}

// ============================================================================
// F16/BF16 Specializations (via F32 accumulation)
// ============================================================================

__device__ void cumsum_simple_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * scan_size;
    float acc = 0.0f;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc += __half2float(input[base + i]);
        output[base + i] = __float2half(acc);
    }
}

__device__ void cumsum_strided_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc += __half2float(input[offset]);
        output[offset] = __float2half(acc);
    }
}

__device__ void cumprod_simple_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * scan_size;
    float acc = 1.0f;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc *= __half2float(input[base + i]);
        output[base + i] = __float2half(acc);
    }
}

__device__ void cumprod_strided_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    float acc = 1.0f;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc *= __half2float(input[offset]);
        output[offset] = __float2half(acc);
    }
}

__device__ void cumsum_simple_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * scan_size;
    float acc = 0.0f;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc += __bfloat162float(input[base + i]);
        output[base + i] = __float2bfloat16(acc);
    }
}

__device__ void cumsum_strided_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    float acc = 0.0f;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc += __bfloat162float(input[offset]);
        output[offset] = __float2bfloat16(acc);
    }
}

__device__ void cumprod_simple_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * scan_size;
    float acc = 1.0f;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc *= __bfloat162float(input[base + i]);
        output[base + i] = __float2bfloat16(acc);
    }
}

__device__ void cumprod_strided_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    float acc = 1.0f;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc *= __bfloat162float(input[offset]);
        output[offset] = __float2bfloat16(acc);
    }
}

// ============================================================================
// FP8 Specializations (via F32 accumulation, byte-level load/store)
// ============================================================================

// Macro for FP8 cumulative kernels (cumsum/cumprod)
#define DEFINE_FP8_CUMOP_SIMPLE(name, fp8_suffix, load_macro, store_macro, identity, op) \
__device__ void name##_simple_##fp8_suffix##_impl( \
    const unsigned char* __restrict__ input, \
    unsigned char* __restrict__ output, \
    unsigned int scan_size, \
    unsigned int outer_size \
) { \
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (outer_idx >= outer_size) return; \
    unsigned int base = outer_idx * scan_size; \
    float acc = identity; \
    for (unsigned int i = 0; i < scan_size; i++) { \
        float v = load_macro(input, base + i); \
        acc = acc op v; \
        store_macro(output, base + i, acc); \
    } \
}

#define DEFINE_FP8_CUMOP_STRIDED(name, fp8_suffix, load_macro, store_macro, identity, op) \
__device__ void name##_strided_##fp8_suffix##_impl( \
    const unsigned char* __restrict__ input, \
    unsigned char* __restrict__ output, \
    unsigned int scan_size, \
    unsigned int outer_size, \
    unsigned int inner_size \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total_inner = outer_size * inner_size; \
    if (idx >= total_inner) return; \
    unsigned int outer_idx = idx / inner_size; \
    unsigned int inner_idx = idx % inner_size; \
    float acc = identity; \
    for (unsigned int s = 0; s < scan_size; s++) { \
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx; \
        float v = load_macro(input, offset); \
        acc = acc op v; \
        store_macro(output, offset, acc); \
    } \
}

DEFINE_FP8_CUMOP_SIMPLE(cumsum, fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3, 0.0f, +)
DEFINE_FP8_CUMOP_SIMPLE(cumsum, fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2, 0.0f, +)
DEFINE_FP8_CUMOP_SIMPLE(cumprod, fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3, 1.0f, *)
DEFINE_FP8_CUMOP_SIMPLE(cumprod, fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2, 1.0f, *)

DEFINE_FP8_CUMOP_STRIDED(cumsum, fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3, 0.0f, +)
DEFINE_FP8_CUMOP_STRIDED(cumsum, fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2, 0.0f, +)
DEFINE_FP8_CUMOP_STRIDED(cumprod, fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3, 1.0f, *)
DEFINE_FP8_CUMOP_STRIDED(cumprod, fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2, 1.0f, *)

// FP8 logsumexp
#define DEFINE_FP8_LOGSUMEXP_SIMPLE(fp8_suffix, load_macro, store_macro) \
__device__ void logsumexp_simple_##fp8_suffix##_impl( \
    const unsigned char* __restrict__ input, \
    unsigned char* __restrict__ output, \
    unsigned int reduce_size, \
    unsigned int outer_size \
) { \
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (outer_idx >= outer_size) return; \
    unsigned int base = outer_idx * reduce_size; \
    float max_val = load_macro(input, base); \
    for (unsigned int i = 1; i < reduce_size; i++) { \
        float v = load_macro(input, base + i); \
        if (v > max_val) max_val = v; \
    } \
    float sum = 0.0f; \
    for (unsigned int i = 0; i < reduce_size; i++) { \
        sum += expf(load_macro(input, base + i) - max_val); \
    } \
    store_macro(output, outer_idx, max_val + logf(sum)); \
}

#define DEFINE_FP8_LOGSUMEXP_STRIDED(fp8_suffix, load_macro, store_macro) \
__device__ void logsumexp_strided_##fp8_suffix##_impl( \
    const unsigned char* __restrict__ input, \
    unsigned char* __restrict__ output, \
    unsigned int reduce_size, \
    unsigned int outer_size, \
    unsigned int inner_size \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total_inner = outer_size * inner_size; \
    if (idx >= total_inner) return; \
    unsigned int outer_idx = idx / inner_size; \
    unsigned int inner_idx = idx % inner_size; \
    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx; \
    float max_val = load_macro(input, first_offset); \
    for (unsigned int r = 1; r < reduce_size; r++) { \
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx; \
        float v = load_macro(input, offset); \
        if (v > max_val) max_val = v; \
    } \
    float sum = 0.0f; \
    for (unsigned int r = 0; r < reduce_size; r++) { \
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx; \
        sum += expf(load_macro(input, offset) - max_val); \
    } \
    store_macro(output, outer_idx * inner_size + inner_idx, max_val + logf(sum)); \
}

DEFINE_FP8_LOGSUMEXP_SIMPLE(fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3)
DEFINE_FP8_LOGSUMEXP_SIMPLE(fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2)
DEFINE_FP8_LOGSUMEXP_STRIDED(fp8_e4m3, LOAD_FP8_E4M3, STORE_FP8_E4M3)
DEFINE_FP8_LOGSUMEXP_STRIDED(fp8_e5m2, LOAD_FP8_E5M2, STORE_FP8_E5M2)

// F16/BF16 logsumexp
__device__ void logsumexp_simple_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * reduce_size;
    float max_val = __half2float(input[base]);
    for (unsigned int i = 1; i < reduce_size; i++) {
        float v = __half2float(input[base + i]);
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum += expf(__half2float(input[base + i]) - max_val);
    }
    output[outer_idx] = __float2half(max_val + logf(sum));
}

__device__ void logsumexp_strided_f16_impl(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    float max_val = __half2float(input[first_offset]);
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        float v = __half2float(input[offset]);
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum += expf(__half2float(input[offset]) - max_val);
    }
    output[outer_idx * inner_size + inner_idx] = __float2half(max_val + logf(sum));
}

__device__ void logsumexp_simple_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    unsigned int base = outer_idx * reduce_size;
    float max_val = __bfloat162float(input[base]);
    for (unsigned int i = 1; i < reduce_size; i++) {
        float v = __bfloat162float(input[base + i]);
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum += expf(__bfloat162float(input[base + i]) - max_val);
    }
    output[outer_idx] = __float2bfloat16(max_val + logf(sum));
}

__device__ void logsumexp_strided_bf16_impl(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    float max_val = __bfloat162float(input[first_offset]);
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        float v = __bfloat162float(input[offset]);
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum += expf(__bfloat162float(input[offset]) - max_val);
    }
    output[outer_idx * inner_size + inner_idx] = __float2bfloat16(max_val + logf(sum));
}

// ============================================================================
// Extern "C" Wrapper Kernels
// ============================================================================

extern "C" {

// ===== Cumulative Sum =====

__global__ void cumsum_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_wide_impl<int, long long>(in, out, scan_size, outer_size, (long long)INT_MIN, (long long)INT_MAX);
}

// I64 has no wider integer accumulator available in this kernel, so it keeps a
// native one: an i64 total that overflows cannot be represented by the i64
// output either.
__global__ void cumsum_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_wide_impl<unsigned int, unsigned long long>(in, out, scan_size, outer_size, (unsigned long long)0, (unsigned long long)UINT_MAX);
}

// U64: same limit as I64 above.
__global__ void cumsum_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_f16(const __half* in, __half* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_f16_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_bf16_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_fp8_e4m3_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_fp8_e5m2_impl(in, out, scan_size, outer_size);
}

// Strided versions
__global__ void cumsum_strided_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_wide_impl<int, long long>(in, out, scan_size, outer_size, inner_size, (long long)INT_MIN, (long long)INT_MAX);
}

__global__ void cumsum_strided_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_wide_impl<unsigned int, unsigned long long>(in, out, scan_size, outer_size, inner_size, (unsigned long long)0, (unsigned long long)UINT_MAX);
}

__global__ void cumsum_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_f16(const __half* in, __half* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_f16_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_bf16_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_fp8_e4m3_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_fp8_e5m2_impl(in, out, scan_size, outer_size, inner_size);
}

// ===== Cumulative Product =====

__global__ void cumprod_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_f16(const __half* in, __half* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_f16_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_bf16_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_fp8_e4m3_impl(in, out, scan_size, outer_size);
}

__global__ void cumprod_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_fp8_e5m2_impl(in, out, scan_size, outer_size);
}

// Strided versions
__global__ void cumprod_strided_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_f16(const __half* in, __half* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_f16_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_bf16_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_fp8_e4m3_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_fp8_e5m2_impl(in, out, scan_size, outer_size, inner_size);
}

// ===== Log-Sum-Exp =====

__global__ void logsumexp_f32(const float* in, float* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_impl(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_f64(const double* in, double* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_f64_impl(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_f16(const __half* in, __half* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_f16_impl(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_bf16_impl(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_fp8_e4m3_impl(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_fp8_e5m2_impl(in, out, reduce_size, outer_size);
}

// Strided versions
__global__ void logsumexp_strided_f32(const float* in, float* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_impl(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_f64(const double* in, double* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_f64_impl(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_f16(const __half* in, __half* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_f16_impl(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_bf16_impl(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_fp8_e4m3(const unsigned char* in, unsigned char* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_fp8_e4m3_impl(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_fp8_e5m2(const unsigned char* in, unsigned char* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_fp8_e5m2_impl(in, out, reduce_size, outer_size, inner_size);
}

} // extern "C"
