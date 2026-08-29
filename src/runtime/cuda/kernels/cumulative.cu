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
#include "numr128.cuh"

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
// I32 and U32 widen into a native `long long` / `unsigned long long`
// accumulator below. I64 and U64 have no native type wider than 64 bits
// here - `__int128` is not portable across the CUDA versions this builds
// against - so they use the two-limb accumulators further down instead:
// `Numr128` (signed, for I64) and a saturating add on `unsigned long long`
// itself (for U64, see the comment above `cumsum_u64_sat_add`).
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

// ----------------------------------------------------------------------------
// I64 cumsum: accumulate in the shared 128-bit accumulator.
//
// `long long` cannot widen further natively, so the running total lives in
// `Numr128` (see `numr128.cuh`). 128 bits holds the sum of every I64 element a
// scan can hold without itself overflowing, so this matches the CPU kernel's
// i128 accumulator: the accumulator never saturates mid-scan, only the
// narrow-back-to-i64 store does, so a total that overflows I64 and later
// returns to I64's range reports the true value instead of a wrong-sign
// wrapped one.
// ----------------------------------------------------------------------------

__device__ void cumsum_simple_i64_wide_impl(
    const long long* __restrict__ input,
    long long* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    Numr128 acc = numr128_from_i64(0);
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = numr128_add(acc, numr128_from_i64(input[base + i]));
        output[base + i] = numr128_to_i64_sat(acc);
    }
}

__device__ void cumsum_strided_i64_wide_impl(
    const long long* __restrict__ input,
    long long* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    Numr128 acc = numr128_from_i64(0);
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = numr128_add(acc, numr128_from_i64(input[offset]));
        output[offset] = numr128_to_i64_sat(acc);
    }
}

// ----------------------------------------------------------------------------
// U64 cumsum: per-step saturating add
//
// Unsigned cumsum inputs never go negative, so the running total is
// monotonic: once it exceeds u64::MAX it can never come back into range, so a
// per-step saturating add matches CPU's wide accumulator exactly - unlike the
// signed case, there is no "overflow and later recover" sequence to lose.
// Same reasoning as `numr_u32_sat_add` in
// `runtime/wgpu/shaders/int_saturate.wgsl`, one limb size up.
// ----------------------------------------------------------------------------

__device__ __forceinline__ unsigned long long cumsum_u64_sat_add(unsigned long long a, unsigned long long b) {
    if (a > ULLONG_MAX - b) {
        return ULLONG_MAX;
    }
    return a + b;
}

__device__ void cumsum_simple_u64_wide_impl(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    unsigned long long acc = 0ULL;
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = cumsum_u64_sat_add(acc, input[base + i]);
        output[base + i] = acc;
    }
}

__device__ void cumsum_strided_u64_wide_impl(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    unsigned long long acc = 0ULL;
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = cumsum_u64_sat_add(acc, input[offset]);
        output[offset] = acc;
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

// ----------------------------------------------------------------------------
// Integer cumprod: exact magnitude plus sign, saturating on store
//
// The output must be the true mathematical product clamped to the element
// type's range, matching the CPU kernel's i128 accumulator (`WideAcc` in
// `runtime/cpu/kernels/wide_acc.rs`). WebGPU's cumprod shaders
// (`runtime/wgpu/shaders/int_saturate.wgsl`) track the same magnitude-plus-sign
// state division-free, since WGSL cannot divide inside the loop. A per-step
// saturating multiply does not give that: once it clamps to the maximum, a
// later negative factor reports `-MAX` where the true product's clamp is
// `MIN`.
//
// No wide accumulator is needed, because integer products only ever grow.
// Multiplying by 0 pins the true product at 0 forever after, and multiplying
// by any factor of magnitude >= 1 never shrinks the magnitude. So once the
// true magnitude leaves the range it can never come back, and from there the
// clamped answer depends only on the sign. Three pieces of O(1) state carry
// that: `zero_seen`, `saturated`, and the sign parity of the negative factors.
// `__int128` is banned here and is not needed either.
// ----------------------------------------------------------------------------

__device__ __forceinline__ bool numr_is_negative(int v) { return v < 0; }
__device__ __forceinline__ bool numr_is_negative(long long v) { return v < 0; }
__device__ __forceinline__ bool numr_is_negative(unsigned int) { return false; }
__device__ __forceinline__ bool numr_is_negative(unsigned long long) { return false; }

// Magnitude as an unsigned value. The unsigned negation is what makes the most
// negative input (whose magnitude has no signed representation) come out right.
__device__ __forceinline__ unsigned int numr_magnitude(int v) {
    unsigned int b = (unsigned int)v;
    return (v < 0) ? (0u - b) : b;
}
__device__ __forceinline__ unsigned long long numr_magnitude(long long v) {
    unsigned long long b = (unsigned long long)v;
    return (v < 0) ? (0ULL - b) : b;
}
__device__ __forceinline__ unsigned int numr_magnitude(unsigned int v) { return v; }
__device__ __forceinline__ unsigned long long numr_magnitude(unsigned long long v) { return v; }

// One scan of `scan_size` elements starting at `base`, stepping by `stride`.
// `limit` is the largest magnitude the element type can represent under either
// sign: `hi` for an unsigned type, `hi + 1` for a signed one.
template<typename T, typename U>
__device__ void cumprod_int_scan(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int base,
    unsigned int stride,
    unsigned int scan_size,
    T lo,
    T hi,
    U limit
) {
    U mag = (U)1;
    bool negative = false;
    bool zero_seen = false;
    bool saturated = false;

    for (unsigned int i = 0; i < scan_size; i++) {
        unsigned int offset = base + i * stride;
        T v = input[offset];

        if (!zero_seen) {
            if (v == (T)0) {
                zero_seen = true;
            } else {
                if (numr_is_negative(v)) {
                    negative = !negative;
                }
                if (!saturated) {
                    U m = numr_magnitude(v);
                    // Division is the overflow check here; CUDA has no cheap
                    // wide multiply for every width and division is allowed.
                    if (mag > limit / m) {
                        saturated = true;
                    } else {
                        mag = mag * m;
                    }
                }
            }
        }

        T result;
        if (zero_seen) {
            result = (T)0;
        } else if (saturated || mag > (U)hi) {
            // `mag == hi + 1` is representable only as `lo`, so it lands in
            // this branch and comes out right for both signs.
            result = negative ? lo : hi;
        } else {
            result = negative ? (T)((T)0 - (T)mag) : (T)mag;
        }
        output[offset] = result;
    }
}

template<typename T, typename U>
__device__ void cumprod_simple_int_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    T lo,
    T hi,
    U limit
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    cumprod_int_scan<T, U>(input, output, outer_idx * scan_size, 1u, scan_size, lo, hi, limit);
}

template<typename T, typename U>
__device__ void cumprod_strided_int_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size,
    T lo,
    T hi,
    U limit
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int base = outer_idx * scan_size * inner_size + inner_idx;

    cumprod_int_scan<T, U>(input, output, base, inner_size, scan_size, lo, hi, limit);
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

// I64: two-limb 128-bit accumulator (see `Numr128` in `numr128.cuh`), saturating on narrow.
__global__ void cumsum_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_i64_wide_impl(in, out, scan_size, outer_size);
}

__global__ void cumsum_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_wide_impl<unsigned int, unsigned long long>(in, out, scan_size, outer_size, (unsigned long long)0, (unsigned long long)UINT_MAX);
}

// U64: per-step saturating add (see `cumsum_u64_sat_add` above).
__global__ void cumsum_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_u64_wide_impl(in, out, scan_size, outer_size);
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
    cumsum_strided_i64_wide_impl(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_wide_impl<unsigned int, unsigned long long>(in, out, scan_size, outer_size, inner_size, (unsigned long long)0, (unsigned long long)UINT_MAX);
}

__global__ void cumsum_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_u64_wide_impl(in, out, scan_size, outer_size, inner_size);
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
    cumprod_simple_int_impl<int, unsigned int>(in, out, scan_size, outer_size, INT_MIN, INT_MAX, (unsigned int)INT_MAX + 1u);
}

__global__ void cumprod_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_int_impl<long long, unsigned long long>(in, out, scan_size, outer_size, LLONG_MIN, LLONG_MAX, (unsigned long long)LLONG_MAX + 1ULL);
}

__global__ void cumprod_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_int_impl<unsigned int, unsigned int>(in, out, scan_size, outer_size, 0u, UINT_MAX, UINT_MAX);
}

__global__ void cumprod_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_int_impl<unsigned long long, unsigned long long>(in, out, scan_size, outer_size, 0ULL, ULLONG_MAX, ULLONG_MAX);
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
    cumprod_strided_int_impl<int, unsigned int>(in, out, scan_size, outer_size, inner_size, INT_MIN, INT_MAX, (unsigned int)INT_MAX + 1u);
}

__global__ void cumprod_strided_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_int_impl<long long, unsigned long long>(in, out, scan_size, outer_size, inner_size, LLONG_MIN, LLONG_MAX, (unsigned long long)LLONG_MAX + 1ULL);
}

__global__ void cumprod_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_int_impl<unsigned int, unsigned int>(in, out, scan_size, outer_size, inner_size, 0u, UINT_MAX, UINT_MAX);
}

__global__ void cumprod_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_int_impl<unsigned long long, unsigned long long>(in, out, scan_size, outer_size, inner_size, 0ULL, ULLONG_MAX, ULLONG_MAX);
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
