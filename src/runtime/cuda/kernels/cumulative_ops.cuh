// Shared device templates for the floating-point cumulative kernels
// (cumsum, cumprod, logsumexp).
//
// Every float dtype runs the same three loops; only the storage type and the
// accumulator differ. A policy struct supplies those, so `cumulative.cu` holds
// nothing but one `extern "C"` row per dtype.
//
// Accumulator per storage type, matching the CPU kernels' `WideAcc`
// (`runtime/cpu/kernels/wide_acc.rs`):
//
//   f32                      float
//   f64                      double
//   f16, bf16, fp8_e4m3/e5m2 float - a running total held in a narrow float
//                            stops growing once its spacing exceeds twice the
//                            increment, so the accumulator must be wider.
//
// Integer cumsum/cumprod are NOT here: they accumulate in `Numr128` and live in
// their own translation unit, `cumulative_int.cu`.

#ifndef NUMR_CUMULATIVE_OPS_CUH
#define NUMR_CUMULATIVE_OPS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Storage policies
// ============================================================================
// `S` is the element type in the kernel signature, `A` the accumulator.
// `load`/`store` are the only places a conversion happens.

struct CumF32 {
    typedef float S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct CumF64 {
    typedef double S;
    typedef double A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return p[i]; }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = v; }
};

struct CumF16 {
    typedef __half S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __half2float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2half(v); }
};

struct CumBF16 {
    typedef __nv_bfloat16 S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return __bfloat162float(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = __float2bfloat16(v); }
};

// The FP8 kernels take `unsigned char*`, so both formats share a storage type
// and are told apart by the policy rather than by the pointer type.
struct CumFp8E4M3 {
    typedef unsigned char S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return fp8_e4m3_to_f32(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = f32_to_fp8_e4m3(v); }
};

struct CumFp8E5M2 {
    typedef unsigned char S;
    typedef float A;
    static __device__ __forceinline__ A load(const S* p, unsigned int i) { return fp8_e5m2_to_f32(p[i]); }
    static __device__ __forceinline__ void store(S* p, unsigned int i, A v) { p[i] = f32_to_fp8_e5m2(v); }
};

// exp/log at the accumulator's own precision: a float accumulator must not pay
// for double-precision transcendentals, and a double one must not lose bits.
__device__ __forceinline__ float numr_cum_exp(float v) { return expf(v); }
__device__ __forceinline__ double numr_cum_exp(double v) { return exp(v); }
__device__ __forceinline__ float numr_cum_log(float v) { return logf(v); }
__device__ __forceinline__ double numr_cum_log(double v) { return log(v); }

// ============================================================================
// Inclusive scan (cumsum / cumprod)
// ============================================================================

// One scan of `n` elements from `base`, stepping by `stride`. `IsProd` picks
// the identity and the combining operation at compile time, so one body covers
// both operations.
template<typename P, bool IsProd>
__device__ __forceinline__ void cum_scan(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    unsigned int base,
    unsigned int stride,
    unsigned int n
) {
    typedef typename P::A A;
    A acc = IsProd ? (A)1 : (A)0;
    for (unsigned int i = 0; i < n; i++) {
        unsigned int offset = base + i * stride;
        A v = P::load(input, offset);
        acc = IsProd ? (acc * v) : (acc + v);
        P::store(output, offset, acc);
    }
}

// Scan along the last dimension: one thread per contiguous segment.
template<typename P, bool IsProd>
__device__ void cum_simple_impl(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    cum_scan<P, IsProd>(input, output, outer_idx * scan_size, 1u, scan_size);
}

// Scan along a non-last dimension: one thread per (outer, inner) pair.
template<typename P, bool IsProd>
__device__ void cum_strided_impl(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int base = outer_idx * scan_size * inner_size + inner_idx;
    cum_scan<P, IsProd>(input, output, base, inner_size, scan_size);
}

// ============================================================================
// Log-sum-exp
// ============================================================================
// logsumexp = max(x) + log(sum(exp(x - max(x)))). This is a reduction, not a
// scan: it writes one value per segment. Subtracting the max first is what
// keeps `exp` from overflowing.

template<typename P>
__device__ __forceinline__ typename P::A logsumexp_reduce(
    const typename P::S* __restrict__ input,
    unsigned int base,
    unsigned int stride,
    unsigned int n
) {
    typedef typename P::A A;
    A max_val = P::load(input, base);
    for (unsigned int i = 1; i < n; i++) {
        A v = P::load(input, base + i * stride);
        if (v > max_val) max_val = v;
    }

    A sum = (A)0;
    for (unsigned int i = 0; i < n; i++) {
        sum = sum + numr_cum_exp(P::load(input, base + i * stride) - max_val);
    }

    return max_val + numr_cum_log(sum);
}

template<typename P>
__device__ void logsumexp_simple_impl(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    P::store(output, outer_idx,
             logsumexp_reduce<P>(input, outer_idx * reduce_size, 1u, reduce_size));
}

template<typename P>
__device__ void logsumexp_strided_impl(
    const typename P::S* __restrict__ input,
    typename P::S* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int base = outer_idx * reduce_size * inner_size + inner_idx;
    P::store(output, outer_idx * inner_size + inner_idx,
             logsumexp_reduce<P>(input, base, inner_size, reduce_size));
}

#endif // NUMR_CUMULATIVE_OPS_CUH
