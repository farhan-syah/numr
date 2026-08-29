// Integer cumulative CUDA kernels: cumsum and cumprod (I64, I32, I16, I8, U64,
// U32, U16, U8).
//
// Split into their own translation unit for the same reason `reduce_int.cu` and
// `gemv_int.cu` were: the running total lives in `Numr128` rather than in a
// float register, and nothing else about the loop differs. There is no integer
// logsumexp, so `cumulative.cu` keeps that operation to itself.
//
// This file is PTX module "cumulative_int" (kernel_names::CUMULATIVE_INT_MODULE,
// selected by `cumulative_module` in loader.rs). The kernel names and the launch
// ABI are identical to the float module's, so the Rust launcher swaps only the
// module.
//
// Semantics, matching `runtime/cpu/kernels/cumulative.rs` with its `i128`
// `WideAcc` (`runtime/cpu/kernels/wide_acc.rs`):
//
//   cumsum   acc = saturating_add(acc, elem), narrowed and saturated at each
//            store.
//   cumprod  acc = saturating_mul(acc, elem), same store rule.
//
// Accumulators SATURATE; they do not wrap. A total that leaves the element
// type's range and later returns to it reports the true value, because only the
// store clamps - the 128-bit accumulator itself stays exact for any sum or
// product a scan can build. Elementwise ops (add, sub, mul) wrap instead; that
// line is drawn in `wide_acc.rs`.
//
// `Numr128From<T>` picks sign- or zero-extension from the element type. Routing
// a U64 element through the signed widening would record any value above
// LLONG_MAX as negative, so the widening is never chosen by hand here.

#include "numr128.cuh"

// ============================================================================
// Scan bodies
// ============================================================================

// One scan of `n` elements from `base`, stepping by `stride`. `IsProd` picks the
// identity and the combining operation at compile time, so one body covers both
// operations.
template<typename T, bool IsProd>
__device__ __forceinline__ void cum_int_scan(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int base,
    unsigned int stride,
    unsigned int n
) {
    Numr128 acc = numr128_from_i64(IsProd ? 1 : 0);
    for (unsigned int i = 0; i < n; i++) {
        unsigned int offset = base + i * stride;
        Numr128 v = Numr128From<T>::apply(input[offset]);
        acc = IsProd ? numr128_mul_sat(acc, v) : numr128_add_sat(acc, v);
        output[offset] = Numr128Narrow<T>::apply(acc);
    }
}

// Scan along the last dimension: one thread per contiguous segment.
template<typename T, bool IsProd>
__device__ void cum_int_simple_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;
    cum_int_scan<T, IsProd>(input, output, outer_idx * scan_size, 1u, scan_size);
}

// Scan along a non-last dimension: one thread per (outer, inner) pair.
template<typename T, bool IsProd>
__device__ void cum_int_strided_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;
    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;
    unsigned int base = outer_idx * scan_size * inner_size + inner_idx;
    cum_int_scan<T, IsProd>(input, output, base, inner_size, scan_size);
}

// ============================================================================
// extern "C" wrappers for Rust FFI
// ============================================================================

#define NUMR_CUMULATIVE_INT_ROW(T, SUF)                                         \
    __global__ void cumsum_##SUF(                                               \
        const T* in, T* out, unsigned int scan_size, unsigned int outer_size    \
    ) { cum_int_simple_impl<T, false>(in, out, scan_size, outer_size); }        \
    __global__ void cumsum_strided_##SUF(                                       \
        const T* in, T* out, unsigned int scan_size, unsigned int outer_size,   \
        unsigned int inner_size                                                 \
    ) { cum_int_strided_impl<T, false>(in, out, scan_size, outer_size, inner_size); } \
    __global__ void cumprod_##SUF(                                              \
        const T* in, T* out, unsigned int scan_size, unsigned int outer_size    \
    ) { cum_int_simple_impl<T, true>(in, out, scan_size, outer_size); }         \
    __global__ void cumprod_strided_##SUF(                                      \
        const T* in, T* out, unsigned int scan_size, unsigned int outer_size,   \
        unsigned int inner_size                                                 \
    ) { cum_int_strided_impl<T, true>(in, out, scan_size, outer_size, inner_size); }

extern "C" {

NUMR_CUMULATIVE_INT_ROW(long long, i64)
NUMR_CUMULATIVE_INT_ROW(int, i32)
NUMR_CUMULATIVE_INT_ROW(short, i16)
NUMR_CUMULATIVE_INT_ROW(signed char, i8)
NUMR_CUMULATIVE_INT_ROW(unsigned long long, u64)
NUMR_CUMULATIVE_INT_ROW(unsigned int, u32)
NUMR_CUMULATIVE_INT_ROW(unsigned short, u16)
NUMR_CUMULATIVE_INT_ROW(unsigned char, u8)

} // extern "C"
