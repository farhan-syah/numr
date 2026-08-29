// Shared machinery for the element-wise binary kernels instantiated in
// binary.cu: one device function per operation, four kernel-body templates, and
// the macros that expand one dtype into a full row of `extern "C"` wrappers.
//
// Numerical conventions, all of them matching the CPU reference in
// src/runtime/cpu/kernels/binary.rs:
//
//  * Integer add/sub/mul WRAP. Elementwise ops wrap, accumulators saturate —
//    the convention documented in src/runtime/cpu/kernels/wide_acc.rs — and
//    CPU spells it out as wrapping_add/wrapping_sub/wrapping_mul.
//  * Signed integer overflow is undefined behaviour in C++, so the wrapping
//    arithmetic runs in the dtype's unsigned counterpart and converts back.
//    That conversion is modulo 2^N, which is exactly what Rust's wrapping_*
//    produces. Inputs that do not overflow are unaffected.
//  * Integer division by zero yields 0, and INT_MIN / -1 yields INT_MIN
//    (Rust's wrapping_div). A raw `a / b` would trap or overflow instead.
//  * Integer pow uses the exact, saturating numr_ipow from ipow.cuh. Overflow
//    saturates there, which is deliberate: pow's result is an accumulator.
//  * max/min are written as `a > b ? a : b`, so a NaN second operand
//    propagates. fmaxf would return the first operand instead and disagree
//    with CPU.

#ifndef NUMR_BINARY_OPS_CUH
#define NUMR_BINARY_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "ipow.cuh"

// ============================================================================
// Per-operation device functions
// ============================================================================

template<typename T> __device__ __forceinline__ T binop_add(T a, T b) { return a + b; }
template<typename T> __device__ __forceinline__ T binop_sub(T a, T b) { return a - b; }
template<typename T> __device__ __forceinline__ T binop_mul(T a, T b) { return a * b; }
template<typename T> __device__ __forceinline__ T binop_div(T a, T b) { return a / b; }
template<typename T> __device__ __forceinline__ T binop_max(T a, T b) { return a > b ? a : b; }
template<typename T> __device__ __forceinline__ T binop_min(T a, T b) { return a < b ? a : b; }
template<typename T> __device__ __forceinline__ T binop_pow(T a, T b) {
    return (T)numr_pow_safe((float)a, (float)b);
}
template<> __device__ __forceinline__ double binop_pow(double a, double b) {
    return numr_pow_safe(a, b);
}

// F16 and BF16 raise in F32: the half-width intrinsics have no pow.
template<> __device__ __forceinline__ __half binop_pow(__half a, __half b) {
    return __float2half(numr_pow_safe(__half2float(a), __half2float(b)));
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_pow(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(numr_pow_safe(__bfloat162float(a), __bfloat162float(b)));
}

// BF16 arithmetic intrinsics need SM 8.0+. Below that the operands round-trip
// through F32, which is what the pre-Ampere path has always done.
#if __CUDA_ARCH__ >= 800
#define NUMR_BF16_BIN(INTRIN, OP, a, b) INTRIN(a, b)
#define NUMR_BF16_CMP(INTRIN, OP, a, b) INTRIN(a, b)
#else
#define NUMR_BF16_BIN(INTRIN, OP, a, b) \
    __float2bfloat16(__bfloat162float(a) OP __bfloat162float(b))
#define NUMR_BF16_CMP(INTRIN, OP, a, b) (__bfloat162float(a) OP __bfloat162float(b))
#endif

template<> __device__ __forceinline__ __nv_bfloat16 binop_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_BIN(__hadd, +, a, b);
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_sub(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_BIN(__hsub, -, a, b);
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_BIN(__hmul, *, a, b);
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_div(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_BIN(__hdiv, /, a, b);
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_max(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_CMP(__hgt, >, a, b) ? a : b;
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_min(__nv_bfloat16 a, __nv_bfloat16 b) {
    return NUMR_BF16_CMP(__hlt, <, a, b) ? a : b;
}

// FP8 has no arithmetic of its own: decode to F32, compute, re-encode. max/min
// return the original operand rather than a re-encoded one, so the value is
// bit-preserved.
#define NUMR_BINOP_FP8_SPEC(T, TO_F32, FROM_F32)                                \
    template<> __device__ __forceinline__ T binop_add(T a, T b) {               \
        return T(FROM_F32(TO_F32(a.data) + TO_F32(b.data)));                    \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_sub(T a, T b) {               \
        return T(FROM_F32(TO_F32(a.data) - TO_F32(b.data)));                    \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_mul(T a, T b) {               \
        return T(FROM_F32(TO_F32(a.data) * TO_F32(b.data)));                    \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_div(T a, T b) {               \
        return T(FROM_F32(TO_F32(a.data) / TO_F32(b.data)));                    \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_pow(T a, T b) {               \
        return T(FROM_F32(numr_pow_safe(TO_F32(a.data), TO_F32(b.data))));      \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_max(T a, T b) {               \
        return (TO_F32(a.data) > TO_F32(b.data)) ? a : b;                       \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_min(T a, T b) {               \
        return (TO_F32(a.data) < TO_F32(b.data)) ? a : b;                       \
    }

NUMR_BINOP_FP8_SPEC(numr_fp8_e4m3, fp8_e4m3_to_f32, f32_to_fp8_e4m3)
NUMR_BINOP_FP8_SPEC(numr_fp8_e5m2, fp8_e5m2_to_f32, f32_to_fp8_e5m2)

// Integer add/sub/mul/pow. See the wrapping and saturation notes at the top of
// this file. max/min keep the primary template: integer comparison is exact.
#define NUMR_BINOP_INT_SPEC(T)                                                  \
    template<> __device__ __forceinline__ T binop_add(T a, T b) {               \
        typedef numr_ipow_traits<T>::U U;                                       \
        return (T)(U)((U)a + (U)b);                                             \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_sub(T a, T b) {               \
        typedef numr_ipow_traits<T>::U U;                                       \
        return (T)(U)((U)a - (U)b);                                             \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_mul(T a, T b) {               \
        typedef numr_ipow_traits<T>::U U;                                       \
        return (T)(U)((U)a * (U)b);                                             \
    }                                                                           \
    template<> __device__ __forceinline__ T binop_pow(T a, T b) {               \
        return numr_ipow<T>(a, b);                                              \
    }

// A signed divisor of -1 overflows on INT_MIN; CPU's wrapping_div answers
// INT_MIN, which is the wrapping negation. The guard is signed-only: on an
// unsigned dtype (T)-1 is the maximum value and `a / max` is an ordinary
// division.
#define NUMR_BINOP_INT_DIV_SIGNED(T)                                            \
    template<> __device__ __forceinline__ T binop_div(T a, T b) {               \
        typedef numr_ipow_traits<T>::U U;                                       \
        if (b == (T)0) return (T)0;                                             \
        if (b == (T)(-1)) return (T)(U)((U)0 - (U)a);                           \
        return a / b;                                                           \
    }

#define NUMR_BINOP_INT_DIV_UNSIGNED(T)                                          \
    template<> __device__ __forceinline__ T binop_div(T a, T b) {               \
        if (b == (T)0) return (T)0;                                             \
        return a / b;                                                           \
    }

#define NUMR_BINOP_SIGNED_SPEC(T) NUMR_BINOP_INT_SPEC(T) NUMR_BINOP_INT_DIV_SIGNED(T)
#define NUMR_BINOP_UNSIGNED_SPEC(T) NUMR_BINOP_INT_SPEC(T) NUMR_BINOP_INT_DIV_UNSIGNED(T)

NUMR_BINOP_SIGNED_SPEC(int8_t)
NUMR_BINOP_SIGNED_SPEC(int16_t)
NUMR_BINOP_SIGNED_SPEC(int32_t)
NUMR_BINOP_SIGNED_SPEC(int64_t)
NUMR_BINOP_UNSIGNED_SPEC(uint8_t)
NUMR_BINOP_UNSIGNED_SPEC(uint16_t)
NUMR_BINOP_UNSIGNED_SPEC(uint32_t)
NUMR_BINOP_UNSIGNED_SPEC(uint64_t)

// ============================================================================
// Kernel body templates
// ============================================================================

template<typename T, typename OpFunc>
__device__ __forceinline__ void binary_elementwise_impl(
    const T* a, const T* b, T* out, unsigned int n, OpFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

// Pointer-based broadcast: strides and shape live in device memory.
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

// CUDA-graph-safe inline broadcast kernels pass strides and shape as individual
// scalar arguments baked into the kernel-parameter block rather than as
// device-memory pointers. This eliminates the H2D memcpy nodes that the
// pointer-based variant creates during graph capture — those nodes encode the
// host-side Vec<u32> addresses, which become stale (dangling) on graph replay,
// causing CUDA_ERROR_ILLEGAL_ADDRESS.
//
// Up to 8 dimensions are supported. Unused trailing dimensions must be
// zero-padded by the caller.
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

// ============================================================================
// Kernel-parameter macros
// ============================================================================

// Pointer-based broadcast signature. out_strides is unused by the kernel body
// but stays in the signature: the host passes it.
#define BROADCAST_PTR_ARGS(T)                                                   \
    const T* a, const T* b, T* out,                                             \
    const unsigned int* a_strides, const unsigned int* b_strides,               \
    const unsigned int* out_strides,                                            \
    const unsigned int* shape, unsigned int ndim, unsigned int n

#define BROADCAST_PTR_CALL a, b, out, a_strides, b_strides, shape, ndim, n

// Inline (CUDA-graph safe) signature, with precomputed magic/shift per dim.
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

// Fast trailing-broadcast signature (contiguous a, contiguous repeating b)
#define BROADCAST_FAST_TRAILING_ARGS \
    unsigned int b_magic, unsigned int b_shift, unsigned int b_numel, unsigned int n

#define BROADCAST_FAST_TRAILING_CALL b_magic, b_shift, b_numel, n

// ============================================================================
// Instantiation macros
// ============================================================================
// NUMR_BINARY_OP emits the four kernels of one (operation, dtype) pair:
//   {op}_{suffix}
//   {op}_broadcast_{suffix}
//   {op}_broadcast_{suffix}_inline
//   {op}_broadcast_fast_trailing_{suffix}
// The suffixes are the ones dtype_suffix() produces in
// src/runtime/cuda/kernels/loader.rs.

#define NUMR_BINARY_OP(T, S, OP)                                                \
    __global__ void OP##_##S(const T* a, const T* b, T* out, unsigned int n) {  \
        binary_elementwise_impl<T>(a, b, out, n, binop_##OP<T>);                \
    }                                                                           \
    __global__ void OP##_broadcast_##S(BROADCAST_PTR_ARGS(T)) {                 \
        broadcast_kernel_impl<T>(BROADCAST_PTR_CALL, binop_##OP<T>);            \
    }                                                                           \
    __global__ void OP##_broadcast_##S##_inline(                                \
        const T* a, const T* b, T* out, BROADCAST_INLINE_ARGS) {                \
        broadcast_kernel_impl_inline<T>(a, b, out, BROADCAST_INLINE_CALL,       \
                                        binop_##OP<T>);                         \
    }                                                                           \
    __global__ void OP##_broadcast_fast_trailing_##S(                           \
        const T* __restrict__ a, const T* __restrict__ b,                       \
        T* __restrict__ out, BROADCAST_FAST_TRAILING_ARGS) {                    \
        broadcast_fast_trailing_impl<T>(a, b, out, BROADCAST_FAST_TRAILING_CALL,\
                                        binop_##OP<T>);                         \
    }

// One dtype, all seven arithmetic operations.
#define NUMR_BINARY_ROW(T, S)                                                   \
    NUMR_BINARY_OP(T, S, add) NUMR_BINARY_OP(T, S, sub)                         \
    NUMR_BINARY_OP(T, S, mul) NUMR_BINARY_OP(T, S, div)                         \
    NUMR_BINARY_OP(T, S, pow) NUMR_BINARY_OP(T, S, max)                         \
    NUMR_BINARY_OP(T, S, min)

#endif // NUMR_BINARY_OPS_CUH
