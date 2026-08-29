// Shared machinery for the tensor-scalar kernels instantiated in scalar.cu:
// the scalar-to-element conversion, the pow device function, and the macros
// that expand one dtype into a full row of `extern "C"` wrappers.
//
// The five arithmetic operations reuse the `binop_*` device functions from
// binary_ops.cuh with the scalar as the right-hand operand, so a tensor-scalar
// op and the equivalent tensor-tensor op cannot drift apart. That also inherits
// every integer rule those functions document:
//
//  * Integer add/sub/mul WRAP, computed in the dtype's unsigned counterpart so
//    signed overflow is never undefined behaviour.
//  * Integer division by zero yields 0, and INT_MIN / -1 yields INT_MIN.
//
// Two things do NOT go through binop_*:
//
//  * pow. binop_pow takes both operands in the element type, which would round
//    an F16 exponent of 0.1 to 0.0999755 before raising. The exponent arrives
//    in its own wire type here and is used unrounded.
//  * Every narrow float — F16, BF16 and the two FP8 dtypes. Their arithmetic
//    decodes to F32, and the scalar stays an unrounded F32 rather than being
//    rounded into the element type first. See NUMR_SCALAR_ROW_NARROW_FLOAT and
//    NUMR_SCALAR_ROW_FP8.

#ifndef NUMR_SCALAR_OPS_CUH
#define NUMR_SCALAR_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "binary_ops.cuh"
#include "dtype_traits.cuh"
#include "ipow.cuh"

// ============================================================================
// Wire scalar -> element type
// ============================================================================
// The scalar reaches the kernel in the type the Rust launcher pushes: the
// element type itself for the F32/F64 and integer rows, and F32 for the F16,
// BF16 and FP8 rows, which have no host-side counterpart to push.
//
// Only the rows whose element type CAN hold the scalar convert it. A narrow
// float cannot, so it never reaches this function: those rows compute in F32
// against the unrounded scalar instead.

template<typename T, typename S>
__device__ __forceinline__ T numr_scalar_to_elem(S s) { return (T)s; }

// ============================================================================
// pow
// ============================================================================
// Declared without a body so a dtype with no specialization fails to link
// rather than silently picking a wrong path.

template<typename T, typename S> __device__ __forceinline__ T numr_scalar_pow(T a, S s);

template<> __device__ __forceinline__ float numr_scalar_pow<float, float>(float a, float s) {
    return numr_pow_safe(a, s);
}
template<> __device__ __forceinline__ double numr_scalar_pow<double, double>(double a, double s) {
    return numr_pow_safe(a, s);
}
// F16 and BF16 raise in F32: the half-width intrinsics have no pow, and the
// exponent must not be rounded to the element type first.
template<> __device__ __forceinline__ __half numr_scalar_pow<__half, float>(__half a, float s) {
    return __float2half(numr_pow_safe(__half2float(a), s));
}
template<> __device__ __forceinline__ __nv_bfloat16
numr_scalar_pow<__nv_bfloat16, float>(__nv_bfloat16 a, float s) {
    return __float2bfloat16(numr_pow_safe(__bfloat162float(a), s));
}

// Integer pow uses the exact, saturating routine from ipow.cuh. Overflow
// saturates there, which is deliberate: pow's result is an accumulator, while
// the other integer ops wrap.
#define NUMR_SCALAR_POW_INT(T)                                                  \
    template<> __device__ __forceinline__ T numr_scalar_pow<T, double>(T a, double s) { \
        return numr_ipow_scalar<T>(a, s);                                       \
    }

NUMR_SCALAR_POW_INT(int8_t)
NUMR_SCALAR_POW_INT(int16_t)
NUMR_SCALAR_POW_INT(int32_t)
NUMR_SCALAR_POW_INT(int64_t)
NUMR_SCALAR_POW_INT(uint8_t)
NUMR_SCALAR_POW_INT(uint16_t)
NUMR_SCALAR_POW_INT(uint32_t)
NUMR_SCALAR_POW_INT(uint64_t)

#undef NUMR_SCALAR_POW_INT

// ============================================================================
// Instantiation macros
// ============================================================================
// The kernel names are the ones `kernel_name(op, dtype)` builds in
// src/runtime/cuda/kernels/loader.rs: `{op}_scalar_{suffix}`.

// One arithmetic operation, forwarding to the matching binop_* with the scalar
// converted to the element type.
#define NUMR_SCALAR_OP(T, S, SUF, OP)                                           \
    __global__ void OP##_scalar_##SUF(const T* a, S scalar, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = binop_##OP<T>(a[idx], numr_scalar_to_elem<T, S>(scalar));\
        }                                                                       \
    }

// `scalar - a`, the one operation whose scalar is the LEFT operand.
#define NUMR_RSUB_SCALAR_OP(T, S, SUF)                                          \
    __global__ void rsub_scalar_##SUF(const T* a, S scalar, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = binop_sub<T>(numr_scalar_to_elem<T, S>(scalar), a[idx]); \
        }                                                                       \
    }

// pow with the exponent in wire type P, which differs from S on the integer
// rows: an integer exponent arrives as a double so a fractional value reaches
// the kernel unrounded.
#define NUMR_POW_SCALAR_OP(T, P, SUF)                                           \
    __global__ void pow_scalar_##SUF(const T* a, P scalar, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = numr_scalar_pow<T, P>(a[idx], scalar);                   \
        }                                                                       \
    }

// One dtype wide enough to hold its own scalar, with the scalar and the
// exponent sharing the wire type S: F32 and F64.
#define NUMR_SCALAR_ROW_FLOAT(T, S, SUF)                                        \
    NUMR_SCALAR_OP(T, S, SUF, add) NUMR_SCALAR_OP(T, S, SUF, sub)               \
    NUMR_SCALAR_OP(T, S, SUF, mul) NUMR_SCALAR_OP(T, S, SUF, div)               \
    NUMR_RSUB_SCALAR_OP(T, S, SUF) NUMR_POW_SCALAR_OP(T, S, SUF)

// One narrow float that CUDA gives an arithmetic type for: F16 and BF16.
//
// The row cannot forward to binop_* the way the F32/F64 row does, because that
// would take the scalar through numr_scalar_to_elem first. F16 cannot hold 0.3;
// rounding it to 0.30004883 and only then adding rounds the answer TWICE, and
// the second rounding starts from a value up to half an ulp away from the one
// the caller asked for, so the result can land an ulp off. Decoding to F32,
// computing against the UNROUNDED scalar and encoding once is the single
// narrowing at write-out that src/runtime/cpu/kernels/wide_acc.rs states for
// every is_narrow_float() dtype, and what the CPU scalar kernels and
// NUMR_SCALAR_ROW_FP8 already do.
//
// pow is already correct: numr_scalar_pow's F16 and BF16 specializations raise
// in F32 against the unrounded exponent.
#define NUMR_SCALAR_ROW_NARROW_FLOAT(T, SUF, TO_F32, FROM_F32)                  \
    __global__ void add_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = FROM_F32(TO_F32(a[idx]) + s);                   \
    }                                                                           \
    __global__ void sub_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = FROM_F32(TO_F32(a[idx]) - s);                   \
    }                                                                           \
    __global__ void mul_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = FROM_F32(TO_F32(a[idx]) * s);                   \
    }                                                                           \
    __global__ void div_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = FROM_F32(TO_F32(a[idx]) / s);                   \
    }                                                                           \
    __global__ void rsub_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = FROM_F32(s - TO_F32(a[idx]));                   \
    }                                                                           \
    NUMR_POW_SCALAR_OP(T, float, SUF)

// One integer dtype. The five arithmetic kernels take the scalar in the element
// type, already saturated to that range by the host's `as` cast (which is what
// `Element::from_f64` does on CPU). pow takes a double: see NUMR_POW_SCALAR_OP.
#define NUMR_SCALAR_ROW_INT(T, SUF)                                             \
    NUMR_SCALAR_OP(T, T, SUF, add) NUMR_SCALAR_OP(T, T, SUF, sub)               \
    NUMR_SCALAR_OP(T, T, SUF, mul) NUMR_SCALAR_OP(T, T, SUF, div)               \
    NUMR_RSUB_SCALAR_OP(T, T, SUF) NUMR_POW_SCALAR_OP(T, double, SUF)

// One FP8 dtype. FP8 has no arithmetic of its own: every operation decodes to
// F32, computes against the UNROUNDED F32 scalar, and re-encodes. Encoding the
// scalar to FP8 first would answer a different question — `x + 0.1` with 0.1
// rounded to the nearest FP8 value — so the row stays outside binop_*.
#define NUMR_SCALAR_ROW_FP8(T, SUF, TO_F32, FROM_F32)                           \
    __global__ void add_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(TO_F32(a[idx].data) + s));           \
    }                                                                           \
    __global__ void sub_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(TO_F32(a[idx].data) - s));           \
    }                                                                           \
    __global__ void mul_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(TO_F32(a[idx].data) * s));           \
    }                                                                           \
    __global__ void div_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(TO_F32(a[idx].data) / s));           \
    }                                                                           \
    __global__ void rsub_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(s - TO_F32(a[idx].data)));           \
    }                                                                           \
    __global__ void pow_scalar_##SUF(const T* a, float s, T* out, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) out[idx] = T(FROM_F32(numr_pow_safe(TO_F32(a[idx].data), s))); \
    }

#endif // NUMR_SCALAR_OPS_CUH
