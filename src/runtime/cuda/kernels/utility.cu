// Deterministic tensor-creation CUDA kernels: fill, arange, linspace, eye.
//
// Random sampling lives in utility_random.cu (PTX module "utility_random");
// this file is PTX module "utility" (kernel_names::UTILITY_MODULE).
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/utility.rs from dtype_suffix() in loader.rs.

#include <climits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "narrow_f64.cuh"

// ============================================================================
// Saturating double -> integer conversion
// ============================================================================
// The CPU kernels build every arange/linspace/eye value in f64 and store it
// through `Element::from_f64`, which is Rust's `as` cast: NaN becomes 0, a
// value below the type's minimum clamps to the minimum, a value above the
// maximum clamps to the maximum, and anything in range truncates toward zero.
// C's `(T)v` is undefined outside the range, so the bounds are tested in double
// first.
//
// `HI_D` is the smallest double at or above which the result must clamp, which
// is NOT always `(double)HI`: I64's maximum and U64's maximum are not
// representable as doubles, so those two rows use the next power of two (2^63
// and 2^64), the first double the type cannot hold.

template<typename T> struct NumrSatF64;

#define NUMR_SAT_F64(T, LO, HI, LO_D, HI_D)                                     \
    template<> struct NumrSatF64<T> {                                           \
        static __device__ __forceinline__ T apply(double v) {                   \
            if (isnan(v)) return (T)0;                                          \
            if (v <= (LO_D)) return (T)(LO);                                    \
            if (v >= (HI_D)) return (T)(HI);                                    \
            return (T)v;                                                        \
        }                                                                       \
    };

NUMR_SAT_F64(long long, LLONG_MIN, LLONG_MAX, -9223372036854775808.0, 9223372036854775808.0)
NUMR_SAT_F64(int, INT_MIN, INT_MAX, -2147483648.0, 2147483647.0)
NUMR_SAT_F64(short, SHRT_MIN, SHRT_MAX, -32768.0, 32767.0)
NUMR_SAT_F64(signed char, SCHAR_MIN, SCHAR_MAX, -128.0, 127.0)
NUMR_SAT_F64(unsigned long long, 0, ULLONG_MAX, 0.0, 18446744073709551616.0)
NUMR_SAT_F64(unsigned int, 0, UINT_MAX, 0.0, 4294967295.0)
NUMR_SAT_F64(unsigned short, 0, USHRT_MAX, 0.0, 65535.0)
NUMR_SAT_F64(unsigned char, 0, UCHAR_MAX, 0.0, 255.0)

#undef NUMR_SAT_F64

// ============================================================================
// double -> float element narrowing
// ============================================================================
// The CPU kernels build every value in f64 and store it through
// `Element::from_f64`. F32/F64 round that f64 straight to the element type in
// one step. F16, BF16 and the FP8 rows do not: each narrows the way its own CPU
// reference narrows, so a value that is a tie in f64 lands on the same element
// on both backends.
//
// F16 and BF16 defer to `narrow_f64.cuh`, which is where the rules live -
// `half`'s F16 stages through f32 on x86-64 with F16C, `half`'s BF16 runs its
// own software algorithm, and neither is a single rounding of the f64. Read
// that header before touching either row. The FP8 rows round through f32
// because their `Element::from_f64` is `from_f32(v as f32)`.

template<typename T> struct NumrNarrowF64;

template<> struct NumrNarrowF64<float> {
    static __device__ __forceinline__ float apply(double v) { return (float)v; }
};
template<> struct NumrNarrowF64<double> {
    static __device__ __forceinline__ double apply(double v) { return v; }
};
template<> struct NumrNarrowF64<__half> {
    static __device__ __forceinline__ __half apply(double v) { return numr_f64_to_f16(v); }
};
template<> struct NumrNarrowF64<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 apply(double v) { return numr_f64_to_bf16(v); }
};
template<> struct NumrNarrowF64<numr_fp8_e4m3> {
    static __device__ __forceinline__ numr_fp8_e4m3 apply(double v) {
        return numr_fp8_e4m3(f32_to_fp8_e4m3((float)v));
    }
};
template<> struct NumrNarrowF64<numr_fp8_e5m2> {
    static __device__ __forceinline__ numr_fp8_e5m2 apply(double v) {
        return numr_fp8_e5m2(f32_to_fp8_e5m2((float)v));
    }
};

// ============================================================================
// The value each index carries
// ============================================================================
// Both expressions are the CPU ones from `runtime/cpu/kernels/memory.rs`, term
// for term, and both are written with the `__d*_rn` intrinsics rather than
// operators. nvcc is compiled here with `--use_fast_math`, which implies
// `--fmad=true`: written as `start + step * idx`, the compiler contracts the
// multiply and add into one `fma.rn.f64` that rounds once where the CPU rounds
// twice, and the two disagree in the last ulp. The intrinsics are never
// contracted. `--use_fast_math` also implies `--prec-div=false`, but that
// downgrades f32 division only, so `__ddiv_rn` here is plain IEEE.

__device__ __forceinline__ double numr_arange_value(double start, double step, unsigned int idx) {
    return __dadd_rn(start, __dmul_rn(step, (double)idx));
}

// Multiply before dividing, as the CPU kernel does. Forming the fraction first
// rounds it, and an integer store turns that last ulp into a whole unit:
// 300 * (1/3) truncates to 99, 300 * 1 / 3 to 100.
__device__ __forceinline__ double numr_linspace_value(double start, double stop,
                                                      unsigned int idx, unsigned int steps) {
    double delta = __dsub_rn(stop, start);
    return __dadd_rn(start, __ddiv_rn(__dmul_rn(delta, (double)idx), (double)(steps - 1)));
}

extern "C" {

// ============================================================================
// Fill - initialise a tensor with a constant value
// ============================================================================
// One kernel per dtype, because the value arrives as a kernel argument of that
// dtype rather than as a widened scalar. Every dtype FillValue can name has a
// kernel here: a dtype missing from this list would fall back to a wider fill
// kernel and write past the end of each element.

#define NUMR_FILL(T, S)                                                         \
    __global__ void fill_##S(T* out, T value, unsigned int n) {                 \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = value;                                                   \
        }                                                                       \
    }

NUMR_FILL(float, f32)
NUMR_FILL(double, f64)
NUMR_FILL(__half, f16)
NUMR_FILL(__nv_bfloat16, bf16)
NUMR_FILL(numr_fp8_e4m3, fp8_e4m3)
NUMR_FILL(numr_fp8_e5m2, fp8_e5m2)
NUMR_FILL(long long, i64)
NUMR_FILL(int, i32)
NUMR_FILL(short, i16)
NUMR_FILL(signed char, i8)
NUMR_FILL(unsigned long long, u64)
NUMR_FILL(unsigned int, u32)
NUMR_FILL(unsigned short, u16)
NUMR_FILL(unsigned char, u8)

// ============================================================================
// Integer arange, linspace, eye
// ============================================================================
// All three take their scalars as double and narrow once at the store, which is
// exactly what the CPU kernels in `runtime/cpu/kernels/memory.rs` do. Computing
// in the element type instead would wrap on a start or step the type cannot
// hold, and would make a negative value on an unsigned dtype come out as a huge
// positive one where CPU answers 0.

#define NUMR_INT_CREATION_ROW(T, S)                                             \
    __global__ void arange_##S(T* out, double start, double step, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = NumrSatF64<T>::apply(numr_arange_value(start, step, idx)); \
        }                                                                       \
    }                                                                           \
    __global__ void linspace_##S(T* out, double start, double stop, unsigned int steps) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < steps) {                                                      \
            out[idx] = NumrSatF64<T>::apply(numr_linspace_value(start, stop, idx, steps)); \
        }                                                                       \
    }                                                                           \
    __global__ void eye_##S(T* out, unsigned int n, unsigned int m) {           \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n * m) {                                                      \
            out[idx] = (idx / m == idx % m) ? (T)1 : (T)0;                      \
        }                                                                       \
    }

NUMR_INT_CREATION_ROW(long long, i64)
NUMR_INT_CREATION_ROW(int, i32)
NUMR_INT_CREATION_ROW(short, i16)
NUMR_INT_CREATION_ROW(signed char, i8)
NUMR_INT_CREATION_ROW(unsigned long long, u64)
NUMR_INT_CREATION_ROW(unsigned int, u32)
NUMR_INT_CREATION_ROW(unsigned short, u16)
NUMR_INT_CREATION_ROW(unsigned char, u8)

// ============================================================================
// Float arange, linspace, eye
// ============================================================================
// The float rows take the same double scalars as the integer rows and narrow
// at the store, exactly where the CPU narrows. Taking them as float instead
// built every value in f32, which is a genuine precision loss for F32 output
// and a wrong starting value for every narrower dtype.

#define NUMR_FLOAT_CREATION_ROW(T, S)                                           \
    __global__ void arange_##S(T* out, double start, double step, unsigned int n) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = NumrNarrowF64<T>::apply(numr_arange_value(start, step, idx)); \
        }                                                                       \
    }                                                                           \
    __global__ void linspace_##S(T* out, double start, double stop, unsigned int steps) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < steps) {                                                      \
            out[idx] = NumrNarrowF64<T>::apply(numr_linspace_value(start, stop, idx, steps)); \
        }                                                                       \
    }                                                                           \
    __global__ void eye_##S(T* out, unsigned int n, unsigned int m) {           \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n * m) {                                                      \
            out[idx] = NumrNarrowF64<T>::apply((idx / m == idx % m) ? 1.0 : 0.0); \
        }                                                                       \
    }

// Linspace divides by `steps - 1`; the launcher handles steps < 2 itself, so
// that is never zero here.

NUMR_FLOAT_CREATION_ROW(float, f32)
NUMR_FLOAT_CREATION_ROW(double, f64)
NUMR_FLOAT_CREATION_ROW(__half, f16)
NUMR_FLOAT_CREATION_ROW(__nv_bfloat16, bf16)
NUMR_FLOAT_CREATION_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_FLOAT_CREATION_ROW(numr_fp8_e5m2, fp8_e5m2)

} // extern "C"
