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
            out[idx] = NumrSatF64<T>::apply(start + step * (double)idx);        \
        }                                                                       \
    }                                                                           \
    __global__ void linspace_##S(T* out, double start, double stop, unsigned int steps) { \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < steps) {                                                      \
            /* Multiply before dividing, exactly as linspace_kernel in       */ \
            /* runtime/cpu/kernels/memory.rs does. Forming the fraction first */ \
            /* rounds it, and an integer store turns that last ulp into a    */ \
            /* whole unit: 300 * (1/3) truncates to 99, 300 * 1 / 3 to 100.  */ \
            double v = start + (stop - start) * (double)idx / (double)(steps - 1); \
            out[idx] = NumrSatF64<T>::apply(v);                                 \
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
// The float rows keep their own scalar types: F32/F16/BF16/FP8 take f32
// parameters and F64 takes double, and those types are part of the launch ABI.

__global__ void arange_f32(float* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (float)idx;
    }
}

__global__ void arange_f64(double* out, double start, double step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = start + step * (double)idx;
    }
}

__global__ void arange_f16(__half* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(start + step * (float)idx);
    }
}

__global__ void arange_bf16(__nv_bfloat16* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(start + step * (float)idx);
    }
}

__global__ void arange_fp8_e4m3(numr_fp8_e4m3* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].data = f32_to_fp8_e4m3(start + step * (float)idx);
    }
}

__global__ void arange_fp8_e5m2(numr_fp8_e5m2* out, float start, float step, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].data = f32_to_fp8_e5m2(start + step * (float)idx);
    }
}

// Linspace: evenly spaced values from start to stop, both inclusive. The
// launcher handles steps < 2 itself, so `steps - 1` is never zero here.

__global__ void linspace_f32(float* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = start + (stop - start) * t;
    }
}

__global__ void linspace_f64(double* out, double start, double stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        double t = (double)idx / (double)(steps - 1);
        out[idx] = start + (stop - start) * t;
    }
}

__global__ void linspace_f16(__half* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = __float2half(start + (stop - start) * t);
    }
}

__global__ void linspace_bf16(__nv_bfloat16* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx] = __float2bfloat16(start + (stop - start) * t);
    }
}

__global__ void linspace_fp8_e4m3(numr_fp8_e4m3* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx].data = f32_to_fp8_e4m3(start + (stop - start) * t);
    }
}

__global__ void linspace_fp8_e5m2(numr_fp8_e5m2* out, float start, float stop, unsigned int steps) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < steps) {
        float t = (float)idx / (float)(steps - 1);
        out[idx].data = f32_to_fp8_e5m2(start + (stop - start) * t);
    }
}

// Eye: identity matrix, ones on the diagonal.

__global__ void eye_f32(float* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx] = (idx / m == idx % m) ? 1.0f : 0.0f;
    }
}

__global__ void eye_f64(double* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx] = (idx / m == idx % m) ? 1.0 : 0.0;
    }
}

__global__ void eye_f16(__half* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx] = __float2half((idx / m == idx % m) ? 1.0f : 0.0f);
    }
}

__global__ void eye_bf16(__nv_bfloat16* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx] = __float2bfloat16((idx / m == idx % m) ? 1.0f : 0.0f);
    }
}

__global__ void eye_fp8_e4m3(numr_fp8_e4m3* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx].data = f32_to_fp8_e4m3((idx / m == idx % m) ? 1.0f : 0.0f);
    }
}

__global__ void eye_fp8_e5m2(numr_fp8_e5m2* out, unsigned int n, unsigned int m) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        out[idx].data = f32_to_fp8_e5m2((idx / m == idx % m) ? 1.0f : 0.0f);
    }
}

} // extern "C"
