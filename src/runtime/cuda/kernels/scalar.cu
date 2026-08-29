// Scalar CUDA kernels (tensor-scalar operations)
//
// Operations: add_scalar, sub_scalar, rsub_scalar, mul_scalar, div_scalar,
// pow_scalar — every dtype below gets all six.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8,
//         c64, c128
//
// Kernel naming, matching the names `kernel_name(op, dtype)` builds in
// src/runtime/cuda/kernels/loader.rs from dtype_suffix():
//   {op}_scalar_{suffix}
//
// The scalar's wire type per row, matching what the Rust launchers in
// scalar.rs push:
//   f32/f64/c64/c128   the row's own float width
//   f16/bf16/fp8       float (no host-side counterpart to push)
//   integers           the element type, except pow_scalar, which takes double
//                      so a fractional exponent arrives unrounded
//
// The operation bodies, the kernel-body template and the row macros live in
// scalar_ops.cuh, which also documents the integer wrapping, division-by-zero
// and pow semantics. Complex stays here: complex pow is a polar-form
// computation and a real scalar touches only some of a complex value's
// components, so neither fits the row macro.

#include "scalar_ops.cuh"

extern "C" {

// ============================================================================
// Float dtypes: 6 operations per row
// ============================================================================

NUMR_SCALAR_ROW_FLOAT(float, float, f32)
NUMR_SCALAR_ROW_FLOAT(double, double, f64)
NUMR_SCALAR_ROW_FLOAT(__half, float, f16)
NUMR_SCALAR_ROW_FLOAT(__nv_bfloat16, float, bf16)

// ============================================================================
// FP8 dtypes: computed in F32 against the unrounded scalar
// ============================================================================

NUMR_SCALAR_ROW_FP8(numr_fp8_e4m3, fp8_e4m3, fp8_e4m3_to_f32, f32_to_fp8_e4m3)
NUMR_SCALAR_ROW_FP8(numr_fp8_e5m2, fp8_e5m2, fp8_e5m2_to_f32, f32_to_fp8_e5m2)

// ============================================================================
// Integer dtypes: add/sub/mul WRAP, div by zero yields 0, pow saturates
// ============================================================================

NUMR_SCALAR_ROW_INT(int64_t, i64)
NUMR_SCALAR_ROW_INT(int32_t, i32)
NUMR_SCALAR_ROW_INT(int16_t, i16)
NUMR_SCALAR_ROW_INT(int8_t, i8)
NUMR_SCALAR_ROW_INT(uint64_t, u64)
NUMR_SCALAR_ROW_INT(uint32_t, u32)
NUMR_SCALAR_ROW_INT(uint16_t, u16)
NUMR_SCALAR_ROW_INT(uint8_t, u8)

// ============================================================================
// Complex64 (float2) Scalar Operations
// Scalar is a real float that operates on complex numbers
// ============================================================================

// Add real scalar to complex: (a+bi) + s = (a+s) + bi
__global__ void add_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x + scalar, a[idx].y);
    }
}

// Subtract real scalar from complex: (a+bi) - s = (a-s) + bi
__global__ void sub_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x - scalar, a[idx].y);
    }
}

// Reverse subtract: s - (a+bi) = (s-a) + (-b)i
__global__ void rsub_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(scalar - a[idx].x, -a[idx].y);
    }
}

// Multiply complex by real scalar: s(a+bi) = sa + sbi
__global__ void mul_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x * scalar, a[idx].y * scalar);
    }
}

// Divide complex by real scalar: (a+bi)/s = a/s + (b/s)i
__global__ void div_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_float2(a[idx].x / scalar, a[idx].y / scalar);
    }
}

// Complex power with real exponent: z^p
// Edge cases:
//   - 0^p where p < 0: returns (Inf, Inf) - division by zero
//   - 0^0: returns (1, 0) - mathematical convention
//   - 0^p where p > 0: returns (0, 0)
__global__ void pow_scalar_c64(const float2* a, float scalar, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float2 z = a[idx];
        float r = complex64_abs(z);

        // Handle edge cases for zero magnitude
        if (r == 0.0f) {
            if (scalar < 0.0f) {
                // 0^(-p) = Inf (division by zero)
                out[idx] = make_float2(NUMR_INF_F, NUMR_INF_F);
            } else if (scalar == 0.0f) {
                // 0^0 = 1 (mathematical convention)
                out[idx] = make_float2(1.0f, 0.0f);
            } else {
                // 0^p = 0 for p > 0
                out[idx] = make_float2(0.0f, 0.0f);
            }
            return;
        }

        // z^p = |z|^p * e^(i * p * theta)
        float theta = complex64_angle(z);
        float r_pow = powf(r, scalar);
        float new_theta = scalar * theta;
        float sin_t, cos_t;
        sincosf(new_theta, &sin_t, &cos_t);

        out[idx] = make_float2(r_pow * cos_t, r_pow * sin_t);
    }
}

// ============================================================================
// Complex128 (double2) Scalar Operations
// ============================================================================

__global__ void add_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x + scalar, a[idx].y);
    }
}

__global__ void sub_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x - scalar, a[idx].y);
    }
}

// Reverse subtract: s - (a+bi) = (s-a) + (-b)i
__global__ void rsub_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(scalar - a[idx].x, -a[idx].y);
    }
}

__global__ void mul_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x * scalar, a[idx].y * scalar);
    }
}

__global__ void div_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_double2(a[idx].x / scalar, a[idx].y / scalar);
    }
}

// Complex128 power with real exponent: z^p
// Edge cases mirror pow_scalar_c64
__global__ void pow_scalar_c128(const double2* a, double scalar, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double2 z = a[idx];
        double r = complex128_abs(z);

        // Handle edge cases for zero magnitude
        if (r == 0.0) {
            if (scalar < 0.0) {
                // 0^(-p) = Inf (division by zero)
                out[idx] = make_double2(NUMR_INF, NUMR_INF);
            } else if (scalar == 0.0) {
                // 0^0 = 1 (mathematical convention)
                out[idx] = make_double2(1.0, 0.0);
            } else {
                // 0^p = 0 for p > 0
                out[idx] = make_double2(0.0, 0.0);
            }
            return;
        }

        // z^p = |z|^p * e^(i * p * theta)
        double theta = complex128_angle(z);
        double r_pow = pow(r, scalar);
        double new_theta = scalar * theta;
        double sin_t, cos_t;
        sincos(new_theta, &sin_t, &cos_t);

        out[idx] = make_double2(r_pow * cos_t, r_pow * sin_t);
    }
}

} // extern "C"
