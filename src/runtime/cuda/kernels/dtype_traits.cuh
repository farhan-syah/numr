// numr DType Traits - Multi-precision CUDA kernel support
// Provides FP8 types with Hopper PTX intrinsics (SM 8.9+) and software fallback (SM 8.0+)
//
// Supports: F32, F64, F16, BF16, FP8E4M3, FP8E5M2, I32, I64
// Note: Using numr_ prefix to avoid collision with NVIDIA's __nv_fp8 types

#ifndef NUMR_DTYPE_TRAITS_CUH
#define NUMR_DTYPE_TRAITS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Mathematical Constants
// ============================================================================
// Following C99/C++11 standards for mathematical constants
// These match std::f32::consts::PI and std::f64::consts::PI exactly

#ifndef M_PI
#define M_PI 3.14159265358979323846  // π to F64 precision (20 digits, ~17 significant)
#endif

#ifndef M_PI_F
#define M_PI_F 3.14159265f  // π to F32 precision (~7 significant digits)
#endif

// ============================================================================
// NaN and Infinity Constants
// ============================================================================
// Using CUDA intrinsics to generate NaN and Inf values

#ifndef NUMR_NAN_F
#define NUMR_NAN_F __int_as_float(0x7FC00000)  // Quiet NaN (single precision)
#endif

#ifndef NUMR_INF_F
#define NUMR_INF_F __int_as_float(0x7F800000)  // Positive infinity (single precision)
#endif

#ifndef NUMR_NAN
#define NUMR_NAN __longlong_as_double(0x7FF8000000000000LL)  // Quiet NaN (double precision)
#endif

#ifndef NUMR_INF
#define NUMR_INF __longlong_as_double(0x7FF0000000000000LL)  // Positive infinity (double precision)
#endif

// ============================================================================
// Sign-Correct Power (fast-math safe)
// ============================================================================
// numr compiles all CUDA kernels with --use_fast_math (see build.rs). Under
// fast-math, powf()/pow() lower to exp2(y * log2(x)), which is NaN for ANY
// negative base x - even when y is an integer, where IEEE/CPU semantics
// define a real result (e.g. (-3)^2 == 9, (-3)^3 == -27). Rust's f64::powf
// (used by the CPU backend) is IEEE-correct, so CUDA must match it:
//   base >= 0                       -> powf/pow(base, exp)   (fast path)
//   base <  0, exp integral         -> powf/pow(|base|, exp), negated iff exp is odd
//   base <  0, exp not integral     -> NaN (correct IEEE behavior, unchanged)
// Route every pow kernel (scalar and elementwise, all dtypes) through these
// helpers instead of calling powf/pow directly on a possibly-negative base.

__device__ __forceinline__ float numr_pow_safe(float base, float exp) {
    if (base >= 0.0f) {
        return powf(base, exp);
    }
    float rounded = roundf(exp);
    if (rounded != exp) {
        // Non-integer exponent of a negative base is undefined -> NaN.
        return NUMR_NAN_F;
    }
    float mag = powf(-base, exp);
    bool is_odd = fmodf(rounded, 2.0f) != 0.0f;
    return is_odd ? -mag : mag;
}

__device__ __forceinline__ double numr_pow_safe(double base, double exp) {
    if (base >= 0.0) {
        return pow(base, exp);
    }
    double rounded = round(exp);
    if (rounded != exp) {
        return NUMR_NAN;
    }
    double mag = pow(-base, exp);
    bool is_odd = fmod(rounded, 2.0) != 0.0;
    return is_odd ? -mag : mag;
}

// ============================================================================
// Complex Number Type Definitions
// ============================================================================
// Complex64 maps to float2 (8 bytes: 2x f32)
// Complex128 maps to double2 (16 bytes: 2x f64)
// These match the Rust Complex64/Complex128 #[repr(C)] layout

// Complex64: 64-bit complex (2x f32)
// Uses CUDA's built-in float2 type
typedef float2 numr_complex64;

// Complex128: 128-bit complex (2x f64)
// Uses CUDA's built-in double2 type
typedef double2 numr_complex128;

// ============================================================================
// Complex Arithmetic Device Functions
// ============================================================================

// --- Complex64 (float2) arithmetic ---

__device__ __forceinline__ numr_complex64 complex64_add(numr_complex64 a, numr_complex64 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ numr_complex64 complex64_sub(numr_complex64 a, numr_complex64 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
__device__ __forceinline__ numr_complex64 complex64_mul(numr_complex64 a, numr_complex64 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,  // real: ac - bd
        a.x * b.y + a.y * b.x   // imag: ad + bc
    );
}

// (a+bi)/(c+di) = (a+bi)*conj(c+di)/|c+di|²
__device__ __forceinline__ numr_complex64 complex64_div(numr_complex64 a, numr_complex64 b) {
    float denom = b.x * b.x + b.y * b.y;  // |b|²
    if (denom == 0.0f) {
        return make_float2(NUMR_NAN_F, NUMR_NAN_F);
    }
    return make_float2(
        (a.x * b.x + a.y * b.y) / denom,  // real: (ac + bd) / |b|²
        (a.y * b.x - a.x * b.y) / denom   // imag: (bc - ad) / |b|²
    );
}

__device__ __forceinline__ numr_complex64 complex64_neg(numr_complex64 a) {
    return make_float2(-a.x, -a.y);
}

__device__ __forceinline__ numr_complex64 complex64_conj(numr_complex64 a) {
    return make_float2(a.x, -a.y);
}

__device__ __forceinline__ float complex64_abs(numr_complex64 a) {
    return sqrtf(a.x * a.x + a.y * a.y);
}

__device__ __forceinline__ float complex64_abs_squared(numr_complex64 a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ float complex64_angle(numr_complex64 a) {
    return atan2f(a.y, a.x);
}

// Reciprocal: 1/z = conj(z)/|z|²
__device__ __forceinline__ numr_complex64 complex64_recip(numr_complex64 a) {
    float mag_sq = a.x * a.x + a.y * a.y;
    if (mag_sq == 0.0f) {
        return make_float2(NUMR_INF_F, NUMR_INF_F);
    }
    return make_float2(a.x / mag_sq, -a.y / mag_sq);
}

// Square: z² = (a+bi)² = a² - b² + 2abi
__device__ __forceinline__ numr_complex64 complex64_square(numr_complex64 a) {
    return make_float2(
        a.x * a.x - a.y * a.y,
        2.0f * a.x * a.y
    );
}

// Exponential: e^z = e^a * (cos(b) + i*sin(b))
__device__ __forceinline__ numr_complex64 complex64_exp(numr_complex64 a) {
    float exp_re = expf(a.x);
    float sin_im, cos_im;
    sincosf(a.y, &sin_im, &cos_im);
    return make_float2(exp_re * cos_im, exp_re * sin_im);
}

// Natural log: ln(z) = ln(|z|) + i*arg(z)
__device__ __forceinline__ numr_complex64 complex64_log(numr_complex64 a) {
    return make_float2(logf(complex64_abs(a)), complex64_angle(a));
}

// Square root (principal branch)
__device__ __forceinline__ numr_complex64 complex64_sqrt(numr_complex64 a) {
    float mag = complex64_abs(a);
    if (mag == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }
    float re = sqrtf((mag + a.x) / 2.0f);
    float im = copysignf(sqrtf((mag - a.x) / 2.0f), a.y);
    return make_float2(re, im);
}

// --- Complex128 (double2) arithmetic ---

__device__ __forceinline__ numr_complex128 complex128_add(numr_complex128 a, numr_complex128 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ numr_complex128 complex128_sub(numr_complex128 a, numr_complex128 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ numr_complex128 complex128_mul(numr_complex128 a, numr_complex128 b) {
    return make_double2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__ numr_complex128 complex128_div(numr_complex128 a, numr_complex128 b) {
    double denom = b.x * b.x + b.y * b.y;
    if (denom == 0.0) {
        return make_double2(NUMR_NAN, NUMR_NAN);
    }
    return make_double2(
        (a.x * b.x + a.y * b.y) / denom,
        (a.y * b.x - a.x * b.y) / denom
    );
}

__device__ __forceinline__ numr_complex128 complex128_neg(numr_complex128 a) {
    return make_double2(-a.x, -a.y);
}

__device__ __forceinline__ numr_complex128 complex128_conj(numr_complex128 a) {
    return make_double2(a.x, -a.y);
}

__device__ __forceinline__ double complex128_abs(numr_complex128 a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

__device__ __forceinline__ double complex128_abs_squared(numr_complex128 a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ double complex128_angle(numr_complex128 a) {
    return atan2(a.y, a.x);
}

__device__ __forceinline__ numr_complex128 complex128_recip(numr_complex128 a) {
    double mag_sq = a.x * a.x + a.y * a.y;
    if (mag_sq == 0.0) {
        return make_double2(NUMR_INF, NUMR_INF);
    }
    return make_double2(a.x / mag_sq, -a.y / mag_sq);
}

__device__ __forceinline__ numr_complex128 complex128_square(numr_complex128 a) {
    return make_double2(
        a.x * a.x - a.y * a.y,
        2.0 * a.x * a.y
    );
}

__device__ __forceinline__ numr_complex128 complex128_exp(numr_complex128 a) {
    double exp_re = exp(a.x);
    double sin_im, cos_im;
    sincos(a.y, &sin_im, &cos_im);
    return make_double2(exp_re * cos_im, exp_re * sin_im);
}

__device__ __forceinline__ numr_complex128 complex128_log(numr_complex128 a) {
    return make_double2(log(complex128_abs(a)), complex128_angle(a));
}

__device__ __forceinline__ numr_complex128 complex128_sqrt(numr_complex128 a) {
    double mag = complex128_abs(a);
    if (mag == 0.0) {
        return make_double2(0.0, 0.0);
    }
    double re = sqrt((mag + a.x) / 2.0);
    double im = copysign(sqrt((mag - a.x) / 2.0), a.y);
    return make_double2(re, im);
}

// ============================================================================
// FP8 Type Definitions
// ============================================================================

// numr_fp8_e4m3: 1 sign + 4 exponent + 3 mantissa bits
// Range: ~[-448, 448], better precision
struct numr_fp8_e4m3 {
    uint8_t data;
    __device__ __forceinline__ numr_fp8_e4m3() : data(0) {}
    __device__ __forceinline__ explicit numr_fp8_e4m3(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

// numr_fp8_e5m2: 1 sign + 5 exponent + 2 mantissa bits
// Range: ~[-57344, 57344], larger dynamic range
struct numr_fp8_e5m2 {
    uint8_t data;
    __device__ __forceinline__ numr_fp8_e5m2() : data(0) {}
    __device__ __forceinline__ explicit numr_fp8_e5m2(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

// ============================================================================
// FP8 Conversion Utilities
// - SM 8.9+ (Hopper): Hardware PTX intrinsics for ~2x speedup
// - SM 8.0+ (Ampere): Software emulation (still functional)
// ============================================================================

#if __CUDA_ARCH__ >= 800  // Ampere and newer support FP8

// ============================================================================
// FP8 E4M3 <-> F32 Conversion
// ============================================================================
//
// SATURATION BEHAVIOR: FP8 types have limited dynamic range. Values outside
// the representable range are silently saturated to the maximum representable
// value. This is IEEE-compliant behavior for narrowing conversions.
//
// E4M3 saturation: |x| > 448.0 → saturates to ±448.0
// E5M2 saturation: |x| > 57344.0 → saturates to ±57344.0
//
// For applications requiring overflow detection, check input ranges before
// conversion or use F16/F32 types instead.

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t u) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction
    // cvt.rn.f32.e4m3x2 unpacks 2xFP8 -> 2xF32 (we use only the first result)
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"
        "cvt.rn.f32.e4m3x2 {%0, dummy}, %1; \n\t"
        "}"
        : "=f"(result)
        : "r"((uint16_t)u)
    );
    return result;
#else
    // Ampere: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 3) & 0xF;
    int mant_bits = u & 0x7;

    if (exp_bits == 0 && mant_bits == 0) {
        return sign ? -0.0f : 0.0f;
    }

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * (1.0f / 8.0f);
        int exp = exp_bits - 7;
        result = ldexpf(mant, exp);
    } else {
        // Subnormal: implicit leading bit is 0
        float mant = mant_bits * (1.0f / 8.0f);
        result = ldexpf(mant, -6);
    }

    return sign ? -result : result;
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3(float x) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction with saturation
    uint32_t result;
    asm ("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));
    return (uint8_t)result;
#else
    // Ampere: Software emulation with saturation
    if (x == 0.0f) return 0;
    if (isnan(x)) return 0x7F;  // NaN -> max positive

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    // Saturate to E4M3 range
    if (abs_x > 448.0f) abs_x = 448.0f;

    int exp_bits;
    float mant = frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-7, min(8, exp_bits));
    int exp_encoded = exp_bits + 7;

    int mant_bits = (int)roundf((mant - 1.0f) * 8.0f);
    mant_bits = max(0, min(7, mant_bits));

    return (sign << 7) | (exp_encoded << 3) | mant_bits;
#endif
}

// ============================================================================
// FP8 E5M2 <-> F32 Conversion
// ============================================================================

__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t u) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"
        "cvt.rn.f32.e5m2x2 {%0, dummy}, %1; \n\t"
        "}"
        : "=f"(result)
        : "r"((uint16_t)u)
    );
    return result;
#else
    // Ampere: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 2) & 0x1F;
    int mant_bits = u & 0x3;

    if (exp_bits == 0 && mant_bits == 0) {
        return sign ? -0.0f : 0.0f;
    }

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * 0.25f;
        int exp = exp_bits - 15;
        result = ldexpf(mant, exp);
    } else {
        // Subnormal
        float mant = mant_bits * 0.25f;
        result = ldexpf(mant, -14);
    }

    return sign ? -result : result;
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8_e5m2(float x) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction with saturation
    uint32_t result;
    asm ("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));
    return (uint8_t)result;
#else
    // Ampere: Software emulation with saturation
    if (x == 0.0f) return 0;
    if (isnan(x)) return 0x7F;  // NaN -> max positive

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    // Saturate to E5M2 range
    if (abs_x > 57344.0f) abs_x = 57344.0f;

    int exp_bits;
    float mant = frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-15, min(16, exp_bits));
    int exp_encoded = exp_bits + 15;

    int mant_bits = (int)roundf((mant - 1.0f) * 4.0f);
    mant_bits = max(0, min(3, mant_bits));

    return (sign << 7) | (exp_encoded << 2) | mant_bits;
#endif
}

#else  // __CUDA_ARCH__ < 800

// Pre-Ampere fallback (very slow, should rarely be used)
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t u) {
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 3) & 0xF;
    int mant_bits = u & 0x7;
    if (exp_bits == 0) return 0.0f;
    float mant = 1.0f + mant_bits * (1.0f / 8.0f);
    int exp = exp_bits - 7;
    float result = ldexpf(mant, exp);
    return sign ? -result : result;
}

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3(float x) {
    if (x == 0.0f) return 0;
    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);
    if (abs_x > 448.0f) abs_x = 448.0f;
    int exp_bits;
    frexpf(abs_x, &exp_bits);
    exp_bits = max(0, min(15, exp_bits - 1 + 7));
    float mant = ldexpf(abs_x, -(exp_bits - 7));
    int mant_bits = max(0, min(7, (int)roundf((mant - 1.0f) * 8.0f)));
    return (sign << 7) | (exp_bits << 3) | mant_bits;
}

__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t u) {
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 2) & 0x1F;
    int mant_bits = u & 0x3;
    if (exp_bits == 0) return 0.0f;
    float mant = 1.0f + mant_bits * 0.25f;
    int exp = exp_bits - 15;
    float result = ldexpf(mant, exp);
    return sign ? -result : result;
}

__device__ __forceinline__ uint8_t f32_to_fp8_e5m2(float x) {
    if (x == 0.0f) return 0;
    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);
    if (abs_x > 57344.0f) abs_x = 57344.0f;
    int exp_bits;
    frexpf(abs_x, &exp_bits);
    exp_bits = max(0, min(31, exp_bits - 1 + 15));
    float mant = ldexpf(abs_x, -(exp_bits - 15));
    int mant_bits = max(0, min(3, (int)roundf((mant - 1.0f) * 4.0f)));
    return (sign << 7) | (exp_bits << 2) | mant_bits;
}

#endif  // __CUDA_ARCH__

// ============================================================================
// Helper Macros for FP8 Kernel Generation
// ============================================================================

// Load FP8 value and convert to F32 for computation
#define LOAD_FP8_E4M3(ptr, idx) fp8_e4m3_to_f32(((const uint8_t*)(ptr))[idx])
#define LOAD_FP8_E5M2(ptr, idx) fp8_e5m2_to_f32(((const uint8_t*)(ptr))[idx])

// Convert F32 result and store as FP8
#define STORE_FP8_E4M3(ptr, idx, val) ((uint8_t*)(ptr))[idx] = f32_to_fp8_e4m3(val)
#define STORE_FP8_E5M2(ptr, idx, val) ((uint8_t*)(ptr))[idx] = f32_to_fp8_e5m2(val)

// ============================================================================
// DType Constants for FP8
// ============================================================================

// E4M3 range constants
#define FP8_E4M3_MAX 448.0f
#define FP8_E4M3_MIN (-448.0f)
#define FP8_E4M3_EPS 0.125f

// E5M2 range constants
#define FP8_E5M2_MAX 57344.0f
#define FP8_E5M2_MIN (-57344.0f)
#define FP8_E5M2_EPS 0.25f

// ============================================================================
// Accumulator Traits for Templated Reduction Kernels
// ============================================================================
// Similar pattern to CPU's Accumulator trait - provides:
// - load: T -> Acc conversion
// - store: Acc -> T conversion
// - identity values (zero, neg_inf, pos_inf)
// - warp reduction primitives

// Warp reduction primitives for f32
__device__ __forceinline__ float warp_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ float warp_max_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ float warp_min_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp reduction primitives for f64
__device__ __forceinline__ double warp_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ double warp_max_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ double warp_min_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp product reduction primitives
__device__ __forceinline__ float warp_prod_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ double warp_prod_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp boolean reduction primitives (for any/all)
__device__ __forceinline__ int warp_or(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val |= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ int warp_and(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val &= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Primary template (not defined - specializations required)
template<typename T, typename Acc> struct AccumTraits;

// ============================================================================
// Simple dtype_traits for Linalg Kernels
// ============================================================================
// Simpler single-parameter template for basic linalg operations.
// Provides: zero, one, abs, max, atomic_add, atomic_max

template<typename T> struct dtype_traits;

template<> struct dtype_traits<float> {
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float two() { return 2.0f; }
    static __device__ __forceinline__ float eps() { return 1.192092896e-07f; }  // FLT_EPSILON
    static __device__ __forceinline__ float small_eps() { return 1e-10f; }
    static __device__ __forceinline__ float tiny_eps() { return 1e-30f; }
    static __device__ __forceinline__ float abs(float x) { return fabsf(x); }
    static __device__ __forceinline__ float sqrt(float x) { return sqrtf(x); }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
    static __device__ __forceinline__ void atomic_add(float* addr, float val) { atomicAdd(addr, val); }
    static __device__ __forceinline__ void atomic_max(float* addr, float val) {
        unsigned int* addr_as_ui = (unsigned int*)addr;
        unsigned int old = *addr_as_ui, assumed;
        do {
            assumed = old;
            if (__uint_as_float(assumed) >= val) break;
            old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
        } while (assumed != old);
    }
};

template<> struct dtype_traits<double> {
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double two() { return 2.0; }
    static __device__ __forceinline__ double eps() { return 2.220446049250313e-16; }  // DBL_EPSILON
    static __device__ __forceinline__ double small_eps() { return 1e-15; }
    static __device__ __forceinline__ double tiny_eps() { return 1e-300; }
    static __device__ __forceinline__ double abs(double x) { return fabs(x); }
    static __device__ __forceinline__ double sqrt(double x) { return ::sqrt(x); }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
    static __device__ __forceinline__ void atomic_add(double* addr, double val) { atomicAdd(addr, val); }
    static __device__ __forceinline__ void atomic_max(double* addr, double val) {
        unsigned long long* addr_as_ull = (unsigned long long*)addr;
        unsigned long long old = *addr_as_ull, assumed;
        do {
            assumed = old;
            if (__longlong_as_double(assumed) >= val) break;
            old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
        } while (assumed != old);
    }
};

// F16 (__half) dtype traits - all operations use F32 internally for precision
template<> struct dtype_traits<__half> {
    static __device__ __forceinline__ __half zero() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half one() { return __float2half(1.0f); }
    static __device__ __forceinline__ __half two() { return __float2half(2.0f); }
    static __device__ __forceinline__ __half eps() { return __float2half(9.77e-04f); }  // FP16 machine epsilon
    static __device__ __forceinline__ __half small_eps() { return __float2half(1e-5f); }
    static __device__ __forceinline__ __half tiny_eps() { return __float2half(6e-8f); }  // Near FP16 min subnormal
    static __device__ __forceinline__ __half abs(__half x) { return __float2half(fabsf(__half2float(x))); }
    static __device__ __forceinline__ __half sqrt(__half x) { return __float2half(sqrtf(__half2float(x))); }
    static __device__ __forceinline__ __half max(__half a, __half b) { return __hmax(a, b); }
    static __device__ __forceinline__ __half min(__half a, __half b) { return __hmin(a, b); }
    // Atomic operations use F32 accumulation for precision
    static __device__ __forceinline__ void atomic_add(__half* addr, __half val) {
        // No native atomicAdd for __half on most architectures, use CAS loop
        unsigned short* addr_as_us = (unsigned short*)addr;
        unsigned short old = *addr_as_us, assumed;
        do {
            assumed = old;
            __half new_val = __hadd(*(__half*)&assumed, val);
            old = atomicCAS(addr_as_us, assumed, *(unsigned short*)&new_val);
        } while (assumed != old);
    }
    static __device__ __forceinline__ void atomic_max(__half* addr, __half val) {
        unsigned short* addr_as_us = (unsigned short*)addr;
        unsigned short old = *addr_as_us, assumed;
        do {
            assumed = old;
            if (__half2float(*(__half*)&assumed) >= __half2float(val)) break;
            old = atomicCAS(addr_as_us, assumed, *(unsigned short*)&val);
        } while (assumed != old);
    }
};

// BF16 (__nv_bfloat16) dtype traits - all operations use F32 internally
template<> struct dtype_traits<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 two() { return __float2bfloat16(2.0f); }
    static __device__ __forceinline__ __nv_bfloat16 eps() { return __float2bfloat16(7.81e-03f); }  // BF16 machine epsilon
    static __device__ __forceinline__ __nv_bfloat16 small_eps() { return __float2bfloat16(1e-4f); }
    static __device__ __forceinline__ __nv_bfloat16 tiny_eps() { return __float2bfloat16(1e-38f); }  // Near BF16 min normal
    static __device__ __forceinline__ __nv_bfloat16 abs(__nv_bfloat16 x) { return __float2bfloat16(fabsf(__bfloat162float(x))); }
    static __device__ __forceinline__ __nv_bfloat16 sqrt(__nv_bfloat16 x) { return __float2bfloat16(sqrtf(__bfloat162float(x))); }
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
    // Atomic operations use CAS loop
    static __device__ __forceinline__ void atomic_add(__nv_bfloat16* addr, __nv_bfloat16 val) {
        unsigned short* addr_as_us = (unsigned short*)addr;
        unsigned short old = *addr_as_us, assumed;
        do {
            assumed = old;
            __nv_bfloat16 new_val = __hadd(*(__nv_bfloat16*)&assumed, val);
            old = atomicCAS(addr_as_us, assumed, *(unsigned short*)&new_val);
        } while (assumed != old);
    }
    static __device__ __forceinline__ void atomic_max(__nv_bfloat16* addr, __nv_bfloat16 val) {
        unsigned short* addr_as_us = (unsigned short*)addr;
        unsigned short old = *addr_as_us, assumed;
        do {
            assumed = old;
            if (__bfloat162float(*(__nv_bfloat16*)&assumed) >= __bfloat162float(val)) break;
            old = atomicCAS(addr_as_us, assumed, *(unsigned short*)&val);
        } while (assumed != old);
    }
};

// ============================================================================
// AccumTraits Macros - Reduce boilerplate for accumulator specializations
// ============================================================================

// Macro for accumulator types using F32 operations
#define DEFINE_ACCUM_TRAITS_F32(SrcType, LOAD_EXPR, STORE_EXPR, NEG_INF_VAL, POS_INF_VAL) \
template<> struct AccumTraits<SrcType, float> { \
    static __device__ __forceinline__ float load(const SrcType* p, int i) { return LOAD_EXPR; } \
    static __device__ __forceinline__ void store(SrcType* p, int i, float v) { STORE_EXPR; } \
    static __device__ __forceinline__ float zero() { return 0.0f; } \
    static __device__ __forceinline__ float one() { return 1.0f; } \
    static __device__ __forceinline__ float neg_inf() { return NEG_INF_VAL; } \
    static __device__ __forceinline__ float pos_inf() { return POS_INF_VAL; } \
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); } \
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); } \
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); } \
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); } \
    static __device__ __forceinline__ float add(float a, float b) { return a + b; } \
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; } \
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); } \
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); } \
};

// Macro for accumulator types using F64 operations
#define DEFINE_ACCUM_TRAITS_F64(SrcType, LOAD_EXPR, STORE_EXPR, NEG_INF_VAL, POS_INF_VAL) \
template<> struct AccumTraits<SrcType, double> { \
    static __device__ __forceinline__ double load(const SrcType* p, int i) { return LOAD_EXPR; } \
    static __device__ __forceinline__ void store(SrcType* p, int i, double v) { STORE_EXPR; } \
    static __device__ __forceinline__ double zero() { return 0.0; } \
    static __device__ __forceinline__ double one() { return 1.0; } \
    static __device__ __forceinline__ double neg_inf() { return NEG_INF_VAL; } \
    static __device__ __forceinline__ double pos_inf() { return POS_INF_VAL; } \
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); } \
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); } \
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); } \
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); } \
    static __device__ __forceinline__ double add(double a, double b) { return a + b; } \
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; } \
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); } \
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); } \
};

// Macro for BF16 accumulator (uses F32 internally via conversion)
#define DEFINE_ACCUM_TRAITS_BF16(SrcType, LOAD_EXPR, STORE_EXPR, NEG_INF_VAL, POS_INF_VAL) \
template<> struct AccumTraits<SrcType, __nv_bfloat16> { \
    static __device__ __forceinline__ __nv_bfloat16 load(const SrcType* p, int i) { return LOAD_EXPR; } \
    static __device__ __forceinline__ void store(SrcType* p, int i, __nv_bfloat16 v) { STORE_EXPR; } \
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); } \
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); } \
    static __device__ __forceinline__ __nv_bfloat16 neg_inf() { return __float2bfloat16(NEG_INF_VAL); } \
    static __device__ __forceinline__ __nv_bfloat16 pos_inf() { return __float2bfloat16(POS_INF_VAL); } \
    static __device__ __forceinline__ __nv_bfloat16 warp_sum(__nv_bfloat16 v) { return __float2bfloat16(warp_sum_f32(__bfloat162float(v))); } \
    static __device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) { return __float2bfloat16(warp_max_f32(__bfloat162float(v))); } \
    static __device__ __forceinline__ __nv_bfloat16 warp_min(__nv_bfloat16 v) { return __float2bfloat16(warp_min_f32(__bfloat162float(v))); } \
    static __device__ __forceinline__ __nv_bfloat16 warp_prod(__nv_bfloat16 v) { return __float2bfloat16(warp_prod_f32(__bfloat162float(v))); } \
    static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); } \
    static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); } \
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); } \
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); } \
};

// ============================================================================
// F32 Accumulator Specializations
// ============================================================================
DEFINE_ACCUM_TRAITS_F32(float, p[i], p[i] = v, -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F32(__half, __half2float(p[i]), p[i] = __float2half(v), -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F32(__nv_bfloat16, __bfloat162float(p[i]), p[i] = __float2bfloat16(v), -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F32(numr_fp8_e4m3, fp8_e4m3_to_f32(p[i].data), p[i].data = f32_to_fp8_e4m3(v), -FP8_E4M3_MAX, FP8_E4M3_MAX)
DEFINE_ACCUM_TRAITS_F32(numr_fp8_e5m2, fp8_e5m2_to_f32(p[i].data), p[i].data = f32_to_fp8_e5m2(v), -FP8_E5M2_MAX, FP8_E5M2_MAX)

// ============================================================================
// F64 Accumulator Specializations
// ============================================================================
DEFINE_ACCUM_TRAITS_F64(float, (double)p[i], p[i] = (float)v, -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F64(double, p[i], p[i] = v, -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F64(__half, (double)__half2float(p[i]), p[i] = __float2half((float)v), -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F64(__nv_bfloat16, (double)__bfloat162float(p[i]), p[i] = __float2bfloat16((float)v), -INFINITY, INFINITY)
DEFINE_ACCUM_TRAITS_F64(numr_fp8_e4m3, (double)fp8_e4m3_to_f32(p[i].data), p[i].data = f32_to_fp8_e4m3((float)v), -FP8_E4M3_MAX, FP8_E4M3_MAX)
DEFINE_ACCUM_TRAITS_F64(numr_fp8_e5m2, (double)fp8_e5m2_to_f32(p[i].data), p[i].data = f32_to_fp8_e5m2((float)v), -FP8_E5M2_MAX, FP8_E5M2_MAX)

// ============================================================================
// BF16 Accumulator Specializations (for FP8 types)
// ============================================================================
DEFINE_ACCUM_TRAITS_BF16(numr_fp8_e4m3, __float2bfloat16(fp8_e4m3_to_f32(p[i].data)), p[i].data = f32_to_fp8_e4m3(__bfloat162float(v)), -FP8_E4M3_MAX, FP8_E4M3_MAX)
DEFINE_ACCUM_TRAITS_BF16(numr_fp8_e5m2, __float2bfloat16(fp8_e5m2_to_f32(p[i].data)), p[i].data = f32_to_fp8_e5m2(__bfloat162float(v)), -FP8_E5M2_MAX, FP8_E5M2_MAX)

// ============================================================================
// Native Half/BF16 Accumulator (non-macro for native ops)
// ============================================================================
template<> struct AccumTraits<__half, __half> {
    static __device__ __forceinline__ __half load(const __half* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(__half* p, int i, __half v) { p[i] = v; }
    static __device__ __forceinline__ __half zero() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half one() { return __float2half(1.0f); }
    static __device__ __forceinline__ __half neg_inf() { return __float2half(-INFINITY); }
    static __device__ __forceinline__ __half pos_inf() { return __float2half(INFINITY); }
    static __device__ __forceinline__ __half warp_sum(__half v) { return __float2half(warp_sum_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_max(__half v) { return __float2half(warp_max_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_min(__half v) { return __float2half(warp_min_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_prod(__half v) { return __float2half(warp_prod_f32(__half2float(v))); }
    static __device__ __forceinline__ __half add(__half a, __half b) { return __hadd(a, b); }
    static __device__ __forceinline__ __half mul(__half a, __half b) { return __hmul(a, b); }
    static __device__ __forceinline__ __half max(__half a, __half b) { return __hmax(a, b); }
    static __device__ __forceinline__ __half min(__half a, __half b) { return __hmin(a, b); }
};

template<> struct AccumTraits<__nv_bfloat16, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 load(const __nv_bfloat16* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, int i, __nv_bfloat16 v) { p[i] = v; }
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 neg_inf() { return __float2bfloat16(-INFINITY); }
    static __device__ __forceinline__ __nv_bfloat16 pos_inf() { return __float2bfloat16(INFINITY); }
    static __device__ __forceinline__ __nv_bfloat16 warp_sum(__nv_bfloat16 v) { return __float2bfloat16(warp_sum_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) { return __float2bfloat16(warp_max_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_min(__nv_bfloat16 v) { return __float2bfloat16(warp_min_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_prod(__nv_bfloat16 v) { return __float2bfloat16(warp_prod_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
};

#endif // NUMR_DTYPE_TRAITS_CUH
