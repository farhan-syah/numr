//! Unary and activation operation kernels
//!
//! Provides element-wise unary operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

pub mod activations;
mod complex;
pub mod fused_activations;
pub mod scalar;

pub use activations::{elu_kernel, gelu_kernel, leaky_relu_kernel, sigmoid_kernel, silu_kernel};
pub use fused_activations::{
    gelu_mul_kernel, relu_mul_kernel, sigmoid_mul_kernel, silu_mul_kernel,
};
pub use scalar::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};

use crate::dtype::{DType, Element};
use crate::ops::UnaryOp;

/// Execute a unary operation element-wise with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_op_kernel<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    // Complex types need component-wise operations, not magnitude-based to_f64/from_f64
    match T::DTYPE {
        DType::Complex64 => {
            complex::unary_op_complex64(op, a as *const f32, out as *mut f32, len);
            return;
        }
        DType::Complex128 => {
            complex::unary_op_complex128(op, a as *const f64, out as *mut f64, len);
            return;
        }
        _ => {}
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::unary_f32(op, a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::unary_f64(op, a as *const f64, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                unary::unary_f16(op, a as *const half::f16, out as *mut half::f16, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                unary::unary_bf16(op, a as *const half::bf16, out as *mut half::bf16, len);
                return;
            }
            _ => {}
        }
    }

    unary_op_scalar(op, a, out, len);
}

/// Scalar unary operation for all Element types
#[inline]
unsafe fn unary_op_scalar<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(-v);
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.abs());
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sqrt());
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp());
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln());
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sin());
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cos());
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tan());
            }
        }
        UnaryOp::Atan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.atan());
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tanh());
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v);
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v * v);
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.floor());
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ceil());
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round());
            }
        }
        UnaryOp::RoundTiesEven => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round_ties_even());
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out_slice[i] = T::from_f64(sign);
            }
        }
        UnaryOp::Rsqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v.sqrt());
            }
        }
        UnaryOp::Cbrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cbrt());
            }
        }
        UnaryOp::Exp2 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp2());
            }
        }
        UnaryOp::Expm1 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp_m1());
            }
        }
        UnaryOp::Log2 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.log2());
            }
        }
        UnaryOp::Log10 => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.log10());
            }
        }
        UnaryOp::Log1p => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln_1p());
            }
        }
        UnaryOp::Asin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.asin());
            }
        }
        UnaryOp::Acos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.acos());
            }
        }
        UnaryOp::Sinh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sinh());
            }
        }
        UnaryOp::Cosh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cosh());
            }
        }
        UnaryOp::Asinh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.asinh());
            }
        }
        UnaryOp::Acosh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.acosh());
            }
        }
        UnaryOp::Atanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.atanh());
            }
        }
        UnaryOp::Trunc => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.trunc());
            }
        }
    }
}

/// ReLU activation: max(0, x) with automatic SIMD dispatch
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::relu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::relu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                unary::relu_f16(a as *const half::f16, out as *mut half::f16, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                unary::relu_bf16(a as *const half::bf16, out as *mut half::bf16, len);
                return;
            }
            _ => {}
        }
    }

    relu_scalar(a, out, len);
}

/// Scalar ReLU for all Element types
#[inline]
unsafe fn relu_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        out_slice[i] = if a_slice[i] > zero { a_slice[i] } else { zero };
    }
}

/// Check for NaN values element-wise
///
/// Returns 1 (u8) if the value is NaN, 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isnan_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_nan() { 1 } else { 0 };
    }
}

/// Check for Inf values element-wise
///
/// Returns 1 (u8) if the value is infinite (positive or negative), 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isinf_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_infinite() { 1 } else { 0 };
    }
}

/// Clamp values to a range: out[i] = min(max(a[i], min_val), max_val)
///
/// On x86-64, dispatches to SIMD for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn clamp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::clamp;

        match T::DTYPE {
            DType::F32 => {
                clamp::clamp_f32(
                    a as *const f32,
                    out as *mut f32,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            DType::F64 => {
                clamp::clamp_f64(a as *const f64, out as *mut f64, len, min_val, max_val);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                clamp::clamp_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                clamp::clamp_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                    min_val as f32,
                    max_val as f32,
                );
                return;
            }
            _ => {}
        }
    }

    clamp_scalar(a, out, len, min_val, max_val);
}

/// Scalar clamp for all Element types
#[inline]
unsafe fn clamp_scalar<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let val = a_slice[i].to_f64();
        let clamped = if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        };
        out_slice[i] = T::from_f64(clamped);
    }
}
