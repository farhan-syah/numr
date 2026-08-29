//! Scalar operation kernels
//!
//! Provides tensor-scalar operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

use super::ipow::pow_elem_scalar;
use crate::dtype::Element;
use crate::ops::BinaryOp;

/// Binary operation with a scalar (tensor op scalar) with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn scalar_op_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    scalar: f64,
    out: *mut T,
    len: usize,
) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::scalar;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                scalar::scalar_f32(op, a as *const f32, scalar as f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                scalar::scalar_f64(op, a as *const f64, scalar, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                scalar::scalar_f16(
                    op,
                    a as *const half::f16,
                    scalar as f32,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                scalar::scalar_bf16(
                    op,
                    a as *const half::bf16,
                    scalar as f32,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    scalar_op_kernel_scalar(op, a, scalar, out, len);
}

/// Scalar fallback for all Element types
#[inline]
unsafe fn scalar_op_kernel_scalar<T: Element>(
    op: BinaryOp,
    a: *const T,
    scalar: f64,
    out: *mut T,
    len: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let s = T::from_f64(scalar);

    match op {
        BinaryOp::Add => {
            for i in 0..len {
                out_slice[i] = a_slice[i] + s;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                out_slice[i] = a_slice[i] - s;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                out_slice[i] = a_slice[i] * s;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                out_slice[i] = a_slice[i] / s;
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                out_slice[i] = pow_elem_scalar(a_slice[i], scalar);
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] > s { a_slice[i] } else { s };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] < s { a_slice[i] } else { s };
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                let y = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(y.atan2(scalar));
            }
        }
    }
}

/// Reverse scalar subtract kernel: out[i] = scalar - a[i]
///
/// On x86-64, dispatches to SIMD (AVX-512/AVX2) for f32/f64.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn rsub_scalar_kernel<T: Element>(a: *const T, scalar: f64, out: *mut T, len: usize) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::scalar;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                scalar::rsub_scalar_f32(a as *const f32, scalar as f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                scalar::rsub_scalar_f64(a as *const f64, scalar, out as *mut f64, len);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                scalar::rsub_scalar_f16(
                    a as *const half::f16,
                    scalar as f32,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                scalar::rsub_scalar_bf16(
                    a as *const half::bf16,
                    scalar as f32,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback for other types
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let s = T::from_f64(scalar);

    for i in 0..len {
        out_slice[i] = s - a_slice[i];
    }
}
