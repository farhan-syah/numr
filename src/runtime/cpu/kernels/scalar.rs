//! Scalar operation kernels
//!
//! Provides tensor-scalar operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

use super::binary_int::binary_int_elem;
use super::ipow::pow_elem_scalar;
use super::wide_acc::WideAcc;
use crate::dtype::Element;
use crate::ops::BinaryOp;

/// The f32 operation a narrow-float scalar op runs, or `None` when the op
/// already widens on its own.
///
/// A narrow float (F16, BF16, FP8) cannot hold the scalar, so rounding the
/// scalar into `T` before the operation rounds twice: once into the element
/// type, and again when the result is stored. `wide_acc`'s convention is one
/// narrowing at write-out, and `WideAcc for f32` is the accumulator it names
/// for every `is_narrow_float()` dtype, so these ops run in f32 against the
/// unrounded scalar. The CUDA FP8 kernels do the same (`NUMR_SCALAR_ROW_FP8`
/// in `scalar_ops.cuh`), so both backends answer bit for bit alike.
///
/// Pow and Atan2 return `None`: they already widen through f64 in the callers
/// below, and pow's exponent must never be rounded into `T` at all.
#[inline]
fn narrow_float_scalar_op<T: Element>(op: BinaryOp) -> Option<fn(f32, f32) -> f32> {
    if !T::DTYPE.is_narrow_float() {
        return None;
    }
    match op {
        BinaryOp::Add => Some(|x, s| x + s),
        BinaryOp::Sub => Some(|x, s| x - s),
        BinaryOp::Mul => Some(|x, s| x * s),
        BinaryOp::Div => Some(|x, s| x / s),
        BinaryOp::Max => Some(|x, s| if x > s { x } else { s }),
        BinaryOp::Min => Some(|x, s| if x < s { x } else { s }),
        BinaryOp::Pow | BinaryOp::Atan2 => None,
    }
}

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

    // Narrow floats compute against the unrounded scalar and narrow once at the
    // store; see `narrow_float_scalar_op`.
    if let Some(f) = narrow_float_scalar_op::<T>(op) {
        let s32 = scalar as f32;
        for i in 0..len {
            out_slice[i] = f(f32::from_elem(a_slice[i]), s32).to_elem::<T>();
        }
        return;
    }

    let s = T::from_f64(scalar);

    // Integer add/sub/mul WRAP and integer div treats a zero divisor as 0.
    // `T`'s bare `+`/`-`/`*` panic on overflow in a debug build and `/` panics
    // on a zero divisor in both, so a tensor op must not use them. This is the
    // same element function the tensor-tensor kernels use (see
    // `binary_int.rs`), so `add_scalar(a, s)` and `add(a, full(s))` agree.
    //
    // Pow keeps `pow_elem_scalar`: its exponent must stay an f64, not be
    // rounded into `T`. Max, min and atan2 cannot overflow.
    if matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    ) && T::DTYPE.is_int()
    {
        for i in 0..len {
            // `None` only for a non-integer dtype, which the guard excluded.
            if let Some(v) = binary_int_elem(op, a_slice[i], s) {
                out_slice[i] = v;
            }
        }
        return;
    }

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

    // Narrow floats compute against the unrounded scalar and narrow once at the
    // store; see `narrow_float_scalar_op`. Only the operand order differs from
    // `BinaryOp::Sub` there.
    if T::DTYPE.is_narrow_float() {
        let s32 = scalar as f32;
        for i in 0..len {
            out_slice[i] = (s32 - f32::from_elem(a_slice[i])).to_elem::<T>();
        }
        return;
    }

    let s = T::from_f64(scalar);

    // Integer subtraction wraps; see the note in `scalar_op_kernel_scalar`.
    // Only the operand order differs here.
    if T::DTYPE.is_int() {
        for i in 0..len {
            if let Some(v) = binary_int_elem(BinaryOp::Sub, s, a_slice[i]) {
                out_slice[i] = v;
            }
        }
        return;
    }

    for i in 0..len {
        out_slice[i] = s - a_slice[i];
    }
}
