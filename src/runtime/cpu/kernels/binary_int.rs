//! Wrapping integer arithmetic for the element-wise binary kernels.
//!
//! Rust's `+`, `-` and `*` panic on integer overflow in a debug build and wrap
//! in a release build, and `/` panics on a zero divisor in both. Neither is what
//! a tensor op may do: the result must not depend on the build profile, and a
//! kernel must not abort the process over one element's data.
//!
//! The convention this file implements is the one
//! [`wide_acc`](super::wide_acc) documents and the SIMD i32 kernel already
//! follows: **element-wise integer ops wrap, accumulators saturate**.
//!
//! * add, sub, mul — `wrapping_*`.
//! * div — a zero divisor yields 0, and `i32::MIN / -1` yields `i32::MIN`
//!   (`wrapping_div`) rather than overflowing.
//! * pow — [`pow_elem`](super::ipow::pow_elem), which is exact and saturating.
//!   Pow's result is an accumulator, so it saturates rather than wraps.
//!
//! The CUDA kernels in `src/runtime/cuda/kernels/binary_ops.cuh` mirror all of
//! this, so the two backends agree element for element.

use super::ipow::pow_elem;
use crate::dtype::{DType, Element};
use crate::ops::BinaryOp;

/// An integer element type with the wrapping arithmetic described above.
trait WrappingInt: Element {
    fn w_add(self, other: Self) -> Self;
    fn w_sub(self, other: Self) -> Self;
    fn w_mul(self, other: Self) -> Self;
    fn w_div(self, other: Self) -> Self;
}

macro_rules! impl_wrapping_int {
    ($($ty:ty),* $(,)?) => {
        $(
            impl WrappingInt for $ty {
                #[inline]
                fn w_add(self, other: Self) -> Self {
                    self.wrapping_add(other)
                }
                #[inline]
                fn w_sub(self, other: Self) -> Self {
                    self.wrapping_sub(other)
                }
                #[inline]
                fn w_mul(self, other: Self) -> Self {
                    self.wrapping_mul(other)
                }
                #[inline]
                fn w_div(self, other: Self) -> Self {
                    if other == 0 { 0 } else { self.wrapping_div(other) }
                }
            }
        )*
    };
}

impl_wrapping_int!(i8, i16, i32, i64, u8, u16, u32, u64);

/// One element of an integer binary operation.
#[inline]
fn elem<T: WrappingInt>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a.w_add(b),
        BinaryOp::Sub => a.w_sub(b),
        BinaryOp::Mul => a.w_mul(b),
        BinaryOp::Div => a.w_div(b),
        BinaryOp::Pow => pow_elem(a, b),
        BinaryOp::Max => {
            if a > b {
                a
            } else {
                b
            }
        }
        BinaryOp::Min => {
            if a < b {
                a
            } else {
                b
            }
        }
        BinaryOp::Atan2 => T::from_f64(a.to_f64().atan2(b.to_f64())),
    }
}

/// Run `op` over `len` contiguous elements when `T` is an integer dtype.
///
/// Returns `false` for every other dtype, leaving the caller on its own path.
/// The `op` match sits outside the loop so the branch is resolved once.
///
/// # Safety
///
/// `a`, `b` and `out` must be valid for `len` elements, and `out` must not
/// overlap `a` or `b` unless it is the same pointer.
#[inline]
pub(super) unsafe fn binary_int_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
) -> bool {
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the `T::DTYPE` arm that selected this branch proves
            // `T` is exactly `$ty`, so the three casts are identity casts.
            let a = a as *const $ty;
            let b = b as *const $ty;
            let out = out as *mut $ty;
            unsafe {
                for i in 0..len {
                    *out.add(i) = elem::<$ty>(op, *a.add(i), *b.add(i));
                }
            }
            true
        }};
    }

    match T::DTYPE {
        DType::I8 => run!(i8),
        DType::I16 => run!(i16),
        DType::I32 => run!(i32),
        DType::I64 => run!(i64),
        DType::U8 => run!(u8),
        DType::U16 => run!(u16),
        DType::U32 => run!(u32),
        DType::U64 => run!(u64),
        _ => false,
    }
}

/// One element of `op` when `T` is an integer dtype, `None` otherwise.
///
/// Used by the strided (broadcasting) kernel, which cannot hoist the dtype
/// dispatch out of its index walk.
#[inline]
pub(super) fn binary_int_elem<T: Element>(op: BinaryOp, a: T, b: T) -> Option<T> {
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the `T::DTYPE` arm that selected this branch proves `T`
            // is exactly `$ty`, so both transmutes are between the same type.
            let (x, y) = unsafe {
                (
                    std::mem::transmute_copy::<T, $ty>(&a),
                    std::mem::transmute_copy::<T, $ty>(&b),
                )
            };
            let r = elem::<$ty>(op, x, y);
            Some(unsafe { std::mem::transmute_copy::<$ty, T>(&r) })
        }};
    }

    match T::DTYPE {
        DType::I8 => run!(i8),
        DType::I16 => run!(i16),
        DType::I32 => run!(i32),
        DType::I64 => run!(i64),
        DType::U8 => run!(u8),
        DType::U16 => run!(u16),
        DType::U32 => run!(u32),
        DType::U64 => run!(u64),
        _ => None,
    }
}

/// Two wrapping steps composed: `elem(op2, elem(op1, a, b), c)`.
///
/// The fused kernels must answer exactly what the unfused sequence answers, so
/// they compose the same element function twice rather than computing in `f64`
/// and converting once. `None` for a non-integer dtype, as with
/// [`binary_int_elem`].
#[inline]
pub(super) fn binary_int_fused_elem<T: Element>(
    op1: BinaryOp,
    op2: BinaryOp,
    a: T,
    b: T,
    c: T,
) -> Option<T> {
    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the `T::DTYPE` arm that selected this branch proves `T`
            // is exactly `$ty`, so every transmute is between the same type.
            let (x, y, z) = unsafe {
                (
                    std::mem::transmute_copy::<T, $ty>(&a),
                    std::mem::transmute_copy::<T, $ty>(&b),
                    std::mem::transmute_copy::<T, $ty>(&c),
                )
            };
            let r = elem::<$ty>(op2, elem::<$ty>(op1, x, y), z);
            Some(unsafe { std::mem::transmute_copy::<$ty, T>(&r) })
        }};
    }

    match T::DTYPE {
        DType::I8 => run!(i8),
        DType::I16 => run!(i16),
        DType::I32 => run!(i32),
        DType::I64 => run!(i64),
        DType::U8 => run!(u8),
        DType::U16 => run!(u16),
        DType::U32 => run!(u32),
        DType::U64 => run!(u64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_mul_wrap_instead_of_panicking() {
        assert_eq!(elem::<u32>(BinaryOp::Sub, 0, 1), u32::MAX);
        assert_eq!(elem::<u32>(BinaryOp::Mul, u32::MAX, 2), u32::MAX - 1);
        assert_eq!(elem::<i32>(BinaryOp::Add, i32::MAX, 1), i32::MIN);
        assert_eq!(elem::<i8>(BinaryOp::Add, 127, 1), -128);
    }

    #[test]
    fn division_by_zero_yields_zero() {
        assert_eq!(elem::<u32>(BinaryOp::Div, 7, 0), 0);
        assert_eq!(elem::<i64>(BinaryOp::Div, -9, 0), 0);
        assert_eq!(elem::<i32>(BinaryOp::Div, i32::MIN, -1), i32::MIN);
    }

    #[test]
    fn non_overflowing_inputs_are_unchanged() {
        assert_eq!(elem::<u32>(BinaryOp::Add, 2, 3), 5);
        assert_eq!(elem::<i16>(BinaryOp::Div, -9, 2), -4);
        assert_eq!(elem::<u8>(BinaryOp::Max, 3, 9), 9);
    }

    #[test]
    fn fused_composition_wraps_at_every_step() {
        // (u8::MAX * 2) wraps to 254, then + 3 wraps to 1. Computing in f64 and
        // converting once would saturate to 255 instead.
        assert_eq!(
            binary_int_fused_elem::<u8>(BinaryOp::Mul, BinaryOp::Add, u8::MAX, 2, 3),
            Some(1)
        );
        assert_eq!(
            binary_int_fused_elem::<i32>(BinaryOp::Add, BinaryOp::Mul, i32::MAX, 1, 2),
            Some(0)
        );
        assert_eq!(
            binary_int_fused_elem::<i32>(BinaryOp::Mul, BinaryOp::Add, 3, 4, 5),
            Some(17)
        );
    }

    #[test]
    fn float_dtypes_are_not_claimed() {
        assert!(binary_int_elem::<f32>(BinaryOp::Add, 1.0, 2.0).is_none());
        assert!(
            binary_int_fused_elem::<f32>(BinaryOp::Mul, BinaryOp::Add, 1.0, 2.0, 3.0).is_none()
        );
        let (a, b) = ([1.0f64], [2.0f64]);
        let mut out = [0.0f64];
        let handled = unsafe {
            binary_int_kernel(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1)
        };
        assert!(!handled);
    }
}
