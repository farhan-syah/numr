//! Wrapping integer `neg`, `abs` and `sign` for the element-wise unary kernels.
//!
//! The generic unary path converts each element to `f64`, applies the operation
//! and converts back. That is wrong for integers twice over:
//!
//! * `T::from_f64` saturates, so `neg(i32::MIN)` answers `i32::MAX` and
//!   `abs(i8::MIN)` answers `i8::MAX`. The convention
//!   [`wide_acc`](crate::runtime::cpu::kernels::wide_acc) documents, and that
//!   [`binary_int`](crate::runtime::cpu::kernels::binary_int) implements, is
//!   **element-wise integer ops wrap, accumulators saturate**. `neg` and `abs`
//!   are element-wise, so both must answer `MIN`.
//! * `f64` has 53 mantissa bits, so an `i64` or `u64` past 2^53 does not
//!   survive the round trip at all. `neg(9007199254740993)` came back one off.
//!
//! CUDA's `neg_i32` / `abs_i32` (`runtime/cuda/kernels/unary.cu`) and WebGPU's
//! `neg_i32` / `abs_i32` (`runtime/wgpu/shaders/unary_i32.wgsl`) both wrap
//! already, WGSL by definition and CUDA by two's-complement hardware, so this
//! file is what brings CPU into line with them rather than the reverse.
//!
//! Rust's `-x` and `x.abs()` panic on overflow in a debug build and wrap in a
//! release build, so neither may appear here: a tensor op's answer must not
//! depend on the build profile.

use crate::dtype::{DType, Element};
use crate::ops::UnaryOp;

/// An integer element type with the wrapping unary arithmetic described above.
trait WrappingIntUnary: Element {
    fn w_neg(self) -> Self;
    fn w_abs(self) -> Self;
    fn w_sign(self) -> Self;
}

macro_rules! impl_signed_unary {
    ($($ty:ty),* $(,)?) => {
        $(
            impl WrappingIntUnary for $ty {
                #[inline]
                fn w_neg(self) -> Self {
                    self.wrapping_neg()
                }
                #[inline]
                fn w_abs(self) -> Self {
                    self.wrapping_abs()
                }
                #[inline]
                fn w_sign(self) -> Self {
                    if self > 0 {
                        1
                    } else if self < 0 {
                        -1
                    } else {
                        0
                    }
                }
            }
        )*
    };
}

macro_rules! impl_unsigned_unary {
    ($($ty:ty),* $(,)?) => {
        $(
            impl WrappingIntUnary for $ty {
                #[inline]
                fn w_neg(self) -> Self {
                    self.wrapping_neg()
                }
                #[inline]
                fn w_abs(self) -> Self {
                    self
                }
                #[inline]
                fn w_sign(self) -> Self {
                    if self > 0 { 1 } else { 0 }
                }
            }
        )*
    };
}

impl_signed_unary!(i8, i16, i32, i64);
impl_unsigned_unary!(u8, u16, u32, u64);

/// One element of an integer unary operation.
#[inline]
fn elem<T: WrappingIntUnary>(op: UnaryOp, a: T) -> Option<T> {
    match op {
        UnaryOp::Neg => Some(a.w_neg()),
        UnaryOp::Abs => Some(a.w_abs()),
        UnaryOp::Sign => Some(a.w_sign()),
        _ => None,
    }
}

/// Run `neg`, `abs` or `sign` over `len` contiguous elements when `T` is an
/// integer dtype.
///
/// Returns `false` for every other operation and every other dtype, leaving the
/// caller on its generic `f64` path. The `op` match sits inside the per-dtype
/// branch but outside the loop, so the branch is resolved once.
///
/// # Safety
///
/// `a` and `out` must be valid for `len` elements, and `out` must not overlap
/// `a` unless it is the same pointer.
#[inline]
pub(super) unsafe fn unary_int_kernel<T: Element>(
    op: UnaryOp,
    a: *const T,
    out: *mut T,
    len: usize,
) -> bool {
    if !matches!(op, UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sign) {
        return false;
    }

    macro_rules! run {
        ($ty:ty) => {{
            // SAFETY: the `T::DTYPE` arm that selected this branch proves `T`
            // is exactly `$ty`, so both casts are identity casts.
            let a = a as *const $ty;
            let out = out as *mut $ty;
            unsafe {
                for i in 0..len {
                    // `None` only for an op the guard above excluded.
                    if let Some(v) = elem::<$ty>(op, *a.add(i)) {
                        *out.add(i) = v;
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_and_abs_wrap_at_the_signed_minimum() {
        assert_eq!(elem::<i32>(UnaryOp::Neg, i32::MIN), Some(i32::MIN));
        assert_eq!(elem::<i32>(UnaryOp::Abs, i32::MIN), Some(i32::MIN));
        assert_eq!(elem::<i8>(UnaryOp::Neg, i8::MIN), Some(i8::MIN));
        assert_eq!(elem::<i8>(UnaryOp::Abs, i8::MIN), Some(i8::MIN));
        assert_eq!(elem::<i64>(UnaryOp::Neg, i64::MIN), Some(i64::MIN));
        assert_eq!(elem::<i64>(UnaryOp::Abs, i64::MIN), Some(i64::MIN));
    }

    #[test]
    fn wide_integers_survive_exactly() {
        // 2^53 + 1 has no f64 representation, so the old to_f64 round trip
        // answered one less than this.
        assert_eq!(
            elem::<i64>(UnaryOp::Neg, 9007199254740993),
            Some(-9007199254740993)
        );
        assert_eq!(elem::<u64>(UnaryOp::Neg, 1), Some(u64::MAX));
    }

    #[test]
    fn sign_reports_three_values() {
        assert_eq!(elem::<i16>(UnaryOp::Sign, -7), Some(-1));
        assert_eq!(elem::<i16>(UnaryOp::Sign, 0), Some(0));
        assert_eq!(elem::<i16>(UnaryOp::Sign, 7), Some(1));
        assert_eq!(elem::<i16>(UnaryOp::Sign, i16::MIN), Some(-1));
        assert_eq!(elem::<u8>(UnaryOp::Sign, 0), Some(0));
        assert_eq!(elem::<u8>(UnaryOp::Sign, 200), Some(1));
    }

    #[test]
    fn abs_is_the_identity_on_unsigned_dtypes() {
        assert_eq!(elem::<u32>(UnaryOp::Abs, u32::MAX), Some(u32::MAX));
    }

    #[test]
    fn other_ops_and_dtypes_are_not_claimed() {
        let (a, mut out) = ([1i32], [0i32]);
        let handled = unsafe { unary_int_kernel(UnaryOp::Sqrt, a.as_ptr(), out.as_mut_ptr(), 1) };
        assert!(!handled);

        let (a, mut out) = ([1.0f64], [0.0f64]);
        let handled = unsafe { unary_int_kernel(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 1) };
        assert!(!handled);
    }
}
