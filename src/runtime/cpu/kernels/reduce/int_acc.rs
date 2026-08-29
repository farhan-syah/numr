//! Wide-accumulator epilogues for integer `sum`, `prod`, and `mean` reductions.
//!
//! Split out of `reduce/mod.rs` to keep that file under its line limit; these
//! two kernels are the integer counterpart of the SIMD float paths that live
//! beside them.

use super::super::wide_acc::{WideAcc, int_mean_from_sum};
use crate::dtype::Element;
use crate::ops::ReduceOp;

/// Mean over a contiguous dimension for integer dtypes, accumulating in i128.
#[inline]
pub(super) unsafe fn reduce_mean_int_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let mut sum = 0i128;
        for r in 0..reduce_size {
            sum = sum.saturating_add((*a.add(o * reduce_size + r)).to_i128());
        }
        *out.add(o) = int_mean_from_sum::<T>(sum, reduce_size);
    }
}

/// Sum or product over a contiguous dimension for integer dtypes, accumulating
/// in i128 and saturating once at narrow-back via [`WideAcc`]. `op` must be
/// `Sum` or `Prod`.
#[inline]
pub(super) unsafe fn reduce_sum_prod_int_kernel<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    let is_prod = matches!(op, ReduceOp::Prod);
    for o in 0..outer_size {
        let mut acc = if is_prod { i128::ONE } else { i128::ZERO };
        for r in 0..reduce_size {
            let v = i128::from_elem(*a.add(o * reduce_size + r));
            acc = if is_prod {
                acc.wide_mul(v)
            } else {
                acc.wide_add(v)
            };
        }
        *out.add(o) = acc.to_elem::<T>();
    }
}

#[cfg(test)]
mod tests {
    use super::super::reduce_kernel;
    use crate::ops::ReduceOp;

    /// Catches an integer `mean` that sums in the element type.
    ///
    /// The sum of the two elements needs 33 bits, but the mean is exactly
    /// 2_000_000_000, which i32 holds. Summing in i32 panics on the overflow in
    /// a debug build; in a release build it wraps to -294_967_296 and the
    /// division then reports -147_483_648. Unlike a plain sum, a division is
    /// not recoverable from a wrapped total.
    #[test]
    fn test_mean_i32_sums_in_a_wider_integer() {
        let a = [2_000_000_000i32, 2_000_000_000];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Mean, a.as_ptr(), out.as_mut_ptr(), 2, 1);
        }

        assert_eq!(out[0], 2_000_000_000);
    }

    /// Pins the integer `mean` rounding convention: truncate toward zero, which
    /// is what the previous float-division epilogue did for every sum it could
    /// represent.
    #[test]
    fn test_mean_i32_truncates_toward_zero() {
        let a = [7i32, 0, -7, 0];
        let mut out = [0i32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Mean, a.as_ptr(), out.as_mut_ptr(), 2, 2);
        }

        assert_eq!(out, [3, -3]);
    }

    /// A non-overflowing integer sum still returns the exact ordinary value:
    /// widening the accumulator must not perturb a total that already fit.
    #[test]
    fn test_sum_i32_exact_when_not_overflowing() {
        let a = [1i32, 2, 3, 4];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out[0], 10);
    }

    /// A sum whose total exceeds `i32::MAX` must saturate, not wrap negative.
    #[test]
    fn test_sum_i32_saturates_on_overflow() {
        let a = [i32::MAX, i32::MAX];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 2, 1);
        }

        assert_eq!(out[0], i32::MAX);
    }

    /// A sum whose total goes below `i32::MIN` must saturate at `i32::MIN`.
    #[test]
    fn test_sum_i32_saturates_below_min() {
        let a = [i32::MIN, i32::MIN];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 2, 1);
        }

        assert_eq!(out[0], i32::MIN);
    }

    /// An overflowing product saturates to `i32::MAX` when the true sign is
    /// positive.
    #[test]
    fn test_prod_i32_saturates_to_max() {
        let a = [100_000i32, 100_000, 100_000];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Prod, a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out[0], i32::MAX);
    }

    /// An overflowing product saturates to `i32::MIN` when the true sign is
    /// negative.
    #[test]
    fn test_prod_i32_saturates_to_min() {
        let a = [-100_000i32, 100_000, 100_000];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Prod, a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out[0], i32::MIN);
    }

    /// A u32 sum that exceeds `u32::MAX` must saturate there, never wrap to a
    /// small value.
    #[test]
    fn test_sum_u32_saturates_at_max() {
        let a = [u32::MAX, u32::MAX];
        let mut out = [0u32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 2, 1);
        }

        assert_eq!(out[0], u32::MAX);
    }

    /// A u32 product that exceeds `u32::MAX` must saturate there, never wrap
    /// to a small value.
    #[test]
    fn test_prod_u32_saturates_at_max() {
        let a = [u32::MAX, 2u32];
        let mut out = [0u32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Prod, a.as_ptr(), out.as_mut_ptr(), 2, 1);
        }

        assert_eq!(out[0], u32::MAX);
    }

    /// A non-overflowing integer product still returns the exact ordinary
    /// value: widening the accumulator must not perturb a total that already
    /// fit.
    #[test]
    fn test_prod_i32_exact_when_not_overflowing() {
        let a = [2i32, 3, 4];
        let mut out = [0i32; 1];

        unsafe {
            reduce_kernel(ReduceOp::Prod, a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out[0], 24);
    }
}
