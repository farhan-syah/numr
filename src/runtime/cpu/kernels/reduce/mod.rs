//! Reduction operation kernels
//!
//! Provides reduction operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

mod int_acc;
mod special;

pub use special::{
    argmax_kernel, argmin_kernel, softmax_bwd_kernel, softmax_kernel, variance_kernel,
};

use crate::dtype::Element;
use crate::ops::{AccumulationPrecision, ReduceOp};
use int_acc::{reduce_mean_int_kernel, reduce_sum_prod_int_kernel};

/// Reduce along contiguous dimension with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    // A zero-length reduce dimension leaves `Max` and `Min` with no element to
    // seed from, and every path below seeds them by reading `a[o * 0]`. The
    // input allocation is empty here — `CpuRuntime::allocate` hands back a null
    // pointer for zero bytes — so that read is a null dereference, not merely a
    // wrong answer. `Sum`, `Mean`, `Prod`, `All` and `Any` all start from their
    // identity and iterate zero times, so they need no special case; these two
    // are given the identity of their own reduction instead.
    if reduce_size == 0 {
        match op {
            ReduceOp::Max => {
                let identity = T::from_f64(T::DTYPE.min_value());
                for o in 0..outer_size {
                    *out.add(o) = identity;
                }
                return;
            }
            ReduceOp::Min => {
                let identity = T::from_f64(T::DTYPE.max_value());
                for o in 0..outer_size {
                    *out.add(o) = identity;
                }
                return;
            }
            _ => {}
        }
    }

    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::reduce;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                reduce::reduce_f32(
                    op,
                    a as *const f32,
                    out as *mut f32,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            DType::F64 => {
                reduce::reduce_f64(
                    op,
                    a as *const f64,
                    out as *mut f64,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                reduce::reduce_f16(
                    op,
                    a as *const half::f16,
                    out as *mut half::f16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                reduce::reduce_bf16(
                    op,
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback. A float narrower than F32 must never accumulate in its
    // own dtype: the running sum saturates and returns a constant. Widen to
    // f32 and narrow only the final result. This is what reaches FP8 on every
    // architecture, and F16/BF16 on architectures without the SIMD paths above.
    //
    // All/Any hold no accumulator, and routing them here would bounce back
    // into this function through `reduce_kernel_acc`.
    if T::DTYPE.is_narrow_float() && !matches!(op, ReduceOp::All | ReduceOp::Any) {
        reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size);
        return;
    }

    // Integer `sum`, `prod`, and `mean` all build a running total wider than
    // one element, so accumulating in the element type wraps (release) or
    // panics (debug) on a total the output dtype cannot represent even though
    // the final result fits. Widen to i128, then narrow once with saturation
    // (`WideAcc`) — the same convention `cumsum`, `cumprod`, and `matmul` use.
    // `mean` additionally divides the wide sum before narrowing.
    if T::DTYPE.is_int() && matches!(op, ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Mean) {
        match op {
            ReduceOp::Mean => reduce_mean_int_kernel(a, out, reduce_size, outer_size),
            _ => reduce_sum_prod_int_kernel(op, a, out, reduce_size, outer_size),
        }
        return;
    }

    reduce_kernel_scalar(op, a, out, reduce_size, outer_size);
}

/// Scalar reduce kernel for all Element types
#[inline]
unsafe fn reduce_kernel_scalar<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum;
            }
        }
        ReduceOp::Mean => {
            let scale = 1.0 / reduce_size as f64;
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = T::from_f64(sum.to_f64() * scale);
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = max_val;
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = min_val;
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = T::one();
                for r in 0..reduce_size {
                    prod = prod * *a.add(o * reduce_size + r);
                }
                *out.add(o) = prod;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions - convert to/from f64 (0.0 = false, non-zero = true)
            let is_any = matches!(op, ReduceOp::Any);
            for o in 0..outer_size {
                let mut result = !is_any; // All starts true, Any starts false
                for r in 0..reduce_size {
                    let val = (*a.add(o * reduce_size + r)).to_f64() != 0.0;
                    if is_any {
                        result = result || val;
                    } else {
                        result = result && val;
                    }
                }
                *out.add(o) = T::from_f64(if result { 1.0 } else { 0.0 });
            }
        }
    }
}

/// Reduce kernel with explicit accumulation precision
///
/// For reduced-precision types (F16, BF16, FP8), this allows accumulating
/// in a higher precision format for better numerical stability.
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
/// * `precision` - Accumulation precision
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel_with_precision<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
    precision: AccumulationPrecision,
) {
    match precision {
        AccumulationPrecision::Native => {
            // Use native type accumulation (existing behavior)
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
            // Accumulate in f32 for better precision
            // BF16 uses f32 on CPU since there's no native bf16 arithmetic
            reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP64 => {
            // Accumulate in f64 for maximum precision (math/science)
            reduce_kernel_acc::<T, f64>(op, a, out, reduce_size, outer_size);
        }
    }
}

/// Trait for accumulation types (f32, f64) used in precision-aware reductions.
///
/// This allows a single generic implementation for both FP32 and FP64 accumulation,
/// avoiding code duplication while maintaining type safety and performance.
///
/// Uses `Into<f64>` for output conversion, `acc_in` for input (f64 -> Self).
pub trait Accumulator: Copy + PartialOrd + PartialEq + Into<f64> {
    const ZERO: Self;
    const ONE: Self;
    /// Convert f64 input to accumulator type
    fn acc_in(v: f64) -> Self;
    fn acc_add(self, other: Self) -> Self;
    fn acc_mul(self, other: Self) -> Self;
    fn acc_div(self, n: usize) -> Self;
}

impl Accumulator for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f32
    }
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f64
    }
}

/// Generic reduce kernel with configurable accumulation precision.
///
/// Converts input elements to accumulator type A, performs reduction, then converts back to T.
#[inline]
unsafe fn reduce_kernel_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    // Same null-dereference guard as `reduce_kernel`: `Max` and `Min` seed
    // themselves from `a[o * 0]`, which is a read from the null pointer an empty
    // CPU allocation hands back. See the comment there.
    if reduce_size == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min) {
        reduce_kernel(op, a, out, reduce_size, outer_size);
        return;
    }

    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.into());
            }
        }
        ReduceOp::Mean => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.acc_div(reduce_size).into());
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = T::from_f64(max_val.into());
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = T::from_f64(min_val.into());
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = A::ONE;
                for r in 0..reduce_size {
                    prod = prod.acc_mul(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(prod.into());
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions don't benefit from higher precision accumulation
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
    }
}

// Unit tests for the integer `sum`/`prod`/`mean` accumulation paths live in
// `int_acc.rs`, beside the kernels they exercise.
