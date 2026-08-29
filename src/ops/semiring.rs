//! Semiring operation types for generalized matrix multiplication.
//!
//! Standard matmul uses the (+, ×) semiring. This module defines alternative
//! semirings for graph algorithms, scheduling, and fuzzy logic:
//!
//! - `MinPlus` — shortest paths, tropical geometry
//! - `MaxPlus` — longest paths, scheduling
//! - `MaxMin` — bottleneck/network capacity
//! - `MinMax` — fuzzy relations
//! - `OrAnd` — boolean matmul, transitive closure
//! - `PlusMax` — dynamic programming formulations

use crate::dtype::{DType, Element};

/// Semiring operations for generalized matrix multiplication.
///
/// Each variant encodes both the reduce (⊕) and combine (⊗) operations:
/// `C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SemiringOp {
    /// (min, +) — shortest path distances
    MinPlus,
    /// (max, +) — longest path distances
    MaxPlus,
    /// (max, min) — bottleneck / max-capacity paths
    MaxMin,
    /// (min, max) — fuzzy relations
    MinMax,
    /// (OR, AND) — boolean matmul / transitive closure
    OrAnd,
    /// (+, max) — certain DP formulations
    PlusMax,
}

/// Wrapping add for `combine`'s `+` (`MinPlus`/`MaxPlus`) and `PlusMax`'s
/// `reduce`.
///
/// Rust's plain `+` panics on integer overflow in a debug build and wraps in
/// release, so a library op cannot use it directly — the same defect class
/// already fixed for element-wise binary ops in
/// `src/runtime/cpu/kernels/binary_int.rs`. Per the convention documented in
/// `src/runtime/cpu/kernels/wide_acc.rs`, **element-wise ops wrap, only
/// accumulators saturate**.
///
/// `combine` is element-wise by definition (one op, one pair of elements), so
/// it wraps. `PlusMax`'s `reduce` folds across K and looks like an
/// accumulator, but it is not one here: the CUDA reference kernel
/// (`runtime/cuda/kernels/semiring_matmul_ops.cuh`, `numr_sr_add`) does the
/// addition in the unsigned type of the same width and casts back, i.e. it
/// wraps, precisely because `combine` for `MinPlus`/`MaxPlus` and `reduce` for
/// `PlusMax` are both "one elementwise addition" to that kernel — there is no
/// separate saturating accumulator path for `+` in this semiring family.
/// Wrapping here, not saturating, is what keeps CPU and CUDA producing
/// identical results.
///
/// `binary_int_elem` in `src/runtime/cpu/kernels/binary_int.rs` already
/// implements this wrapping convention, but it is `pub(super)` — visible only
/// inside `runtime::cpu::kernels`, not from `ops::semiring`. Reusing it would
/// require widening that visibility for a single call site, so this defines
/// its own narrow helper instead: `SemiringOp::validate_dtype` only ever
/// admits `I32`/`I64` as integer element types here (`OrAnd`, the only other
/// integer-carrying variant, never calls this), so only those two need a
/// dedicated path.
#[inline]
fn wrapping_add<T: Element>(a: T, b: T) -> T {
    match T::DTYPE {
        DType::I32 => {
            let r = bytemuck::cast::<T, i32>(a).wrapping_add(bytemuck::cast::<T, i32>(b));
            bytemuck::cast::<i32, T>(r)
        }
        DType::I64 => {
            let r = bytemuck::cast::<T, i64>(a).wrapping_add(bytemuck::cast::<T, i64>(b));
            bytemuck::cast::<i64, T>(r)
        }
        _ => a + b,
    }
}

impl SemiringOp {
    /// Returns the identity element for the reduce operation as f64.
    ///
    /// - min → +∞
    /// - max → -∞
    /// - sum (+) → 0
    /// - OR → 0 (false)
    // KEEP IN SYNC: runtime/cuda/kernels/semiring_matmul.cu identity init
    // KEEP IN SYNC: runtime/wgpu/shaders/semiring_matmul_*_f32.wgsl and
    // semiring_matmul_*_i32.wgsl, whose `acc` initializers hold these identities
    pub fn reduce_identity_f64(self) -> f64 {
        match self {
            SemiringOp::MinPlus | SemiringOp::MinMax => f64::INFINITY,
            SemiringOp::MaxPlus | SemiringOp::MaxMin => f64::NEG_INFINITY,
            SemiringOp::OrAnd => 0.0,
            SemiringOp::PlusMax => 0.0,
        }
    }

    /// Returns the identity element for the reduce operation for a given Element type.
    pub fn reduce_identity<T: Element>(self) -> T {
        T::from_f64(self.reduce_identity_f64())
    }

    /// Apply the combine (⊗) operation.
    #[inline]
    pub fn combine<T: Element>(self, a: T, b: T) -> T {
        match self {
            SemiringOp::MinPlus | SemiringOp::MaxPlus => wrapping_add(a, b),
            SemiringOp::MaxMin => {
                // combine = min
                if a <= b { a } else { b }
            }
            SemiringOp::MinMax | SemiringOp::PlusMax => {
                // combine = max
                if a >= b { a } else { b }
            }
            SemiringOp::OrAnd => {
                // AND: both nonzero → 1, else 0
                let az = a.to_f64();
                let bz = b.to_f64();
                if az != 0.0 && bz != 0.0 {
                    T::one()
                } else {
                    T::zero()
                }
            }
        }
    }

    /// Apply the reduce (⊕) operation: accumulate `val` into `acc`.
    #[inline]
    pub fn reduce<T: Element>(self, acc: T, val: T) -> T {
        match self {
            SemiringOp::MinPlus | SemiringOp::MinMax => {
                // min
                if val <= acc { val } else { acc }
            }
            SemiringOp::MaxPlus | SemiringOp::MaxMin => {
                // max
                if val >= acc { val } else { acc }
            }
            SemiringOp::OrAnd => {
                // OR: any nonzero → 1
                let az = acc.to_f64();
                let vz = val.to_f64();
                if az != 0.0 || vz != 0.0 {
                    T::one()
                } else {
                    T::zero()
                }
            }
            SemiringOp::PlusMax => {
                // sum
                wrapping_add(acc, val)
            }
        }
    }

    /// Combine operation for MaxMin is min, but for MinMax the combine is max.
    /// This is already handled in `combine` above. This helper documents
    /// the combine operation name for display/debug.
    pub fn combine_name(self) -> &'static str {
        match self {
            SemiringOp::MinPlus | SemiringOp::MaxPlus => "add",
            SemiringOp::MaxMin => "min",
            SemiringOp::MinMax | SemiringOp::PlusMax => "max",
            SemiringOp::OrAnd => "and",
        }
    }

    /// Name of the reduce operation for display/debug.
    pub fn reduce_name(self) -> &'static str {
        match self {
            SemiringOp::MinPlus | SemiringOp::MinMax => "min",
            SemiringOp::MaxPlus | SemiringOp::MaxMin => "max",
            SemiringOp::OrAnd => "or",
            SemiringOp::PlusMax => "add",
        }
    }

    /// Validate that the given dtype is supported for this semiring operation.
    pub fn validate_dtype(self, dtype: DType) -> bool {
        match self {
            SemiringOp::OrAnd => matches!(dtype, DType::Bool | DType::U8),
            _ => {
                matches!(dtype, DType::F32 | DType::F64 | DType::I32 | DType::I64) || {
                    #[cfg(feature = "f16")]
                    if matches!(dtype, DType::F16 | DType::BF16) {
                        return true;
                    }
                    #[cfg(feature = "fp8")]
                    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
                        return true;
                    }
                    false
                }
            }
        }
    }
}

impl std::fmt::Display for SemiringOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (reduce, combine) = match self {
            SemiringOp::MinPlus => ("min", "+"),
            SemiringOp::MaxPlus => ("max", "+"),
            SemiringOp::MaxMin => ("max", "min"),
            SemiringOp::MinMax => ("min", "max"),
            SemiringOp::OrAnd => ("OR", "AND"),
            SemiringOp::PlusMax => ("+", "max"),
        };
        write!(f, "({}, {})", reduce, combine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_plus_combine() {
        // combine is addition
        assert_eq!(SemiringOp::MinPlus.combine(3.0f32, 5.0), 8.0);
    }

    #[test]
    fn test_min_plus_reduce() {
        // reduce is min
        assert_eq!(SemiringOp::MinPlus.reduce(3.0f32, 5.0), 3.0);
        assert_eq!(SemiringOp::MinPlus.reduce(5.0f32, 3.0), 3.0);
    }

    #[test]
    fn test_max_min_combine() {
        // combine is min
        assert_eq!(SemiringOp::MaxMin.combine(3.0f32, 5.0), 3.0);
    }

    #[test]
    fn test_max_min_reduce() {
        // reduce is max
        assert_eq!(SemiringOp::MaxMin.reduce(3.0f32, 5.0), 5.0);
    }

    #[test]
    fn test_min_max_combine() {
        // MinMax = (min, max): combine = max, reduce = min
        assert_eq!(SemiringOp::MinMax.combine(3.0f32, 5.0), 5.0);
    }

    #[test]
    fn test_min_max_reduce() {
        assert_eq!(SemiringOp::MinMax.reduce(3.0f32, 5.0), 3.0);
    }

    #[test]
    fn test_plus_max() {
        // PlusMax = (+, max): combine = max, reduce = +
        assert_eq!(SemiringOp::PlusMax.combine(3.0f32, 5.0), 5.0);
        assert_eq!(SemiringOp::PlusMax.reduce(3.0f32, 5.0), 8.0);
    }

    #[test]
    fn test_identity_elements() {
        assert_eq!(SemiringOp::MinPlus.reduce_identity::<f32>(), f32::INFINITY);
        assert_eq!(
            SemiringOp::MaxPlus.reduce_identity::<f32>(),
            f32::NEG_INFINITY
        );
        assert_eq!(SemiringOp::PlusMax.reduce_identity::<f32>(), 0.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", SemiringOp::MinPlus), "(min, +)");
        assert_eq!(format!("{}", SemiringOp::OrAnd), "(OR, AND)");
    }

    #[test]
    fn test_combine_int_wraps_at_overflow_boundary() {
        // MinPlus/MaxPlus combine is `+`. On the pre-fix code this panics in a
        // debug build instead of wrapping.
        assert_eq!(SemiringOp::MinPlus.combine(i32::MAX, 1i32), i32::MIN);
        assert_eq!(SemiringOp::MaxPlus.combine(i32::MAX, 1i32), i32::MIN);
        assert_eq!(SemiringOp::MinPlus.combine(i64::MAX, 1i64), i64::MIN);
    }

    #[test]
    fn test_plus_max_reduce_int_wraps_at_overflow_boundary() {
        // PlusMax's reduce is `+` too, and it wraps rather than saturates:
        // the CUDA reference kernel does the same elementwise addition (in
        // the unsigned type of the same width) for this path, so CPU must
        // agree rather than clamp.
        assert_eq!(SemiringOp::PlusMax.reduce(i32::MAX, 1i32), i32::MIN);
        assert_eq!(SemiringOp::PlusMax.reduce(i64::MAX, 1i64), i64::MIN);
    }

    #[test]
    fn test_combine_and_reduce_int_non_overflowing_unchanged() {
        assert_eq!(SemiringOp::MinPlus.combine(3i32, 5i32), 8);
        assert_eq!(SemiringOp::PlusMax.reduce(3i32, 5i32), 8);
        assert_eq!(SemiringOp::MaxPlus.combine(3i64, 5i64), 8);
    }

    #[test]
    fn test_validate_dtype() {
        assert!(SemiringOp::MinPlus.validate_dtype(DType::F32));
        assert!(SemiringOp::MinPlus.validate_dtype(DType::F64));
        assert!(SemiringOp::MinPlus.validate_dtype(DType::I32));
        assert!(!SemiringOp::MinPlus.validate_dtype(DType::Bool));

        assert!(SemiringOp::OrAnd.validate_dtype(DType::Bool));
        assert!(!SemiringOp::OrAnd.validate_dtype(DType::F32));
    }
}
