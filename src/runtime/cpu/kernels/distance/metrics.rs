//! Per-pair distance metrics.
//!
//! Each function reduces one pair of `d`-component vectors to a single value.
//! Every one of them is generic over the accumulator `A` rather than computing
//! in the element type, so a narrow-float tensor accumulates in f32 and matches
//! the `AccT` CUDA's `distance.cu` uses. See [`DistAcc`] for which accumulator
//! each element type gets.

use super::acc::DistAcc;
use crate::dtype::Element;

/// Squared Euclidean distance: `sum((a - b)^2)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn sqeuclidean<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        let diff = A::widen(*a.add(k)) - A::widen(*b.add(k));
        sum = sum + diff * diff;
    }
    sum
}

/// Euclidean (L2) distance.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn euclidean<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    sqeuclidean::<T, A>(a, b, d).sqrt()
}

/// Manhattan (L1) distance: `sum(|a - b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn manhattan<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        sum = sum + (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs();
    }
    sum
}

/// Chebyshev (L-infinity) distance: `max(|a - b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn chebyshev<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut max = A::zero();
    for k in 0..d {
        let abs_diff = (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs();
        if abs_diff > max {
            max = abs_diff;
        }
    }
    max
}

/// Minkowski (Lp) distance: `sum(|a - b|^p)^(1/p)`.
///
/// `p` arrives in the accumulator's precision, never rounded into the element
/// type first: an exponent rounded into F16 changes which curve is being
/// measured, not just the last digit of the answer.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn minkowski<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize, p: A) -> A {
    let mut sum = A::zero();
    for k in 0..d {
        sum = sum + (A::widen(*a.add(k)) - A::widen(*b.add(k))).abs().powf(p);
    }
    sum.powf(A::one() / p)
}

/// Cosine distance: `1 - dot(a, b) / (|a| * |b|)`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn cosine<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut dot = A::zero();
    let mut norm_a = A::zero();
    let mut norm_b = A::zero();

    for k in 0..d {
        let ak = A::widen(*a.add(k));
        let bk = A::widen(*b.add(k));
        dot = dot + ak * bk;
        norm_a = norm_a + ak * ak;
        norm_b = norm_b + bk * bk;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom.is_zero() {
        A::zero()
    } else {
        A::one() - dot / denom
    }
}

/// Correlation distance: `1 - Pearson r`.
///
/// Measures how similar the patterns in two vectors are, invariant to linear
/// transformations. 0 means perfect positive correlation, 2 perfect negative.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn correlation<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let d_a = A::count(d);

    let mut sum_a = A::zero();
    let mut sum_b = A::zero();
    for k in 0..d {
        sum_a = sum_a + A::widen(*a.add(k));
        sum_b = sum_b + A::widen(*b.add(k));
    }
    let mean_a = sum_a / d_a;
    let mean_b = sum_b / d_a;

    let mut cov = A::zero();
    let mut var_a = A::zero();
    let mut var_b = A::zero();
    for k in 0..d {
        let da = A::widen(*a.add(k)) - mean_a;
        let db = A::widen(*b.add(k)) - mean_b;
        cov = cov + da * db;
        var_a = var_a + da * da;
        var_b = var_b + db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom.is_zero() {
        A::zero()
    } else {
        A::one() - cov / denom
    }
}

/// Hamming distance: the fraction of positions where the vectors differ.
///
/// For continuous-valued vectors this counts exact inequality. Widening is
/// value-preserving, so comparing widened components decides the same way
/// comparing elements would; only the count and the division change width.
///
/// Returns a value in `[0, 1]`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn hamming<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut count = A::zero();
    for k in 0..d {
        if A::widen(*a.add(k)) != A::widen(*b.add(k)) {
            count = count + A::one();
        }
    }
    count / A::count(d)
}

/// Jaccard distance for binary/set vectors: `1 - |intersection| / |union|`.
///
/// Non-zero values are treated as "element present in set".
///
/// Returns a value in `[0, 1]`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
pub unsafe fn jaccard<T: Element, A: DistAcc<T>>(a: *const T, b: *const T, d: usize) -> A {
    let mut intersection = A::zero();
    let mut union_count = A::zero();

    for k in 0..d {
        let a_nonzero = !A::widen(*a.add(k)).is_zero();
        let b_nonzero = !A::widen(*b.add(k)).is_zero();

        if a_nonzero && b_nonzero {
            intersection = intersection + A::one();
        }
        if a_nonzero || b_nonzero {
            union_count = union_count + A::one();
        }
    }

    if union_count.is_zero() {
        A::zero()
    } else {
        A::one() - intersection / union_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_matches_hand_computation() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let dist: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 1.0).abs() < 1e-6);

        let c = [1.0f32, 1.0, 1.0];
        let dist2: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 3.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn manhattan_sums_absolute_differences() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        let dist: f32 = unsafe { manhattan::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 6.0).abs() < 1e-6);
    }

    #[test]
    fn chebyshev_takes_the_largest_difference() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 5.0, 3.0];
        let dist: f32 = unsafe { chebyshev::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_is_zero_for_parallel_and_one_for_orthogonal() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [2.0f32, 0.0, 0.0];
        let dist: f32 = unsafe { cosine::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        assert!(dist.abs() < 1e-6);

        let c = [0.0f32, 1.0, 0.0];
        let dist2: f32 = unsafe { cosine::<f32, f32>(a.as_ptr(), c.as_ptr(), 3) };
        assert!((dist2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hamming_counts_the_fraction_that_differs() {
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist: f32 = unsafe { hamming::<f32, f32>(a.as_ptr(), b.as_ptr(), 4) };
        // 2 differences out of 4
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn jaccard_compares_non_zero_patterns() {
        // a = [1, 0, 1, 1] -> non-zero at 0, 2, 3; b = [1, 1, 0, 1] -> 0, 1, 3.
        // intersection = {0, 3} = 2, union = {0, 1, 2, 3} = 4, so 1 - 2/4.
        let a = [1.0f32, 0.0, 1.0, 1.0];
        let b = [1.0f32, 1.0, 0.0, 1.0];
        let dist: f32 = unsafe { jaccard::<f32, f32>(a.as_ptr(), b.as_ptr(), 4) };
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn minkowski_at_p2_equals_euclidean() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        let euc: f32 = unsafe { euclidean::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        let mink: f32 = unsafe { minkowski::<f32, f32>(a.as_ptr(), b.as_ptr(), 3, 2.0) };
        assert!((euc - mink).abs() < 1e-5);
    }

    #[test]
    fn minkowski_at_p1_equals_manhattan() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [3.0f32, 4.0, 5.0];
        let man: f32 = unsafe { manhattan::<f32, f32>(a.as_ptr(), b.as_ptr(), 3) };
        let mink: f32 = unsafe { minkowski::<f32, f32>(a.as_ptr(), b.as_ptr(), 3, 1.0) };
        assert!((man - mink).abs() < 1e-5);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_manhattan_accumulates_in_f32() {
        // 1024 then 64 terms of 0.25: an f16 accumulator freezes at 1024 (its
        // spacing there is 1.0), an f32 accumulator reaches 1040.
        let mut a = vec![half::f16::from_f32(0.25); 65];
        a[0] = half::f16::from_f32(1024.0);
        let b = vec![half::f16::ZERO; 65];
        let dist: f32 = unsafe { manhattan::<half::f16, f32>(a.as_ptr(), b.as_ptr(), 65) };
        assert_eq!(dist, 1040.0);
    }
}
