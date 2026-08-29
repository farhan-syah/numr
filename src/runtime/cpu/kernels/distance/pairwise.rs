//! Pairwise distance kernels (cdist, pdist) and condensed-form conversion.

use super::acc::DistAcc;
use super::metrics;
use crate::dtype::{DType, Element};
use crate::ops::DistanceMetric;
use num_traits::{Float, FromPrimitive, Zero};

/// Compute pairwise distances between two point sets (cdist).
///
/// # Safety
///
/// - `x` must point to valid data of length `n * d`
/// - `y` must point to valid data of length `m * d`
/// - `out` must point to valid memory of length `n * m`
/// - All pointers must be properly aligned for type T
#[inline]
pub unsafe fn cdist_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    if T::DTYPE == DType::F64 {
        cdist_acc::<T, f64>(x, y, out, n, m, d, metric);
    } else {
        cdist_acc::<T, f32>(x, y, out, n, m, d, metric);
    }
}

/// Compute pairwise distances within a single point set (pdist, condensed form).
///
/// # Safety
///
/// - `x` must point to valid data of length `n * d`
/// - `out` must point to valid memory of length `n * (n - 1) / 2`
/// - All pointers must be properly aligned for type T
#[inline]
pub unsafe fn pdist_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    metric: DistanceMetric,
) {
    if T::DTYPE == DType::F64 {
        pdist_acc::<T, f64>(x, out, n, d, metric);
    } else {
        pdist_acc::<T, f32>(x, out, n, d, metric);
    }
}

/// One pair of rows, reduced by `metric` in accumulator `A`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
unsafe fn distance<T: Element, A: DistAcc<T>>(
    a: *const T,
    b: *const T,
    d: usize,
    metric: DistanceMetric,
) -> A {
    match metric {
        DistanceMetric::Euclidean => metrics::euclidean::<T, A>(a, b, d),
        DistanceMetric::SquaredEuclidean => metrics::sqeuclidean::<T, A>(a, b, d),
        DistanceMetric::Manhattan => metrics::manhattan::<T, A>(a, b, d),
        DistanceMetric::Chebyshev => metrics::chebyshev::<T, A>(a, b, d),
        DistanceMetric::Minkowski(p) => metrics::minkowski::<T, A>(a, b, d, A::exponent(p)),
        DistanceMetric::Cosine => metrics::cosine::<T, A>(a, b, d),
        DistanceMetric::Correlation => metrics::correlation::<T, A>(a, b, d),
        DistanceMetric::Hamming => metrics::hamming::<T, A>(a, b, d),
        DistanceMetric::Jaccard => metrics::jaccard::<T, A>(a, b, d),
    }
}

/// cdist with an explicit accumulator.
///
/// # Safety
/// Same as [`cdist_kernel`].
#[inline]
unsafe fn cdist_acc<T: Element, A: DistAcc<T>>(
    x: *const T,
    y: *const T,
    out: *mut T,
    n: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    for i in 0..n {
        for j in 0..m {
            let dist = distance::<T, A>(x.add(i * d), y.add(j * d), d, metric);
            *out.add(i * m + j) = dist.narrow();
        }
    }
}

/// pdist with an explicit accumulator.
///
/// # Safety
/// Same as [`pdist_kernel`].
#[inline]
unsafe fn pdist_acc<T: Element, A: DistAcc<T>>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    metric: DistanceMetric,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = distance::<T, A>(x.add(i * d), x.add(j * d), d, metric);
            *out.add(k) = dist.narrow();
            k += 1;
        }
    }
}

/// Convert condensed distance vector to square matrix.
///
/// This moves already-computed distances, so there is no running total and no
/// accumulator: the element type carries the values unchanged.
///
/// # Safety
///
/// - `condensed` must point to valid data of length `n * (n - 1) / 2`
/// - `square` must point to valid memory of length `n * n`
#[inline]
pub unsafe fn squareform_kernel<T: Element + Float>(condensed: *const T, square: *mut T, n: usize) {
    // Fill diagonal with zeros
    for i in 0..n {
        *square.add(i * n + i) = <T as Zero>::zero();
    }

    // Fill upper and lower triangles
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let val = *condensed.add(k);
            *square.add(i * n + j) = val;
            *square.add(j * n + i) = val;
            k += 1;
        }
    }
}

/// Convert square distance matrix to condensed form.
///
/// # Safety
///
/// - `square` must point to valid data of length `n * n`
/// - `condensed` must point to valid memory of length `n * (n - 1) / 2`
#[inline]
pub unsafe fn squareform_inverse_kernel<T: Element + Float>(
    square: *const T,
    condensed: *mut T,
    n: usize,
) {
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            *condensed.add(k) = *square.add(i * n + j);
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdist_euclidean_over_two_point_sets() {
        // X = [[0, 0], [1, 1]], Y = [[1, 0], [2, 2]]
        let x = [0.0f32, 0.0, 1.0, 1.0];
        let y = [1.0f32, 0.0, 2.0, 2.0];
        let mut out = [0.0f32; 4];

        unsafe {
            cdist_kernel(
                x.as_ptr(),
                y.as_ptr(),
                out.as_mut_ptr(),
                2,
                2,
                2,
                DistanceMetric::Euclidean,
            );
        }

        // d(x0, y0) = sqrt(1) = 1
        // d(x0, y1) = sqrt(4+4) = 2*sqrt(2)
        // d(x1, y0) = sqrt(0+1) = 1
        // d(x1, y1) = sqrt(1+1) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (8.0f32).sqrt()).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn pdist_euclidean_over_one_point_set() {
        // X = [[0, 0], [1, 0], [0, 1]] - 3 points in 2D
        let x = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 3]; // 3 = n*(n-1)/2 for n=3

        unsafe {
            pdist_kernel(
                x.as_ptr(),
                out.as_mut_ptr(),
                3,
                2,
                DistanceMetric::Euclidean,
            );
        }

        // d(0,1) = 1, d(0,2) = 1, d(1,2) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn squareform_expands_the_condensed_vector() {
        let condensed = [1.0f32, 2.0, 3.0]; // d(0,1), d(0,2), d(1,2)
        let mut square = [0.0f32; 9];

        unsafe {
            squareform_kernel(condensed.as_ptr(), square.as_mut_ptr(), 3);
        }

        // Expected:
        // [[0, 1, 2],
        //  [1, 0, 3],
        //  [2, 3, 0]]
        assert_eq!(square, [0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn squareform_inverse_recovers_the_condensed_vector() {
        let square = [0.0f32, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let mut condensed = [0.0f32; 3];

        unsafe {
            squareform_inverse_kernel(square.as_ptr(), condensed.as_mut_ptr(), 3);
        }

        assert_eq!(condensed, [1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_sqeuclidean_accumulates_in_f32() {
        // Row 0 is [32.0, 0.5 x 256], row 1 is all zeros. The squared terms are
        // 1024 followed by 256 terms of 0.25. f16 steps by 1.0 across
        // [1024, 2048), so each 0.25 rounds straight back and an f16
        // accumulator freezes at 1024; f32 reaches 1024 + 64 = 1088, which f16
        // then represents exactly.
        let d = 257;
        let mut x = vec![half::f16::from_f32(0.5); 2 * d];
        x[0] = half::f16::from_f32(32.0);
        for slot in x.iter_mut().skip(d) {
            *slot = half::f16::ZERO;
        }
        let mut out = [half::f16::ZERO; 1];

        unsafe {
            pdist_kernel(
                x.as_ptr(),
                out.as_mut_ptr(),
                2,
                d,
                DistanceMetric::SquaredEuclidean,
            );
        }

        assert_eq!(out[0].to_f32(), 1088.0);
    }
}
