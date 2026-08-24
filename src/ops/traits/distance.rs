//! Distance computation operations.
//!
//! This module defines the `DistanceOps` trait for computing pairwise distances
//! between points using various distance metrics.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Distance metric for pairwise distance computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance: sqrt(sum((x - y)^2))
    Euclidean,
    /// Squared Euclidean distance: sum((x - y)^2)
    /// More efficient when sqrt is not needed (e.g., for comparisons)
    SquaredEuclidean,
    /// Manhattan (L1/cityblock) distance: sum(|x - y|)
    Manhattan,
    /// Chebyshev (L-infinity) distance: max(|x - y|)
    Chebyshev,
    /// Minkowski (Lp) distance: (sum(|x - y|^p))^(1/p)
    Minkowski(f64),
    /// Cosine distance: 1 - (x · y) / (||x|| ||y||)
    Cosine,
    /// Correlation distance: 1 - Pearson correlation coefficient
    Correlation,
    /// Hamming distance: fraction of differing elements
    Hamming,
    /// Jaccard distance: 1 - intersection/union (for binary vectors)
    Jaccard,
}

impl DistanceMetric {
    /// Returns the name of the metric for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::SquaredEuclidean => "sqeuclidean",
            DistanceMetric::Manhattan => "manhattan",
            DistanceMetric::Chebyshev => "chebyshev",
            DistanceMetric::Minkowski(_) => "minkowski",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Correlation => "correlation",
            DistanceMetric::Hamming => "hamming",
            DistanceMetric::Jaccard => "jaccard",
        }
    }
}

/// Distance computation operations.
///
/// Provides efficient computation of pairwise distances between point sets.
/// These operations are fundamental for spatial algorithms, clustering,
/// and nearest neighbor search.
///
/// # Backend Support
///
/// ## Data Types
///
/// - **CPU**: Supports F32, F64, F16, BF16 (with `f16` feature)
/// - **CUDA**: Supports F32, F64, F16, BF16 (with `f16` feature)
/// - **WebGPU**: Currently supports F32 only
///
/// All backends require floating-point dtypes. Integer dtypes are not supported.
pub trait DistanceOps<R: Runtime> {
    /// Compute pairwise distances between two point sets.
    ///
    /// Given two sets of points X and Y, computes the distance between
    /// every pair (x_i, y_j) and returns a distance matrix.
    ///
    /// # Arguments
    ///
    /// * `x` - First point set with shape `(n, d)` where `n` is the number of points
    ///   and `d` is the dimensionality
    /// * `y` - Second point set with shape `(m, d)`
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    ///
    /// Distance matrix with shape `(n, m)` where element `(i, j)` is the distance
    /// between `x[i]` and `y[j]`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - Inputs are not 2D tensors
    /// - Dimensionality doesn't match (`x.shape[1]` != `y.shape[1]`)
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::DistanceMetric;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::DistanceOps;
    ///
    /// // Points in 3D space
    /// let x = Tensor::<CpuRuntime>::try_from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3], &device)?;
    /// let y = Tensor::<CpuRuntime>::try_from_slice(&[1.0, 0.0, 0.0, 2.0, 2.0, 2.0], &[2, 3], &device)?;
    ///
    /// // Euclidean distances
    /// let d = client.cdist(&x, &y, DistanceMetric::Euclidean)?;
    /// // d has shape (2, 2), `d[i,j]` = ||x[i] - y[j]||
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn cdist(&self, x: &Tensor<R>, y: &Tensor<R>, metric: DistanceMetric) -> Result<Tensor<R>> {
        let _ = (x, y, metric);
        Err(Error::NotImplemented {
            feature: "DistanceOps::cdist",
        })
    }

    /// Compute pairwise distances within a single point set (condensed form).
    ///
    /// Computes distances between all pairs of points in X and returns
    /// the upper triangle in condensed (1D) form. This is more memory
    /// efficient than the full distance matrix for symmetric distance
    /// computation.
    ///
    /// # Arguments
    ///
    /// * `x` - Point set with shape (n, d)
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    ///
    /// Condensed distance vector with shape `(n*(n-1)/2,)` containing the upper
    /// triangle of the distance matrix in row-major order.
    ///
    /// For `n` points, the condensed form stores distances as:
    /// `[d(0,1), d(0,2), ..., d(0,n-1), d(1,2), ..., d(n-2,n-1)]`
    ///
    /// # Index Conversion
    ///
    /// To convert from condensed index `k` to matrix indices `(i, j)` where `i < j`:
    /// - `i = n - 2 - floor(sqrt(-8*k + 4*n*(n-1) - 7) / 2 - 0.5)`
    /// - `j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2`
    ///
    /// To convert from `(i, j)` to condensed index `k`:
    /// - `k = n*i - i*(i+1)/2 + j - i - 1`
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - Input is not a 2D tensor
    /// - Input has fewer than 2 points
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::DistanceMetric;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::DistanceOps;
    ///
    /// let x = Tensor::<CpuRuntime>::try_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device)?;
    ///
    /// // Condensed distances: [d(0,1), d(0,2), d(1,2)]
    /// let d = client.pdist(&x, DistanceMetric::Euclidean)?;
    /// // d has shape (3,) = n*(n-1)/2 for n=3
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn pdist(&self, x: &Tensor<R>, metric: DistanceMetric) -> Result<Tensor<R>> {
        let _ = (x, metric);
        Err(Error::NotImplemented {
            feature: "DistanceOps::pdist",
        })
    }

    /// Convert condensed distance vector to square distance matrix.
    ///
    /// Takes a condensed distance vector (from `pdist`) and expands it to
    /// a full symmetric distance matrix with zeros on the diagonal.
    ///
    /// # Arguments
    ///
    /// * `condensed` - Condensed distance vector with shape (n*(n-1)/2,)
    /// * `n` - Number of original points
    ///
    /// # Returns
    ///
    /// Square distance matrix with shape (n, n) where:
    /// - Diagonal elements are 0
    /// - Matrix is symmetric (`d[i,j]` == `d[j,i]`)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `condensed` is not 1D
    /// - Length doesn't match n*(n-1)/2
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::DistanceMetric;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::DistanceOps;
    ///
    /// # let x = Tensor::<CpuRuntime>::try_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device)?;
    /// let condensed = client.pdist(&x, DistanceMetric::Euclidean)?;
    /// let square = client.squareform(&condensed, 3)?;
    /// // square has shape (3, 3), symmetric with zero diagonal
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn squareform(&self, condensed: &Tensor<R>, n: usize) -> Result<Tensor<R>> {
        let _ = (condensed, n);
        Err(Error::NotImplemented {
            feature: "DistanceOps::squareform",
        })
    }

    /// Convert square distance matrix to condensed form.
    ///
    /// Takes a square symmetric distance matrix and extracts the upper
    /// triangle in condensed (1D) form.
    ///
    /// # Arguments
    ///
    /// * `square` - Square distance matrix with shape (n, n)
    ///
    /// # Returns
    ///
    /// Condensed distance vector with shape (n*(n-1)/2,)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `square` is not 2D
    /// - `square` is not square (`shape[0]` != `shape[1]`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::DistanceMetric;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::DistanceOps;
    ///
    /// # let x = Tensor::<CpuRuntime>::try_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device)?;
    /// let square = client.cdist(&x, &x, DistanceMetric::Euclidean)?;
    /// let condensed = client.squareform_inverse(&square)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn squareform_inverse(&self, square: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = square;
        Err(Error::NotImplemented {
            feature: "DistanceOps::squareform_inverse",
        })
    }
}
