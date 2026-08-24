//! Statistical operations trait for tensor statistics.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Statistical operations trait for tensor statistics
pub trait StatisticalOps<R: Runtime> {
    /// Variance along specified dimensions
    ///
    /// Computes the variance of elements along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    /// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
    ///
    /// # Returns
    ///
    /// Tensor containing variance values
    fn var(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>> {
        let _ = (a, dims, keepdim, correction);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::var",
        })
    }

    /// Standard deviation along specified dimensions
    ///
    /// Computes the standard deviation (sqrt of variance) along the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over
    /// * `keepdim` - If true, reduced dimensions are kept with size 1
    /// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
    ///
    /// # Returns
    ///
    /// Tensor containing standard deviation values
    fn std(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>> {
        let _ = (a, dims, keepdim, correction);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::std",
        })
    }

    /// Compute the q-th quantile along a dimension
    ///
    /// Returns the value at the given quantile (0.0 to 1.0) along the specified dimension.
    /// The input is sorted along the dimension, then the value at the quantile position
    /// is computed using the specified interpolation method.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `q` - Quantile to compute, must be in `` `[0.0, 1.0]` ``
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    /// * `interpolation` - Method for interpolating between data points:
    ///   - "linear" (default): Linear interpolation between adjacent values
    ///   - "lower": Use lower index value
    ///   - "higher": Use higher index value
    ///   - "nearest": Use nearest index value
    ///   - "midpoint": Average of lower and higher values
    ///
    /// # Returns
    ///
    /// Tensor with quantile values. Shape is the same as input with the specified
    /// dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Sort input along dimension
    /// 2. Compute index: idx = q * (n - 1) where n = dimension size
    /// 3. Interpolate based on method:
    ///    - linear: result = `` `sorted[floor]` `` * (1 - frac) + `` `sorted[ceil]` `` * frac
    ///    - lower: result = `` `sorted[floor(idx)]` ``
    ///    - higher: result = `` `sorted[ceil(idx)]` ``
    ///    - nearest: result = `` `sorted[round(idx)]` ``
    ///    - midpoint: result = (`` `sorted[floor]` `` + `` `sorted[ceil]` ``) / 2
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let median = client.quantile(&a, 0.5, Some(0), false, "linear")?;  // 3.0
    /// let q25 = client.quantile(&a, 0.25, Some(0), false, "linear")?;    // 2.0
    /// let q75 = client.quantile(&a, 0.75, Some(0), false, "linear")?;    // 4.0
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if q is outside `[0.0, 1.0]`
    /// - `InvalidArgument` if interpolation is not a valid method
    /// - `InvalidAxis` if dim is out of bounds
    fn quantile(
        &self,
        a: &Tensor<R>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<R>> {
        let _ = (a, q, dim, keepdim, interpolation);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::quantile",
        })
    }

    /// Compute the p-th percentile along a dimension
    ///
    /// Convenience wrapper for `quantile(a, p/100, dim, keepdim, "linear")`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `p` - Percentile to compute, must be in `[0.0, 100.0]`
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let p50 = client.percentile(&a, 50.0, Some(0), false)?;  // 3.0 (median)
    /// let p25 = client.percentile(&a, 25.0, Some(0), false)?;  // 2.0
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn percentile(
        &self,
        a: &Tensor<R>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<R>> {
        let _ = (a, p, dim, keepdim);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::percentile",
        })
    }

    /// Compute median (50th percentile) along a dimension
    ///
    /// Returns the median value along the specified dimension.
    /// Equivalent to `quantile(a, 0.5, dim, keepdim, "linear")`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension to reduce (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 3.0, 2.0, 5.0, 4.0], &[5], &device)?;
    /// let med = client.median(&a, Some(0), false)?;  // 3.0
    ///
    /// let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)?;
    /// let med = client.median(&b, Some(0), false)?;  // 2.5 (interpolated)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn median(&self, a: &Tensor<R>, dim: Option<isize>, keepdim: bool) -> Result<Tensor<R>> {
        let _ = (a, dim, keepdim);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::median",
        })
    }

    /// Compute histogram of input values
    ///
    /// Counts the number of values that fall into each of the specified bins.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (flattened internally)
    /// * `bins` - Number of equal-width bins
    /// * `range` - Optional (min, max) range for bins. Defaults to (a.min(), a.max())
    ///
    /// # Returns
    ///
    /// Tuple of (histogram, bin_edges):
    /// - histogram: I64 tensor of shape `[bins]` with counts
    /// - bin_edges: Tensor of shape `[bins + 1]` with bin boundaries
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Determine range: `[min, max]` (from input or provided)
    /// 2. Compute bin_width = (max - min) / bins
    /// 3. For each value x:
    ///    bin_idx = floor((x - min) / bin_width)
    ///    Clamp to `[0, bins-1]`
    ///    counts`[bin_idx]`++
    /// 4. bin_edges = `[min, min+w, min+2w, ..., max]`
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, 1.5, 2.5, 1.0, 2.0], &[5], &device)?;
    /// let (hist, edges) = client.histogram(&a, 3, None)?;
    /// // hist = [1, 2, 2] for bins [0.5-1.17), [1.17-1.83), [1.83-2.5]
    /// // edges = [0.5, 1.17, 1.83, 2.5]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn histogram(
        &self,
        a: &Tensor<R>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (a, bins, range);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::histogram",
        })
    }

    /// Covariance matrix of observations
    ///
    /// Computes the covariance matrix from a matrix where each row is an observation
    /// and each column is a variable (feature).
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    /// * `ddof` - Delta degrees of freedom. Default: 1 (sample covariance)
    ///
    /// # Returns
    ///
    /// Covariance matrix `[n_features, n_features]`
    ///
    /// # Properties
    ///
    /// - Symmetric: cov`[i,j]` = cov`[j,i]`
    /// - Diagonal elements are variances: `cov[i,i]` = `var(X[:,i])`
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// // 3 samples, 2 features
    /// let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)?;
    /// let c = client.cov(&x, None)?;  // [2, 2] covariance matrix
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn cov(&self, a: &Tensor<R>, ddof: Option<usize>) -> Result<Tensor<R>> {
        let _ = (a, ddof);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::cov",
        })
    }

    /// Pearson correlation coefficient matrix
    ///
    /// Computes the correlation coefficient matrix from observations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    ///
    /// # Returns
    ///
    /// Correlation matrix `[n_features, n_features]` with values in `[-1, 1]`
    ///
    /// # Properties
    ///
    /// - Diagonal elements are 1.0 (unless feature has zero variance)
    /// - Off-diagonal in `[-1, 1]`: correlation coefficient
    /// - Symmetric: corr`[i,j]` = corr`[j,i]`
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)?;
    /// let corr = client.corrcoef(&x)?;  // [2, 2] correlation matrix
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn corrcoef(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "StatisticalOps::corrcoef",
        })
    }

    /// Skewness (third standardized moment)
    ///
    /// Measures the asymmetry of the distribution. Positive skew indicates a tail
    /// extending toward positive values; negative skew indicates a tail toward
    /// negative values.
    ///
    /// # Formula
    ///
    /// ```text
    /// skew = E[(X - mean)³] / std³
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over (empty = all dimensions)
    /// * `keepdim` - If true, keep reduced dimensions as size 1
    /// * `correction` - Degrees of freedom correction (default 0)
    ///
    /// # Returns
    ///
    /// Tensor containing skewness values
    ///
    /// # Interpretation
    ///
    /// - skew ≈ 0: Symmetric distribution
    /// - skew > 0: Right-skewed (tail toward positive)
    /// - skew < 0: Left-skewed (tail toward negative)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// // Symmetric distribution
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device)?;
    /// let s = client.skew(&a, &[], false, 0)?;  // ≈ 0.0
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn skew(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>> {
        let _ = (a, dims, keepdim, correction);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::skew",
        })
    }

    /// Kurtosis (fourth standardized moment, excess)
    ///
    /// Measures the "tailedness" of the distribution. Higher kurtosis indicates
    /// heavier tails and sharper peak; lower kurtosis indicates lighter tails.
    ///
    /// Returns excess kurtosis (kurtosis - 3), so a normal distribution has
    /// kurtosis of 0.
    ///
    /// # Formula
    ///
    /// ```text
    /// kurtosis = E[(X - mean)⁴] / std⁴ - 3
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dims` - Dimensions to reduce over (empty = all dimensions)
    /// * `keepdim` - If true, keep reduced dimensions as size 1
    /// * `correction` - Degrees of freedom correction (default 0)
    ///
    /// # Returns
    ///
    /// Tensor containing excess kurtosis values
    ///
    /// # Interpretation
    ///
    /// - kurtosis ≈ 0: Normal-like tails (mesokurtic)
    /// - kurtosis > 0: Heavy tails, sharp peak (leptokurtic)
    /// - kurtosis < 0: Light tails, flat peak (platykurtic)
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::{StatisticalOps, RandomOps};
    /// use numr::dtype::DType;
    ///
    /// // Normal distribution has excess kurtosis ≈ 0
    /// let normal_data = client.randn(&[10000], DType::F32)?;
    /// let k = client.kurtosis(&normal_data, &[], false, 0)?;  // ≈ 0.0
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn kurtosis(
        &self,
        a: &Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<R>> {
        let _ = (a, dims, keepdim, correction);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::kurtosis",
        })
    }

    /// Compute the mode (most frequent value) along a dimension
    ///
    /// Returns the most frequently occurring value along the specified dimension,
    /// along with the count of how many times it appears. If multiple values have
    /// the same maximum frequency, the smallest value is returned.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension to reduce along (None = flatten first)
    /// * `keepdim` - If true, keep reduced dimension as size 1
    ///
    /// # Returns
    ///
    /// Tuple of (mode_values, mode_counts):
    /// - mode_values: Tensor containing the most frequent values
    /// - mode_counts: I64 tensor containing the count of each mode
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Sort the values along the specified dimension
    /// 2. Count consecutive equal values (run-length encoding)
    /// 3. Return the value with the highest count
    ///    (if tied, return the smallest value)
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::StatisticalOps;
    ///
    /// // Simple 1D mode
    /// let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 2.0, 3.0, 2.0], &[5], &device)?;
    /// let (values, counts) = client.mode(&a, Some(0), false)?;
    /// // values = [2.0], counts = [3]
    ///
    /// // 2D mode along axis 1
    /// let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 2.0, 3.0, 3.0, 3.0], &[2, 3], &device)?;
    /// let (values, counts) = client.mode(&b, Some(1), false)?;
    /// // values = [1.0, 3.0], counts = [2, 3]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - For continuous data, mode may be less meaningful than for discrete data
    /// - Floating point comparison uses exact equality
    /// - If all values are unique, returns the smallest value with count 1
    fn mode(
        &self,
        a: &Tensor<R>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (a, dim, keepdim);
        Err(Error::NotImplemented {
            feature: "StatisticalOps::mode",
        })
    }
}
