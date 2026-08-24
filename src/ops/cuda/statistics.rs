//! Statistical operations for CUDA runtime
use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, StatisticalOps, UnaryOps};
use crate::runtime::cuda::ops::reduce_epilogue::sum_then_divide;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

// Import helper functions from statistics module
use crate::runtime::cuda::ops::statistics::{
    histogram_impl, kurtosis_impl, median_impl, mode_impl, percentile_impl, quantile_impl,
    skew_impl,
};

impl StatisticalOps<CudaRuntime> for CudaClient {
    fn var(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Variance implementation using existing ops
        // var(x) = mean((x - mean(x))^2) * N / (N - correction)

        let shape = a.shape();

        // When dims is empty, reduce over all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..shape.len()).collect()
        } else {
            dims.to_vec()
        };

        // Compute count of elements being reduced
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        // Promote ONCE for the whole computation, not just the epilogue.
        // `square` is the reason: on a narrow float a squared difference
        // overflows long before the sum does. A single F16 element 300 away
        // from the mean squares to 90000, past F16's 65504 ceiling, so the
        // element becomes `inf` and poisons the sum — even though the final
        // variance would have been perfectly representable. Everything below
        // therefore runs in the promoted dtype and narrows exactly once.
        let (a_wide, original_dtype) = linalg_promote(self, a)?;
        let a_wide = a_wide.as_ref();

        // Compute mean (mean already handles empty dims internally)
        let mean_val = self.mean(a_wide, dims, true)?;

        // Compute (x - mean)
        let diff = self.sub(a_wide, &mean_val)?;

        // Compute (x - mean)^2
        let diff_squared = self.square(&diff)?;

        // Sum the squared differences and divide by (N - correction). The
        // input is already promoted, so the epilogue takes its non-narrow
        // path and does not promote a second time.
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        let variance = sum_then_divide(self, &diff_squared, &actual_dims, keepdim, divisor)?;
        linalg_demote(self, variance, original_dtype)
    }

    fn std(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    fn quantile(
        &self,
        a: &Tensor<CudaRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<CudaRuntime>> {
        quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<CudaRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<CudaRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<CudaRuntime>, ddof: Option<usize>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CudaRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        kurtosis_impl(self, a, dims, keepdim, correction)
    }

    fn mode(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        mode_impl(self, a, dim, keepdim)
    }
}
