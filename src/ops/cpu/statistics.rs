//! CPU implementation of statistical operations.

use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{
    StatisticalOps, UnaryOps,
    reduce::{compute_reduce_strides, reduce_dim_output_shape},
};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::tensor::Tensor;

/// StatisticalOps implementation for CPU runtime.
impl StatisticalOps<CpuRuntime> for CpuClient {
    fn var(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // For multi-dimensional reduction, we need to handle the dims properly
        // For simplicity, we support single dimension reduction here
        // Multi-dim can be decomposed into multiple single-dim reductions
        if dims.is_empty() {
            // Reduce over all dimensions - return scalar variance
            let numel = a.numel();
            let a_contig = ensure_contiguous(a)?;
            let a_ptr = a_contig.ptr();

            let variance = dispatch_dtype!(dtype, T => {
                unsafe {
                    let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                    // Compute mean
                    let mut sum = 0.0f64;
                    for &val in slice {
                        sum += val.to_f64();
                    }
                    let mean = sum / (numel as f64);

                    // Compute variance
                    let mut var_sum = 0.0f64;
                    for &val in slice {
                        let diff = val.to_f64() - mean;
                        var_sum += diff * diff;
                    }
                    let divisor = (numel.saturating_sub(correction)).max(1) as f64;
                    var_sum / divisor
                }
            }, "var");

            let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
            let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &self.device)?;
            let out_ptr = out.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    *(out_ptr as *mut T) = T::from_f64(variance);
                }
            }, "var");

            return Ok(out);
        }

        // Single dimension case
        if dims.len() == 1 {
            let dim = dims[0];
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }

            let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
            let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

            let a_contig = ensure_contiguous(a)?;
            let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &self.device)?;

            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::variance_kernel::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        reduce_size,
                        inner_size,
                        correction,
                    );
                }
            }, "var");

            return Ok(out);
        }

        // Multi-dimension case: compute variance iteratively
        // First, compute along the last dimension, then continue
        let mut result = a.clone();
        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort descending so we reduce from last to first

        for dim in sorted_dims {
            result = self.var(&result, &[dim], true, correction)?;
        }

        // Remove keepdim dimensions if not requested
        if !keepdim {
            let final_shape: Vec<usize> = result
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| !dims.contains(i))
                .map(|(_, &s)| s)
                .collect();
            result = result.reshape(&final_shape)?;
        }

        Ok(result)
    }

    fn std(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    fn quantile(
        &self,
        a: &Tensor<CpuRuntime>,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::statistics::quantile_impl(self, a, q, dim, keepdim, interpolation)
    }

    fn percentile(
        &self,
        a: &Tensor<CpuRuntime>,
        p: f64,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::statistics::percentile_impl(self, a, p, dim, keepdim)
    }

    fn median(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::statistics::median_impl(self, a, dim, keepdim)
    }

    fn histogram(
        &self,
        a: &Tensor<CpuRuntime>,
        bins: usize,
        range: Option<(f64, f64)>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        crate::runtime::cpu::statistics::histogram_impl(self, a, bins, range)
    }

    fn cov(&self, a: &Tensor<CpuRuntime>, ddof: Option<usize>) -> Result<Tensor<CpuRuntime>> {
        // Delegate to LinalgAlgorithms implementation
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CpuRuntime>>::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        // Delegate to LinalgAlgorithms implementation
        use crate::algorithm::LinearAlgebraAlgorithms;
        <Self as LinearAlgebraAlgorithms<CpuRuntime>>::corrcoef(self, a)
    }

    fn skew(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::statistics::skew_impl(self, a, dims, keepdim, correction)
    }

    fn kurtosis(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::statistics::kurtosis_impl(self, a, dims, keepdim, correction)
    }

    fn mode(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        crate::runtime::cpu::statistics::mode_impl(self, a, dim, keepdim)
    }
}
