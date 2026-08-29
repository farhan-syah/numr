//! CPU implementation of statistical operations.

use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{
    StatisticalOps, UnaryOps,
    reduce::{compute_reduce_strides, reduce_dim_output_shape, reduce_output_shape},
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
        let shape = a.shape().to_vec();
        let ndim = shape.len();

        // An empty `dims` means "reduce over every dimension". Normalizing it
        // to the explicit dim list here is what makes `dims = []` and
        // `dims = [0, .., ndim-1]` the same computation rather than two
        // implementations free to drift apart.
        let mut reduce_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            let mut d = dims.to_vec();
            d.sort_unstable();
            d.dedup();
            d
        };

        for &dim in &reduce_dims {
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }
        }

        // A 0-dim tensor holds one element, which deviates from its own mean by
        // zero. There is no axis for the reduction paths below to work with.
        if ndim == 0 {
            let out = Tensor::<CpuRuntime>::empty(&[], dtype, &self.device)?;
            let out_ptr = out.ptr();
            dispatch_dtype!(dtype, T => {
                unsafe {
                    *(out_ptr as *mut T) = T::from_f64(0.0);
                }
            }, "var");
            return Ok(out);
        }

        if reduce_dims.len() == 1 {
            return var_single_dim(self, a, reduce_dims[0], keepdim, correction);
        }

        // Multi-dimension case. Variance does NOT decompose into a chain of
        // per-dimension variances: the variance of the per-row variances is a
        // different quantity from the variance of the whole block. The reduced
        // set must be a single reduction against a single mean, with
        // `correction` applied once against the total reduced count.
        //
        // Permuting the reduced dims to the end and flattening them into one
        // axis makes this literally the single-dimension case, so the existing
        // `variance_kernel` computes it unchanged.
        let kept: Vec<usize> = (0..ndim).filter(|d| !reduce_dims.contains(d)).collect();
        let mut perm = kept.clone();
        perm.append(&mut reduce_dims);
        let reduce_dims = &perm[kept.len()..];

        let kept_count: usize = kept.iter().map(|&d| shape[d]).product();
        let reduced_count: usize = reduce_dims.iter().map(|&d| shape[d]).product();

        // `permute` is a view, so a non-contiguous input costs exactly one
        // materialization here and none in the kernel.
        let permuted = ensure_contiguous(&a.permute(&perm)?)?;
        let flat = permuted.reshape(&[kept_count, reduced_count])?;

        let flat_var = var_single_dim(self, &flat, 1, false, correction)?;
        let out_shape = reduce_output_shape(&shape, reduce_dims, keepdim);
        flat_var.reshape(&out_shape)
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

/// Variance over a single dimension of `a`.
///
/// Computes one mean over `dim`, then the mean of the squared deviations from
/// that mean, with the divisor `size(dim) - correction`. The multi-dimension
/// path in `var` reduces to this after flattening the reduced dims.
fn var_single_dim(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
    let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

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

    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::ops::StatisticalOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    /// Variance over several dims is the variance of the whole reduced block,
    /// not the variance of the per-row variances. Chaining single-dim variance
    /// calls computed the latter: `[[1, 2], [3, 4]]` reduced over dim 1 gives
    /// `[0.25, 0.25]`, whose variance over dim 0 is `0`. The correct answer is
    /// the deviation from the single mean 2.5, which is `5 / 4 = 1.25`.
    #[test]
    fn test_multi_dim_var_uses_one_mean() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let got: Vec<f64> = client.var(&a, &[0, 1], false, 0).unwrap().to_vec();
        assert_eq!(got, vec![1.25]);
    }

    /// `dims = []` means "reduce over everything", so it must agree exactly
    /// with naming every dimension. The two took different code paths, and
    /// `correction` applies once against the total reduced count in both.
    #[test]
    fn test_empty_dims_matches_all_dims() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f64> = (1..=24).map(|v| (v as f64) * 0.5).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device).unwrap();

        for correction in [0usize, 1] {
            for keepdim in [false, true] {
                let via_empty: Vec<f64> =
                    client.var(&a, &[], keepdim, correction).unwrap().to_vec();
                let via_all: Vec<f64> = client
                    .var(&a, &[0, 1, 2], keepdim, correction)
                    .unwrap()
                    .to_vec();
                assert_eq!(
                    via_empty, via_all,
                    "dims=[] vs dims=[0,1,2] diverged (correction {correction}, keepdim {keepdim})"
                );
            }
        }
    }

    /// A transposed view must give the same variance as the materialized
    /// tensor it is a view of: the permute-and-flatten path has to respect
    /// strides, not read the storage in its raw order.
    #[test]
    fn test_non_contiguous_input_matches_contiguous() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[3, 4], &device).unwrap();

        let view = a.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());
        let materialized = view.contiguous().unwrap();

        for dims in [vec![0usize, 1], vec![0], vec![1]] {
            let from_view: Vec<f64> = client.var(&view, &dims, false, 1).unwrap().to_vec();
            let from_contig: Vec<f64> =
                client.var(&materialized, &dims, false, 1).unwrap().to_vec();
            assert_eq!(from_view, from_contig, "dims {dims:?} diverged");
        }
    }

    /// Partial multi-dim reduction: the kept dims must survive in order, and
    /// each output bucket is the variance of its own reduced block.
    #[test]
    fn test_partial_multi_dim_var_shape_and_values() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        // Two blocks of six: 1..6 and 7..12. Both have population variance
        // 35/12 about their own mean.
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 2], &device).unwrap();

        let out = client.var(&a, &[1, 2], false, 0).unwrap();
        assert_eq!(out.shape(), &[2]);
        let got: Vec<f64> = out.to_vec();
        let expected = 35.0 / 12.0;
        for (i, v) in got.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-12,
                "bucket {i}: got {v}, expected {expected}"
            );
        }

        let kept = client.var(&a, &[1, 2], true, 0).unwrap();
        assert_eq!(kept.shape(), &[2, 1, 1]);
    }
}
