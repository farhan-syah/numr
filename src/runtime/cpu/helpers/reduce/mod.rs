//! Reduction operation helpers for CPU tensors

mod common;
mod multi_dim;
mod precision;
mod single_dim;

pub use precision::reduce_impl_with_precision;

use common::should_fuse_multi_dim_reduction;
use multi_dim::reduce_multi_dim_fused;
use single_dim::reduce_single_dim;

use crate::dispatch_dtype;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, Kernel, ReduceOp, reduce_output_shape};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Reduce implementation with native precision
pub fn reduce_impl(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // Fast path: reduce last dimension when contiguous (uses SIMD kernel)
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::try_empty(&out_shape, dtype, &client.device)?;

        let a_ptr = a.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        // Empty dims = reduce over ALL dimensions → scalar
        let all_dims: Vec<usize> = (0..ndim).collect();
        return reduce_impl(client, op, a, &all_dims, keepdim, op_name);
    } else if should_fuse_multi_dim_reduction(a, dims) {
        reduce_multi_dim_fused(
            client,
            op,
            a,
            dims,
            keepdim,
            AccumulationPrecision::Native,
            op_name,
        )
    } else {
        let a_contig = ensure_contiguous(a)?;

        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim(client, op, &current, dim, keepdim, op_name)?;
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{AccumulationPrecision, ReduceOps};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_fused_multi_dim_sum_matches_expected() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 3, 4], &device).unwrap();

        let out = client.sum(&a, &[1, 2], false).unwrap();
        let got: Vec<f32> = out.to_vec();
        assert_eq!(got, vec![78.0, 222.0]);
    }

    #[test]
    fn test_fused_multi_dim_mean_keepdim_matches_expected() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 3, 4], &device).unwrap();

        let out = client.mean(&a, &[0, 2], true).unwrap();
        assert_eq!(out.shape(), &[1, 3, 1]);
        let got: Vec<f32> = out.to_vec();
        assert_eq!(got, vec![8.5, 12.5, 16.5]);
    }

    /// A sequential F32 sum is NOT reassociated and NOT widened.
    ///
    /// `0.1f32` added 512 times accumulates a specific rounding error:
    /// `51.199790954589844` (bits `0x424ccc96`), against `51.20000076293945`
    /// for an f64-accumulated sum. Pinning the exact bits is what proves the
    /// narrow-float widening left F32 untouched — a widened or reassociated
    /// F32 sum lands on the other value and fails here.
    #[test]
    fn test_f32_fused_multi_dim_sum_is_bit_exact_sequential() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data = vec![0.1f32; 512];
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[512, 1], &device).unwrap();

        let sum: Vec<f32> = client.sum(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(sum[0].to_bits(), 0x424c_cc96, "got {}", sum[0]);

        let mean: Vec<f32> = client.mean(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(mean[0].to_bits(), 0x3dcc_cc96, "got {}", mean[0]);
    }

    /// The training-loss case: 512 BF16 values of `ln(128256) = 11.7618`.
    ///
    /// `bf16(11.7618)` is `11.75`, and a BF16-accumulated sum of 512 of them
    /// stalls at exactly `4096` — BF16's spacing there is `32`, far more than
    /// `2 * 11.75`, so every further addition rounds away. `4096 / 512` is
    /// exactly `8.0` no matter what the inputs were, which is how a freshly
    /// initialized model reported `loss 8.0000` on two different batches.
    ///
    /// F32 accumulation gives `512 * 11.75 = 6016` exactly, and `6016 / 512`
    /// is `11.75` exactly, both representable in BF16.
    #[cfg(feature = "f16")]
    #[test]
    fn test_bf16_fused_multi_dim_mean_does_not_saturate() {
        use half::bf16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data = vec![bf16::from_f32(11.761_783_5); 512];
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[512, 1], &device).unwrap();

        let sum: Vec<bf16> = client.sum(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(sum[0], bf16::from_f32(6016.0), "sum saturated: {}", sum[0]);

        let mean: Vec<bf16> = client.mean(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(
            mean[0],
            bf16::from_f32(11.75),
            "mean saturated: {}",
            mean[0]
        );
    }

    /// The same saturation at the magnitude that masquerades as a healthy loss.
    ///
    /// `bf16(0.7)` is `0.69921875`. A BF16-accumulated sum of 512 of them
    /// stalls at `256`, giving mean `0.5`. The correct sum is `358` exactly
    /// and the correct mean `0.69921875` — which prints as `0.6992`, the value
    /// a run reported after loading real pretrained weights.
    #[cfg(feature = "f16")]
    #[test]
    fn test_bf16_fused_multi_dim_mean_low_magnitude() {
        use half::bf16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data = vec![bf16::from_f32(0.7); 512];
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[512, 1], &device).unwrap();

        let sum: Vec<bf16> = client.sum(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(sum[0], bf16::from_f32(358.0), "sum saturated: {}", sum[0]);

        let mean: Vec<bf16> = client.mean(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(
            mean[0],
            bf16::from_f32(0.699_218_75),
            "mean saturated: {}",
            mean[0]
        );
    }

    /// F16 saturates the same way, just at a different point.
    ///
    /// 4096 values of `1.0`: an F16-accumulated sum stalls at `2048` (spacing
    /// `2` there, against an increment of `1`, and the tie rounds to even), so
    /// the mean comes back `0.5` instead of `1.0`.
    #[cfg(feature = "f16")]
    #[test]
    fn test_f16_fused_multi_dim_mean_does_not_saturate() {
        use half::f16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data = vec![f16::from_f32(1.0); 4096];
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[4096, 1], &device).unwrap();

        let sum: Vec<f16> = client.sum(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(sum[0], f16::from_f32(4096.0), "sum saturated: {}", sum[0]);

        let mean: Vec<f16> = client.mean(&a, &[0, 1], false).unwrap().to_vec();
        assert_eq!(mean[0], f16::from_f32(1.0), "mean saturated: {}", mean[0]);
    }

    /// The non-last-dim single-dim path accumulates widened too.
    ///
    /// Reducing dim 0 of a `[512, 3]` tensor walks a strided loop that never
    /// reaches the SIMD kernel, so it carried the same BF16 accumulator.
    /// The tensor is 3 KiB, under the fused-path threshold, so `dims = [0]`
    /// (a single non-last dim) is what pins this path specifically.
    #[cfg(feature = "f16")]
    #[test]
    fn test_bf16_non_last_dim_sum_does_not_saturate() {
        use half::bf16;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data = vec![bf16::from_f32(11.761_783_5); 512 * 3];
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[512, 3], &device).unwrap();

        let sum: Vec<bf16> = client.sum(&a, &[0], false).unwrap().to_vec();
        assert_eq!(sum.len(), 3);
        for (i, &v) in sum.iter().enumerate() {
            assert_eq!(v, bf16::from_f32(6016.0), "column {i} saturated: {v}");
        }
    }

    #[test]
    fn test_fused_multi_dim_max_and_precision_sum() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 3, 4], &device).unwrap();

        let max_out = client.max(&a, &[0, 1], false).unwrap();
        let max_vals: Vec<f32> = max_out.to_vec();
        assert_eq!(max_vals, vec![21.0, 22.0, 23.0, 24.0]);

        let sum_prec = client
            .sum_with_precision(&a, &[0, 2], false, AccumulationPrecision::FP64)
            .unwrap();
        let sum_vals: Vec<f32> = sum_prec.to_vec();
        assert_eq!(sum_vals, vec![68.0, 100.0, 132.0]);
    }
}
