//! Reduce operations for CUDA runtime
use crate::error::Result;
use crate::ops::ReduceOps;
use crate::runtime::cuda::kernels::AccumulationPrecision;
use crate::runtime::cuda::ops::helpers::native_reduce_op;
use crate::runtime::cuda::ops::reduce_epilogue::sum_then_divide;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Normalize dims for reduction: empty means all dimensions
#[inline]
fn normalize_reduce_dims(dims: &[usize], ndim: usize) -> Vec<usize> {
    if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    }
}

impl ReduceOps<CudaRuntime> for CudaClient {
    fn sum(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "sum", &dims, keepdim, None)
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, Some(precision))
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| a.shape()[d]).product()
        };

        let dims = normalize_reduce_dims(dims, a.shape().len());
        // Narrow floats sum and divide in F32, then narrow once — a F16 sum
        // above 65504 saturates to infinity, and a F16 divisor above 65504
        // does too. F32/F64/integers take the direct path unchanged.
        sum_then_divide(self, a, &dims, keepdim, count as f64)
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "max", &dims, keepdim, None)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, Some(precision))
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "min", &dims, keepdim, None)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, Some(precision))
    }

    fn prod(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "prod", &dims, keepdim, None)
    }

    fn prod_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "prod", dims, keepdim, Some(precision))
    }

    fn any(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "any", &dims, keepdim, None)
    }

    fn all(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dims = normalize_reduce_dims(dims, a.shape().len());
        native_reduce_op(self, a, "all", &dims, keepdim, None)
    }
}
