//! CPU implementation of indexing operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    IndexingOps, ScatterReduceOp,
    reduce::{compute_reduce_strides, reduce_dim_output_shape},
};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{
        bincount_impl, dispatch_dtype, embedding_lookup_impl, ensure_contiguous, gather_2d_impl,
        gather_impl, gather_nd_impl, index_put_impl, index_select_impl, masked_fill_impl,
        masked_select_impl, scatter_impl, scatter_reduce_impl, slice_assign_impl,
    },
    kernels,
};
use crate::tensor::Tensor;

/// IndexingOps implementation for CPU runtime.
impl IndexingOps<CpuRuntime> for CpuClient {
    fn argmax(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // Validate dimension
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &self.device)?;

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::argmax_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut i64,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, "argmax");

        Ok(out)
    }

    fn argmin(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // Validate dimension
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &self.device)?;

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::argmin_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut i64,
                    outer_size,
                    reduce_size,
                    inner_size,
                );
            }
        }, "argmin");

        Ok(out)
    }

    fn gather(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        index: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gather_impl(self, a, dim, index)
    }

    fn scatter(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        index: &Tensor<CpuRuntime>,
        src: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        scatter_impl(self, a, dim, index, src)
    }

    fn index_select(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        index: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        index_select_impl(self, a, dim, index)
    }

    fn index_put(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: usize,
        index: &Tensor<CpuRuntime>,
        src: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        index_put_impl(self, a, dim, index, src)
    }

    fn masked_select(
        &self,
        a: &Tensor<CpuRuntime>,
        mask: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        masked_select_impl(self, a, mask)
    }

    fn masked_fill(
        &self,
        a: &Tensor<CpuRuntime>,
        mask: &Tensor<CpuRuntime>,
        value: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        masked_fill_impl(self, a, mask, value)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<CpuRuntime>,
        indices: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        embedding_lookup_impl(self, embeddings, indices)
    }

    fn scatter_reduce(
        &self,
        dst: &Tensor<CpuRuntime>,
        dim: usize,
        index: &Tensor<CpuRuntime>,
        src: &Tensor<CpuRuntime>,
        op: ScatterReduceOp,
        include_self: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        scatter_reduce_impl(self, dst, dim, index, src, op, include_self)
    }

    fn gather_nd(
        &self,
        input: &Tensor<CpuRuntime>,
        indices: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gather_nd_impl(self, input, indices)
    }

    fn bincount(
        &self,
        input: &Tensor<CpuRuntime>,
        weights: Option<&Tensor<CpuRuntime>>,
        minlength: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        bincount_impl(self, input, weights, minlength)
    }

    fn gather_2d(
        &self,
        input: &Tensor<CpuRuntime>,
        rows: &Tensor<CpuRuntime>,
        cols: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gather_2d_impl(self, input, rows, cols)
    }

    fn slice_assign(
        &self,
        dst: &Tensor<CpuRuntime>,
        src: &Tensor<CpuRuntime>,
        dim: usize,
        start: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        slice_assign_impl(self, dst, src, dim, start)
    }
}
