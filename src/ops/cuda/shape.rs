//! Shape operations for CUDA runtime
use crate::error::Result;
use crate::ops::ShapeOps;
use crate::ops::impl_generic::{repeat_interleave_impl, unfold_impl};
use crate::runtime::cuda::kernels::{launch_cat_copy, launch_pad, launch_repeat, launch_roll};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{common::shape_ops, ensure_contiguous};
use crate::tensor::Tensor;

impl ShapeOps<CudaRuntime> for CudaClient {
    fn cat(&self, tensors: &[&Tensor<CudaRuntime>], dim: isize) -> Result<Tensor<CudaRuntime>> {
        let params = crate::runtime::common::shape_ops::validate_cat(tensors, dim)?;

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, params.dtype, &self.device)?;

        // Copy data from each tensor using CUDA kernel
        let mut cat_offset = 0usize;
        for &tensor in tensors {
            let tensor_contig = ensure_contiguous(tensor)?;
            let src_cat_size = tensor.shape()[params.dim_idx];

            unsafe {
                launch_cat_copy(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    params.dtype,
                    tensor_contig.ptr(),
                    out.ptr(),
                    params.outer_size,
                    src_cat_size,
                    params.cat_dim_total,
                    cat_offset,
                    params.inner_size,
                )?;
            }

            cat_offset += src_cat_size;
        }

        Ok(out)
    }

    fn stack(&self, tensors: &[&Tensor<CudaRuntime>], dim: isize) -> Result<Tensor<CudaRuntime>> {
        // Validate tensors and get normalized dimension
        let _ = crate::runtime::common::shape_ops::validate_stack(tensors, dim)?;

        // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
        let unsqueezed: Vec<Tensor<CudaRuntime>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<_>>()?;

        let refs: Vec<&Tensor<CudaRuntime>> = unsqueezed.iter().collect();
        self.cat(&refs, dim)
    }

    fn split(
        &self,
        tensor: &Tensor<CudaRuntime>,
        split_size: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        shape_ops::split_impl(tensor, split_size, dim)
    }

    fn chunk(
        &self,
        tensor: &Tensor<CudaRuntime>,
        chunks: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CudaRuntime>>> {
        shape_ops::chunk_impl(tensor, chunks, dim)
    }

    fn repeat(
        &self,
        tensor: &Tensor<CudaRuntime>,
        repeats: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_repeat(tensor, repeats)?;

        // Handle no-op case (all repeats are 1)
        if repeats.iter().all(|&r| r == 1) {
            return tensor.contiguous();
        }

        let tensor_contig = ensure_contiguous(tensor)?;
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, tensor.dtype(), &self.device)?;

        unsafe {
            launch_repeat(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                tensor.dtype(),
                tensor_contig.ptr(),
                out.ptr(),
                tensor.shape(),
                &params.out_shape,
            )?;
        }

        Ok(out)
    }

    fn pad(
        &self,
        tensor: &Tensor<CudaRuntime>,
        padding: &[usize],
        value: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_pad(tensor, padding)?;

        // Handle no-op case (all padding is zero)
        if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
            return tensor.contiguous();
        }

        let tensor_contig = ensure_contiguous(tensor)?;
        let out = Tensor::<CudaRuntime>::empty(&params.out_shape, tensor.dtype(), &self.device)?;

        // Extract pad_before from pad_per_dim
        let pad_before: Vec<usize> = params.pad_per_dim.iter().map(|(b, _)| *b).collect();

        unsafe {
            launch_pad(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                tensor.dtype(),
                tensor_contig.ptr(),
                out.ptr(),
                value,
                tensor.shape(),
                &params.out_shape,
                &pad_before,
            )?;
        }

        Ok(out)
    }

    fn roll(
        &self,
        tensor: &Tensor<CudaRuntime>,
        shift: isize,
        dim: isize,
    ) -> Result<Tensor<CudaRuntime>> {
        let params = shape_ops::validate_roll(tensor, shift, dim)?;

        // Handle no-op case (shift is 0 or multiple of dim_size)
        if params.shift == 0 {
            return tensor.contiguous();
        }

        let tensor_contig = ensure_contiguous(tensor)?;
        let out = Tensor::<CudaRuntime>::empty(tensor.shape(), tensor.dtype(), &self.device)?;

        // Compute outer/inner sizes. Never clamp these with `.max(1)`: an empty
        // slice already products to 1, so a clamp only fires on a genuinely zero
        // extent — and then defeats the launcher's own `total_elements == 0` early
        // return, writing to a zero-element output.
        let outer_size: usize = tensor.shape()[..params.dim_idx].iter().product();
        let inner_size: usize = tensor.shape()[params.dim_idx + 1..].iter().product();

        unsafe {
            launch_roll(
                &self.context,
                &self.stream,
                self.device.index,
                tensor.dtype(),
                tensor_contig.ptr(),
                out.ptr(),
                outer_size,
                params.dim_size,
                inner_size,
                params.shift,
            )?;
        }

        Ok(out)
    }

    fn unfold(
        &self,
        tensor: &Tensor<CudaRuntime>,
        dim: isize,
        size: usize,
        step: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        unfold_impl(self, tensor, dim, size, step)
    }

    fn repeat_interleave(
        &self,
        tensor: &Tensor<CudaRuntime>,
        repeats: usize,
        dim: Option<isize>,
    ) -> Result<Tensor<CudaRuntime>> {
        repeat_interleave_impl(self, tensor, repeats, dim)
    }
}
