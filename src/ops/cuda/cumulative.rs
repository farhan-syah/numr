//! Cumulative operations for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::{CumulativeOps, reduce_dim_output_shape, reduce_output_shape};
use crate::runtime::cuda::kernels::{
    launch_cumprod, launch_cumprod_strided, launch_cumsum, launch_cumsum_strided, launch_logsumexp,
    launch_logsumexp_strided,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl CumulativeOps<CudaRuntime> for CudaClient {
    fn cumsum(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                shape,
                a.dtype(),
                &self.device,
            )?);
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a)?;

        // Calculate dimensions for kernel launch. Never clamp these with `.max(1)`:
        // the `numel() == 0` guard above already rules a zero out, and a clamp
        // would fabricate a row the allocation does not contain.
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device)?;

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            unsafe {
                launch_cumsum(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    scan_size,
                    outer_size,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumsum_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    scan_size,
                    outer_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn cumprod(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let shape = a.shape();
        let ndim = shape.len();

        // Normalize dimension (handle negative indexing)
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        // Handle empty tensor
        if a.numel() == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(
                shape,
                a.dtype(),
                &self.device,
            )?);
        }

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(a)?;

        // Calculate dimensions for kernel launch. Never clamp these with `.max(1)`:
        // the `numel() == 0` guard above already rules a zero out, and a clamp
        // would fabricate a row the allocation does not contain.
        let scan_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Allocate output
        let out = Tensor::<CudaRuntime>::empty(shape, a.dtype(), &self.device)?;

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Scan along last dimension or effectively contiguous
            unsafe {
                launch_cumprod(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    scan_size,
                    outer_size,
                )?;
            }
        } else {
            // Strided scan for non-last dimension
            unsafe {
                launch_cumprod_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    scan_size,
                    outer_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn logsumexp(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Support: F32, F64, F16, BF16
        // For F16/BF16: upcast to F32, compute, downcast back
        use crate::dtype::DType;
        use crate::ops::TypeConversionOps;

        let input_dtype = a.dtype();
        if !matches!(
            input_dtype,
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2
        ) {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "logsumexp",
            });
        }

        // F16/BF16/FP8 have native CUDA kernels that accumulate in F32 internally
        let (a_compute, needs_cast) = (a.clone(), false);

        let shape = a_compute.shape();
        let ndim = shape.len();

        // Handle empty dims (reduce over all dimensions)
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..ndim).collect()
        } else {
            dims.to_vec()
        };

        // Validate dimensions
        for &dim in &actual_dims {
            if dim >= ndim {
                return Err(Error::InvalidDimension {
                    dim: dim as isize,
                    ndim,
                });
            }
        }

        // Every output element folds over no input: `log(sum of nothing)` is
        // `log(0)`, so the identity is -inf. An empty output takes the same
        // path and stays empty. Never hand back the uninitialized allocation.
        if a_compute.numel() == 0 {
            let out_shape = reduce_output_shape(shape, &actual_dims, keepdim);
            let out = Tensor::<CudaRuntime>::full_scalar(
                &out_shape,
                a_compute.dtype(),
                f64::NEG_INFINITY,
                &self.device,
            )?;
            // Cast back to original dtype if needed
            return if needs_cast {
                Ok(self.cast(&out, input_dtype)?)
            } else {
                Ok(out)
            };
        }

        // For multi-dimensional reduction, reduce one dimension at a time
        if actual_dims.len() > 1 {
            let mut result = a_compute.clone();
            // Sort dims in descending order to avoid index invalidation
            let mut sorted_dims = actual_dims.clone();
            sorted_dims.sort_by(|a, b| b.cmp(a));

            for &dim in &sorted_dims {
                result = self.logsumexp(&result, &[dim], true)?;
            }

            // Remove keepdim if not requested
            if !keepdim {
                let out_shape = reduce_output_shape(shape, &actual_dims, false);
                result = result.reshape(&out_shape)?;
            }

            return Ok(result);
        }

        // Single dimension reduction
        let dim = actual_dims[0];

        // Ensure contiguous for CUDA kernel
        let a_contig = ensure_contiguous(&a_compute)?;

        // Calculate dimensions for kernel launch. Never clamp these with `.max(1)`:
        // the `numel() == 0` guard above already rules a zero out, and a clamp
        // would fabricate a row the allocation does not contain.
        let reduce_size = shape[dim];
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        // Calculate output shape
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);
        let out_numel: usize = out_shape.iter().product();

        // Allocate output (in F32 if upcast, else in original dtype)
        let compute_dtype = a_compute.dtype();
        let out = Tensor::<CudaRuntime>::empty(&out_shape, compute_dtype, &self.device)?;

        // Choose kernel based on dimension position
        if inner_size == 1 {
            // Reduction along last dimension
            unsafe {
                launch_logsumexp(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a_compute.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    reduce_size,
                    outer_size,
                )?;
            }
        } else {
            // Strided reduction for non-last dimension
            unsafe {
                launch_logsumexp_strided(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    a_compute.dtype(),
                    a_contig.ptr(),
                    out.ptr(),
                    reduce_size,
                    outer_size,
                    inner_size,
                )?;
            }
        }

        // Cast back to original dtype if needed
        let result = if needs_cast {
            self.cast(&out, input_dtype)?
        } else {
            out
        };

        // Handle keepdim reshape if needed
        if keepdim && result.numel() == out_numel {
            Ok(result)
        } else {
            Ok(result)
        }
    }
}
