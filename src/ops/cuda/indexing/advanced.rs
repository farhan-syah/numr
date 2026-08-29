//! Advanced indexing operations for CUDA runtime

use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ScatterReduceOp, TypeConversionOps};
use crate::runtime::cuda::kernels::{
    ScatterReduceOpCuda, launch_bincount_weighted, launch_copy, launch_embedding_lookup,
    launch_fill_with_f64, launch_gather_nd, launch_scatter_reduce, launch_scatter_reduce_count,
    launch_scatter_reduce_int, launch_scatter_reduce_mean_div,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

use super::helpers::normalize_indices_to_i64;

/// Execute embedding_lookup operation.
pub fn embedding_lookup(
    client: &CudaClient,
    embeddings: &Tensor<CudaRuntime>,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    let indices_i64 = normalize_indices_to_i64(client, indices)?;

    let vocab_size = emb_shape[0];
    let embedding_dim = emb_shape[1];
    let num_indices = indices_i64.numel();

    // Output shape: indices.shape() + [embedding_dim]
    let mut out_shape = indices_i64.shape().to_vec();
    out_shape.push(embedding_dim);

    let emb_contig = ensure_contiguous(embeddings)?;
    let idx_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_embedding_lookup(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            emb_contig.ptr(),
            idx_contig.ptr(),
            out.ptr(),
            num_indices,
            vocab_size,
            embedding_dim,
        )?;
    }

    Ok(out)
}

/// Execute scatter_reduce operation.
pub fn scatter_reduce(
    client: &CudaClient,
    dst: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = dst.dtype();

    // The float scatter_reduce kernels use atomics, which CUDA provides for
    // F32 and F64 only. Narrower floats (F16, BF16, FP8) promote to F32,
    // compute, and demote back. Integers take their own kernel instead, which
    // needs no atomic at all — see launch_scatter_reduce_int.
    if dtype.is_float() && !matches!(dtype, DType::F32 | DType::F64) {
        let (dst_promoted, orig_dtype) = linalg_promote(client, dst)?;
        let (src_promoted, _) = linalg_promote(client, src)?;
        let result = scatter_reduce(
            client,
            &dst_promoted,
            dim,
            index,
            &src_promoted,
            op,
            include_self,
        )?;
        return linalg_demote(client, result, orig_dtype);
    }
    let shape = dst.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate that index and src have same shape
    if index_i64.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate that index has same number of dimensions as dst
    if index_i64.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Map ScatterReduceOp to ScatterReduceOpCuda
    // The integer kernel reduces `mean` itself; the float path reaches it as a
    // Sum pass followed by count and divide passes.
    let cuda_op = match (op, dtype.is_int()) {
        (ScatterReduceOp::Sum, _) => ScatterReduceOpCuda::Sum,
        (ScatterReduceOp::Max, _) => ScatterReduceOpCuda::Max,
        (ScatterReduceOp::Min, _) => ScatterReduceOpCuda::Min,
        (ScatterReduceOp::Prod, _) => ScatterReduceOpCuda::Prod,
        (ScatterReduceOp::Mean, true) => ScatterReduceOpCuda::Mean,
        (ScatterReduceOp::Mean, false) => ScatterReduceOpCuda::Sum,
    };

    let dst_contig = ensure_contiguous(dst)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;

    // Allocate output and initialize with dst values if include_self
    let out = Tensor::<CudaRuntime>::empty(shape, dtype, &client.device)?;

    if include_self {
        // Copy dst to output
        unsafe {
            launch_copy(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                dst_contig.ptr(),
                out.ptr(),
                dst.numel(),
            )?;
        }
    } else {
        // Initialize output to identity element for the reduction
        let identity = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => 0.0,
            ScatterReduceOp::Max => f64::NEG_INFINITY,
            ScatterReduceOp::Min => f64::INFINITY,
            ScatterReduceOp::Prod => 1.0,
        };
        unsafe {
            launch_fill_with_f64(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                identity,
                out.ptr(),
                dst.numel(),
            )?;
        }
    }

    // Compute dimensions for scatter
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = src.shape()[dim];

    if dtype.is_int() {
        // One launch covers every integer reduction, mean included: the kernel
        // owns each destination element, so it keeps a 128-bit accumulator and
        // divides once at the end instead of scattering with atomics.
        unsafe {
            launch_scatter_reduce_int(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                src_contig.ptr(),
                index_contig.ptr(),
                out.ptr(),
                outer_size,
                dim_size,
                inner_size,
                src_dim_size,
                cuda_op,
                include_self,
            )?;
        }
        return Ok(out);
    }

    unsafe {
        launch_scatter_reduce(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            src_contig.ptr(),
            index_contig.ptr(),
            out.ptr(),
            dim,
            outer_size,
            dim_size,
            inner_size,
            src_dim_size,
            cuda_op,
        )?;
    }

    // Float mean: divide the scattered sum by the scattered count.
    if matches!(op, ScatterReduceOp::Mean) {
        // Allocate count buffer (same shape as output, zero-initialized)
        let count = Tensor::<CudaRuntime>::empty(shape, dtype, &client.device)?;
        unsafe {
            launch_fill_with_f64(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                0.0,
                count.ptr(),
                dst.numel(),
            )?;
        }

        // If include_self, each dst element starts with count=1
        if include_self {
            unsafe {
                launch_fill_with_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    dtype,
                    1.0,
                    count.ptr(),
                    dst.numel(),
                )?;
            }
        }

        // Scatter count: atomicAdd 1 for each src element
        unsafe {
            launch_scatter_reduce_count(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                index_contig.ptr(),
                count.ptr(),
                dim,
                outer_size,
                dim_size,
                inner_size,
                src_dim_size,
            )?;
        }

        // Divide sum by count
        let result = Tensor::<CudaRuntime>::empty(shape, dtype, &client.device)?;
        unsafe {
            launch_scatter_reduce_mean_div(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                out.ptr(),
                count.ptr(),
                result.ptr(),
                dst.numel(),
            )?;
        }

        return Ok(result);
    }

    Ok(out)
}

/// Execute gather_nd operation.
pub fn gather_nd(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let input_shape = input.shape();
    let indices_i64 = normalize_indices_to_i64(client, indices)?;
    let indices_shape = indices_i64.shape();

    // Indices must have at least 1 dimension
    if indices_shape.is_empty() {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: indices_shape.to_vec(),
        });
    }

    // Last dimension of indices is the number of coordinates (M)
    let indices_ndim = indices_shape.len();
    let index_depth = indices_shape[indices_ndim - 1]; // M

    // M must not exceed input dimensions
    if index_depth > input_shape.len() {
        return Err(Error::InvalidDimension {
            dim: index_depth as isize,
            ndim: input_shape.len(),
        });
    }

    // Compute output shape: indices.shape[:-1] + input.shape[M:]
    let mut out_shape: Vec<usize> = indices_shape[..indices_ndim - 1].to_vec();
    out_shape.extend_from_slice(&input_shape[index_depth..]);

    // Handle scalar output case
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    // Compute num_slices (product of indices.shape[:-1])
    let num_slices: usize = indices_shape[..indices_ndim - 1].iter().product();
    let num_slices = num_slices.max(1);

    // Compute slice_size (product of input.shape[M:])
    let slice_size: usize = input_shape[index_depth..].iter().product();
    let slice_size = slice_size.max(1);

    let input_contig = ensure_contiguous(input)?;
    let indices_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    // Prepare shape and stride arrays as host Vecs (passed as scalar args, no device alloc needed)
    let input_shape_u32: Vec<u32> = input_shape.iter().map(|&s| s as u32).collect();
    let input_strides: Vec<usize> = compute_contiguous_strides(input_shape);
    let input_strides_u32: Vec<u32> = input_strides.iter().map(|&s| s as u32).collect();

    let ndim = input_shape.len();

    unsafe {
        launch_gather_nd(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            input_contig.ptr(),
            indices_contig.ptr(),
            out.ptr(),
            &input_shape_u32,
            &input_strides_u32,
            num_slices,
            slice_size,
            index_depth,
            ndim,
        )?;
    }

    Ok(out)
}

/// Execute bincount operation.
pub fn bincount(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weights: Option<&Tensor<CudaRuntime>>,
    minlength: usize,
) -> Result<Tensor<CudaRuntime>> {
    // Validate input is 1D
    if input.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![input.numel()],
            got: input.shape().to_vec(),
        });
    }

    // Validate input is integer type
    let input_dtype = input.dtype();
    if !matches!(input_dtype, DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: input_dtype,
        });
    }

    // Validate weights if provided
    let weights_dtype = if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: w.shape().to_vec(),
            });
        }
        Some(w.dtype())
    } else {
        None
    };

    let out_dtype = weights_dtype.unwrap_or(DType::I64);
    let input_contig = ensure_contiguous(input)?;
    let numel = input.numel();

    // Find the max value on GPU to determine output size.
    // Cast to F64 for max reduction (CUDA reduce kernels support F64 but not integer types),
    // then read the single scalar back to CPU for allocation sizing —
    // this is a necessary system boundary (same as CPU impl computing max first).
    // F64 preserves full i32/i64 precision (up to 2^53), unlike F32 which loses precision past 2^24.
    let input_f64 = client.cast(input, DType::F64)?;
    let max_tensor = client.max(&input_f64, &[0], false)?;
    let max_val = max_tensor.item::<f64>()? as i64;
    if max_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = ((max_val as usize) + 1).max(minlength);

    // Allocate output and zero-initialize
    let out = Tensor::<CudaRuntime>::empty(&[output_len], out_dtype, &client.device)?;

    // Zero the output buffer
    unsafe {
        launch_fill_with_f64(
            &client.context,
            &client.stream,
            client.device.index,
            out_dtype,
            0.0,
            out.ptr(),
            output_len,
        )?;
    }

    let weights_contig = weights.map(ensure_contiguous).transpose()?;
    let weights_ptr = weights_contig.as_ref().map(|w| w.ptr());

    unsafe {
        launch_bincount_weighted(
            &client.context,
            &client.stream,
            client.device.index,
            input_dtype,
            weights_dtype,
            input_contig.ptr(),
            weights_ptr,
            out.ptr(),
            numel,
            output_len,
        )?;
    }

    Ok(out)
}
