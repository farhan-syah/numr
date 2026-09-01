//! Dimension-wise reduction launcher for the CUDA client.
//!
//! Reduces one dimension per kernel launch, reshaping to the requested output
//! shape once every dimension has been consumed.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::reduce_output_shape;
use crate::runtime::cuda::kernels::{
    AccumulationPrecision, launch_reduce_dim_op, reduce_dim_split_count,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Launch a native CUDA reduction operation (sum, max, min along dimensions).
///
/// # Performance
///
/// - **Single dimension**: Uses optimized CUDA kernel with warp-level reductions (fast)
/// - **Multiple dimensions**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// # Arguments
/// * `op` - Operation name ("sum", "max", "min")
/// * `dims` - Dimensions to reduce over
/// * `keepdim` - Whether to keep reduced dimensions as size 1
/// * `precision` - Optional accumulation precision (higher precision for sum)
pub(crate) fn native_reduce_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    dims: &[usize],
    keepdim: bool,
    precision: Option<AccumulationPrecision>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    // `reduce.cu`/`reduce_int.cu` instantiate no `bool` row, so an unguarded
    // launch would fail kernel lookup with an opaque `Error::Internal`. CPU
    // rejects Bool here too (`dispatch_dtype!` has no Bool arm), so report
    // the same `UnsupportedDType` CPU does instead of a symbol-not-found
    // error.
    if dtype == DType::Bool {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let acc_precision = precision.unwrap_or_default();

    // For single-dimension reduction, use optimized kernel
    if dims.len() == 1 {
        let dim = dims[0];
        let shape = a.shape();

        // Calculate outer, reduce, inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let reduce_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

        // A zero-size output (some non-reduced dimension is 0) has nothing to
        // compute. Never restore a `.max(1)` on the extents above: it would make
        // this guard unreachable and launch a grid over elements the empty
        // allocation does not have.
        if out.numel() == 0 {
            return Ok(out);
        }

        // The single-stage grid is one block per output element and ignores
        // `reduce_size`, so a long axis collapsing to few outputs leaves the
        // device idle. Cutting the axis into equal chunks widens the grid by
        // `splits` at the cost of one extra launch over a much smaller buffer.
        if let Some(splits) = reduce_dim_split_count(
            client.device.index,
            op,
            dtype,
            acc_precision,
            outer_size,
            reduce_size,
            inner_size,
        ) {
            two_stage_reduce_dim(
                client,
                op,
                acc_precision,
                &a_contig,
                &out,
                (outer_size, reduce_size, inner_size),
                splits,
            )?;
            return Ok(out);
        }

        unsafe {
            launch_reduce_dim_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                outer_size,
                reduce_size,
                inner_size,
                acc_precision,
            )?;
        }

        return Ok(out);
    }

    // For multiple dimensions: chain single-dimension reductions on GPU
    // This keeps all computation on the GPU instead of falling back to CPU

    // Sort dimensions from highest to lowest to avoid index shifting issues
    let mut sorted_dims: Vec<usize> = dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    // Reduce one dimension at a time, always keeping dims so each step's indexing
    // stays aligned with the original layout.
    let mut current = a.clone();
    for &dim in &sorted_dims {
        current = native_reduce_op(client, &current, op, &[dim], true, precision)?;
    }

    // Every reduced dim is still present as size 1, so drop them in one step.
    // Squeezing only at the end is what makes a full reduction with keepdim=false
    // collapse to a scalar rather than leaving a trailing size-1 dimension.
    if current.shape() != out_shape.as_slice() {
        current = current.reshape(&out_shape)?;
    }

    Ok(current)
}

/// Reduce one dimension in two stages, cutting the reduced axis into `splits`
/// equal chunks so the grid covers `splits` times as many blocks.
///
/// The input is contiguous `[outer, reduce, inner]` and `splits` divides
/// `reduce` exactly, so the same buffer is also `[outer, splits, chunk, inner]`
/// with `chunk = reduce / splits`. Element `(o, s, c, i)` sits at
/// `((o * splits + s) * chunk + c) * inner + i`, which is precisely what the
/// dim kernel addresses with `outer_size = outer * splits`,
/// `reduce_size = chunk` and `inner_size = inner` — so stage 1 is an ordinary
/// dim reduction whose output is a contiguous `[outer, splits, inner]` buffer.
/// Stage 2 reduces that buffer over `splits` with the same kernel again.
///
/// [`reduce_dim_split_count`] has already established that `op` merges exactly
/// under this split. The scratch buffer is an ordinary tensor: it is freed on
/// the client's canonical stream when it drops here, after both launches are
/// enqueued on that same stream, so the free is ordered behind their use.
fn two_stage_reduce_dim(
    client: &CudaClient,
    op: &'static str,
    acc_precision: AccumulationPrecision,
    input: &Tensor<CudaRuntime>,
    out: &Tensor<CudaRuntime>,
    extents: (usize, usize, usize),
    splits: usize,
) -> Result<()> {
    let (outer_size, reduce_size, inner_size) = extents;
    let dtype = input.dtype();
    let chunk = reduce_size / splits;

    let partial =
        Tensor::<CudaRuntime>::empty(&[outer_size, splits, inner_size], dtype, &client.device)?;
    // The allocation above proved `outer_size * splits * inner_size` fits, so
    // this product cannot overflow.
    let stage1_outer = outer_size.saturating_mul(splits);

    unsafe {
        launch_reduce_dim_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            input.ptr(),
            partial.ptr(),
            stage1_outer,
            chunk,
            inner_size,
            acc_precision,
        )?;

        launch_reduce_dim_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            partial.ptr(),
            out.ptr(),
            outer_size,
            splits,
            inner_size,
            acc_precision,
        )?;
    }

    Ok(())
}
