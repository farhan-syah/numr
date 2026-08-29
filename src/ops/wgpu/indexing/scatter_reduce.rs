//! scatter_reduce for WebGPU.
//!
//! Sum, max, min and prod land in one atomic dispatch. Mean runs as three:
//! scatter the sum, scatter the per-destination count, then divide. Integer sum
//! and mean detour through the 64-bit accumulator in native/scatter_wide.rs so
//! they cannot wrap mid-accumulation.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ScalarOps, ScatterReduceOp};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    MeanDivParams, ScatterReduceParams, alloc_output, create_params_buffer, ensure_i32_indices,
    get_tensor_buffer,
};
use crate::runtime::wgpu::ops::native::native_scatter_reduce_int_wide;
use crate::runtime::wgpu::shaders::{
    launch_scatter_reduce, launch_scatter_reduce_count, launch_scatter_reduce_mean_div,
    launch_scatter_reduce_prod,
};
use crate::tensor::Tensor;

pub(super) fn scatter_reduce(
    client: &WgpuClient,
    dst: &Tensor<WgpuRuntime>,
    dim: usize,
    index: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = dst.dtype();

    // WebGPU covers the three dtypes WGSL has; f32 reaches its atomics
    // through CAS loops with a bitcast.
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce",
        });
    }

    // Validate index dtype
    if !matches!(index.dtype(), DType::I32 | DType::I64) {
        return Err(Error::InvalidArgument {
            arg: "index",
            reason: "scatter_reduce index must be I32 or I64".to_string(),
        });
    }

    // Element-wise semantics (matching CPU/CUDA/PyTorch): index and src must
    // have the same shape — `index[e]` gives the destination coordinate along
    // `dim` for source element `e`.
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }

    // Ensure contiguous
    let dst = ensure_contiguous(dst)?;
    let index_i32 = ensure_i32_indices(client, index)?;
    let index = ensure_contiguous(&index_i32)?;
    let src = ensure_contiguous(src)?;

    // Compute shape parameters
    let dst_shape = dst.shape();
    let ndim = dst_shape.len();
    if dim >= ndim {
        return Err(Error::InvalidArgument {
            arg: "dim",
            reason: format!("dim {} out of bounds for tensor with {} dims", dim, ndim),
        });
    }

    let outer_size: usize = dst_shape[..dim].iter().product();
    let dim_size = dst_shape[dim];
    let inner_size: usize = dst_shape[dim + 1..].iter().product();
    let src_dim_size = src.shape().get(dim).copied().unwrap_or(1);
    let total_src = src.numel();

    // Initialize output with identity for the operation
    let identity = match op {
        ScatterReduceOp::Sum | ScatterReduceOp::Mean => 0.0f64,
        ScatterReduceOp::Max => f64::NEG_INFINITY,
        ScatterReduceOp::Min => f64::INFINITY,
        ScatterReduceOp::Prod => 1.0,
    };
    let output = if include_self {
        // Must deep-copy: clone() shares the GPU buffer, but scatter_reduce
        // modifies it in-place via atomics, which would corrupt the original.
        client.add_scalar(&dst, 0.0)?
    } else {
        Tensor::full_scalar(dst_shape, dtype, identity, client.device())?
    };

    // Create shared params
    let params = ScatterReduceParams {
        dim: dim as u32,
        outer_size: outer_size as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size as u32,
        src_dim_size: src_dim_size as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    let src_buf = get_tensor_buffer(&src)?;
    let index_buf = get_tensor_buffer(&index)?;
    let output_buf = get_tensor_buffer(&output)?;

    // Integer sum and mean accumulate, so they run in a 64-bit accumulator
    // and narrow once. See native/scatter_wide.rs.
    if dtype.is_int() && matches!(op, ScatterReduceOp::Sum | ScatterReduceOp::Mean) {
        return native_scatter_reduce_int_wide(
            client,
            &output,
            &src,
            &index,
            &params_buf,
            total_src,
            matches!(op, ScatterReduceOp::Mean),
            include_self,
        );
    }

    match op {
        ScatterReduceOp::Prod => {
            // The integer product also accumulates, but its running state
            // is a magnitude and a sign rather than a wide sum, so it needs
            // no second buffer: one thread owns each DESTINATION element
            // and scans its own lane. The float kernel still owns one
            // SOURCE element, hence the different dispatch size.
            let items = if dtype.is_int() {
                output.numel()
            } else {
                total_src
            };
            launch_scatter_reduce_prod(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &src_buf,
                &index_buf,
                &output_buf,
                &params_buf,
                items,
                dtype,
            )?;
            Ok(output)
        }
        ScatterReduceOp::Mean => {
            // Step 1: scatter sum
            launch_scatter_reduce(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &src_buf,
                &index_buf,
                &output_buf,
                &params_buf,
                total_src,
                dtype,
                "sum",
            )?;

            // Step 2: scatter count (u32 buffer)
            let numel = dst.numel();
            let count_init = if include_self { 1u32 } else { 0u32 };
            let count_data = vec![count_init; numel];
            let count_tensor =
                Tensor::<WgpuRuntime>::from_slice(&count_data, dst_shape, client.device())?;
            let count_buf = get_tensor_buffer(&count_tensor)?;

            launch_scatter_reduce_count(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &index_buf,
                &count_buf,
                &params_buf,
                total_src,
                dtype,
            )?;

            // Step 3: divide sum by count
            let result = alloc_output(client, dst_shape, dtype)?;
            let result_buf = get_tensor_buffer(&result)?;

            let mean_params = MeanDivParams {
                n: numel as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let mean_params_buf = create_params_buffer(client, &mean_params);

            launch_scatter_reduce_mean_div(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &output_buf,
                &count_buf,
                &result_buf,
                &mean_params_buf,
                numel,
                dtype,
            )?;

            Ok(result)
        }
        _ => {
            // Sum, Max, Min - use existing shader
            let op_str = match op {
                ScatterReduceOp::Sum => "sum",
                ScatterReduceOp::Max => "max",
                ScatterReduceOp::Min => "min",
                _ => unreachable!(),
            };

            launch_scatter_reduce(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &src_buf,
                &index_buf,
                &output_buf,
                &params_buf,
                total_src,
                dtype,
                op_str,
            )?;

            Ok(output)
        }
    }
}
