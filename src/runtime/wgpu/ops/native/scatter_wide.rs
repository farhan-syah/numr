//! Integer `scatter_reduce` sum and mean, accumulated wide.
//!
//! An integer scatter reduction accumulates, and accumulators saturate rather
//! than wrap (runtime/cpu/kernels/wide_acc.rs). A 32-bit atomic on the element
//! type cannot deliver that: a total that crosses the dtype's range wraps and
//! comes back with the wrong sign, and `mean` then divides a total that is
//! already wrong. So the destination is accumulated as a 64-bit value, two u32
//! limbs per element, and narrowed exactly once at the end - after the single
//! division, for `mean`. This mirrors CPU's i128 accumulator in
//! runtime/cpu/kernels/scatter_reduce_int.rs.

use wgpu::Buffer;

use super::helpers::*;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::shaders::{
    ScatterWideParams, launch_scatter_reduce_count, launch_scatter_wide_finalize,
    launch_scatter_wide_seed, launch_scatter_wide_sum,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Scatter-reduce `src` into `seeded_output` with a 64-bit accumulator.
///
/// `seeded_output` already holds the destination's starting values: the original
/// tensor when `include_self` is set, the reduction's identity otherwise. Both
/// are just the accumulator's initial contribution, so one code path covers them.
///
/// `scatter_params` is the shared `ScatterReduceParams` buffer the elementwise
/// scatter kernels also read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_scatter_reduce_int_wide(
    client: &WgpuClient,
    seeded_output: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
    index: &Tensor<WgpuRuntime>,
    scatter_params: &Buffer,
    total_src: usize,
    is_mean: bool,
    include_self: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = seeded_output.dtype();
    let dst_shape = seeded_output.shape().to_vec();
    let numel = seeded_output.numel();

    if numel == 0 {
        return Ok(seeded_output.clone());
    }

    // Two u32 limbs per destination element, low limb first.
    let acc = alloc_output(client, &[numel * 2], DType::U32)?;

    // Mean's denominator. include_self makes the destination's own value one of
    // the averaged contributions, exactly as on CPU. Sum never divides, so its
    // counts are never read - the buffer is still bound, because the finalize
    // kernel is one shader.
    let count_tensor = if is_mean {
        let count_init = u32::from(include_self);
        Tensor::<WgpuRuntime>::from_slice(&vec![count_init; numel], &dst_shape, client.device())?
    } else {
        alloc_output(client, &dst_shape, DType::U32)?
    };

    let wide_params = ScatterWideParams {
        n: numel as u32,
        divide: u32::from(is_mean),
        _pad0: 0,
        _pad1: 0,
    };
    let wide_params_buf = create_params_buffer(client, &wide_params);

    let result = alloc_output(client, &dst_shape, dtype)?;

    let seed_buf = get_tensor_buffer(seeded_output)?;
    let acc_buf = get_tensor_buffer(&acc)?;
    let src_buf = get_tensor_buffer(src)?;
    let index_buf = get_tensor_buffer(index)?;
    let count_buf = get_tensor_buffer(&count_tensor)?;
    let result_buf = get_tensor_buffer(&result)?;

    let cache = client.pipeline_cache();
    let queue = client.wgpu_queue();

    launch_scatter_wide_seed(
        cache,
        queue,
        &seed_buf,
        &acc_buf,
        &wide_params_buf,
        numel,
        dtype,
    )?;
    launch_scatter_wide_sum(
        cache,
        queue,
        &src_buf,
        &index_buf,
        &acc_buf,
        scatter_params,
        total_src,
        dtype,
    )?;
    if is_mean {
        launch_scatter_reduce_count(
            cache,
            queue,
            &index_buf,
            &count_buf,
            scatter_params,
            total_src,
            dtype,
        )?;
    }
    launch_scatter_wide_finalize(
        cache,
        queue,
        &acc_buf,
        &count_buf,
        &result_buf,
        &wide_params_buf,
        numel,
        dtype,
    )?;

    Ok(result)
}
