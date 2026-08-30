//! Overwrite a contiguous run along one dimension with another tensor.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{index, launch_slice_assign};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_slice_assign(
    client: &WgpuClient,
    dst: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
    dim: usize,
    start: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let ndim = dst.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    if src.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: dst.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && src.shape()[d] != dst.shape()[d] {
            return Err(Error::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
    }

    let src_dim_size = src.shape()[dim];
    let dst_dim_size = dst.shape()[dim];
    if start + src_dim_size > dst_dim_size {
        return Err(Error::InvalidArgument {
            arg: "start",
            reason: format!(
                "start ({}) + src dim size ({}) exceeds dst dim size ({})",
                start, src_dim_size, dst_dim_size
            ),
        });
    }

    let dtype = dst.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Never clamp these with `.max(1)`: an empty slice already products to 1, so a
    // clamp fires only on a genuinely zero extent, and then hides it from the
    // `total_src == 0` guard below.
    let outer_size: usize = dst.shape()[..dim].iter().product();
    let inner_size: usize = dst.shape()[dim + 1..].iter().product();
    let total_src = outer_size * src_dim_size * inner_size;

    let dst_contig = ensure_contiguous(dst)?;
    let src_contig = ensure_contiguous(src)?;

    let out = alloc_output(client, dst.shape(), dtype)?;

    // An empty destination has no buffer to bind and no slice to overwrite.
    if dst.numel() == 0 {
        return Ok(out);
    }

    let dst_buf = get_tensor_buffer(&dst_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // First copy dst → output
    let copy_params = CopyParams {
        numel: dst.numel() as u32,
    };
    let copy_params_buf = create_params_buffer(client, &copy_params);
    index::launch_copy(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &dst_buf,
        &out_buf,
        &copy_params_buf,
        dst.numel(),
        dtype,
    )?;

    // An empty source overwrites nothing, so the copy above is already the whole
    // result, and `src` is the zero-byte allocation `get_tensor_buffer` cannot
    // return a buffer for.
    if total_src == 0 {
        return Ok(out);
    }

    let src_buf = get_tensor_buffer(&src_contig)?;

    // Then overwrite the slice with src
    let params = SliceAssignParams {
        outer_size: outer_size as u32,
        dst_dim_size: dst_dim_size as u32,
        src_dim_size: src_dim_size as u32,
        inner_size: inner_size as u32,
        start: start as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_slice_assign(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &src_buf,
        &out_buf,
        &params_buf,
        total_src,
        dtype,
    )?;

    Ok(out)
}
