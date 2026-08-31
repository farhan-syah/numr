//! Slice assign kernel launcher

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_names::INDEX_ND_MODULE, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch slice_assign kernel: copies src into a region of output (pre-copied from dst).
///
/// Output must already contain a copy of dst. This kernel overwrites the slice region
/// [start..start+src_dim_size] along the specified dimension with src data.
///
/// # Safety
///
/// - src_ptr: valid device memory with outer_size * src_dim_size * inner_size elements
/// - output_ptr: valid device memory with outer_size * dst_dim_size * inner_size elements
pub unsafe fn launch_slice_assign(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dst_dim_size: usize,
    src_dim_size: usize,
    inner_size: usize,
    start: usize,
) -> Result<()> {
    let total = outer_size * src_dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_ND_MODULE)?;
        let func_name = format!(
            "slice_assign_{}",
            index_dtype_suffix(dtype, "slice_assign")?
        );
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dst_dim_u32 = dst_dim_size as u32;
        let src_dim_u32 = src_dim_size as u32;
        let inner_u32 = inner_size as u32;
        let start_u32 = start as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dst_dim_u32);
        builder.arg(&src_dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&start_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA slice_assign kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
