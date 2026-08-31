//! Scatter-with-reduction kernel launchers.
//!
//! Two families, matching the two in scatter_reduce.cu:
//!
//! - Float (F32, F64): [`launch_scatter_reduce`] runs one atomic pass per
//!   source element. `mean` needs [`launch_scatter_reduce_count`] and
//!   [`launch_scatter_reduce_mean_div`] after it.
//! - Integer: [`launch_scatter_reduce_int`] does the whole reduction in one
//!   pass, including `mean`, keeping a 128-bit accumulator per destination
//!   element and dividing once at the end.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_names::SCATTER_REDUCE_MODULE, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Scatter reduce operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduceOpCuda {
    /// Sum reduction: accumulate values by addition.
    Sum,
    /// Max reduction: keep the maximum value.
    Max,
    /// Min reduction: keep the minimum value.
    Min,
    /// Product reduction: accumulate values by multiplication.
    Prod,
    /// Mean reduction: sum, then divide by the number of contributions.
    ///
    /// Integer only. The float path spells mean as Sum plus
    /// [`launch_scatter_reduce_count`] and [`launch_scatter_reduce_mean_div`].
    Mean,
}

impl ScatterReduceOpCuda {
    /// The kernel-name fragment for this operation.
    fn name(self) -> &'static str {
        match self {
            ScatterReduceOpCuda::Sum => "sum",
            ScatterReduceOpCuda::Max => "max",
            ScatterReduceOpCuda::Min => "min",
            ScatterReduceOpCuda::Prod => "prod",
            ScatterReduceOpCuda::Mean => "mean",
        }
    }
}

/// Launch the float scatter_reduce kernel.
///
/// `dst` must already hold the initial value — a copy of the destination when
/// include_self is set, otherwise the reduction's identity.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for any dtype but F32 and F64, and for
/// `Mean`, which this family reaches through Sum plus a divide pass.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    indices_ptr: u64,
    dst_ptr: u64,
    dim: usize,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    src_dim_size: usize,
    op: ScatterReduceOpCuda,
) -> Result<()> {
    let total = outer_size * src_dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_reduce",
            });
        }
    };
    if op == ScatterReduceOpCuda::Mean {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce_mean",
        });
    }
    let func_name = format!("scatter_reduce_{}_{}", op.name(), suffix);

    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let dim_u32 = dim as u32;
        let outer_size_u32 = outer_size as u32;
        let dim_size_u32 = dim_size as u32;
        let inner_size_u32 = inner_size as u32;
        let src_dim_size_u32 = src_dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&dim_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&inner_size_u32);
        builder.arg(&src_dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA scatter_reduce kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch the integer scatter_reduce kernel.
///
/// One thread per destination element, so a whole reduction — `mean` included —
/// finishes in this single launch. `dst` must already hold the initial value,
/// exactly as for [`launch_scatter_reduce`], and `include_self` must say which
/// initialisation that was: it is the seed of `mean`'s contribution count.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for a non-integer dtype.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce_int(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    indices_ptr: u64,
    dst_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    src_dim_size: usize,
    op: ScatterReduceOpCuda,
    include_self: bool,
) -> Result<()> {
    let total = outer_size * dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    if !dtype.is_int() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce_int",
        });
    }
    let func_name = format!(
        "scatter_reduce_int_{}_{}",
        op.name(),
        index_dtype_suffix(dtype, "scatter_reduce_int")?
    );

    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_size_u32 = outer_size as u32;
        let dim_size_u32 = dim_size as u32;
        let inner_size_u32 = inner_size as u32;
        let src_dim_size_u32 = src_dim_size as u32;
        let include_self_u32 = u32::from(include_self);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&outer_size_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&inner_size_u32);
        builder.arg(&src_dim_size_u32);
        builder.arg(&include_self_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scatter_reduce_int kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch scatter_reduce_count kernel.
///
/// Atomically increments the count buffer at scattered positions. Used only by
/// the float `mean` path: the integer path counts inside its own kernel.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    indices_ptr: u64,
    count_ptr: u64,
    dim: usize,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    src_dim_size: usize,
) -> Result<()> {
    let total = outer_size * src_dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;

        let func_name = match dtype {
            DType::F32 => "scatter_reduce_count_f32",
            DType::F64 => "scatter_reduce_count_f64",
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "scatter_reduce_count",
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(total)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let dim_u32 = dim as u32;
        let outer_size_u32 = outer_size as u32;
        let dim_size_u32 = dim_size as u32;
        let inner_size_u32 = inner_size as u32;
        let src_dim_size_u32 = src_dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&count_ptr);
        builder.arg(&dim_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&inner_size_u32);
        builder.arg(&src_dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scatter_reduce_count kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch scatter_reduce_mean_div kernel.
///
/// Element-wise: `output[i] = sum[i] / count[i]`, and 0 where `count[i]` is 0.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce_mean_div(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sum_ptr: u64,
    count_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;

        let func_name = match dtype {
            DType::F32 => "scatter_reduce_mean_div_f32",
            DType::F64 => "scatter_reduce_mean_div_f64",
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "scatter_reduce_mean_div",
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(n)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&sum_ptr);
        builder.arg(&count_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scatter_reduce_mean_div kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
