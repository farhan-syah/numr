//! Embedding lookup and bincount kernel launchers

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch embedding_lookup kernel.
///
/// Looks up embeddings from an embedding table using indices.
/// For each index i: output[i, :] = embeddings[indices[i], :]
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - embeddings must be 2D [vocab_size, embedding_dim]
/// - indices must contain values in [0, vocab_size)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_embedding_lookup(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    embeddings_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    num_indices: usize,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<()> {
    if num_indices == 0 || embedding_dim == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!(
            "embedding_lookup_{}",
            index_dtype_suffix(dtype, "embedding_lookup")?
        );
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(num_indices);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let num_indices_u32 = num_indices as u32;
        let vocab_size_u32 = vocab_size as u32;
        let embedding_dim_u32 = embedding_dim as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&embeddings_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&num_indices_u32);
        builder.arg(&vocab_size_u32);
        builder.arg(&embedding_dim_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA embedding_lookup kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch bincount kernel.
///
/// Counts occurrences of each value in an integer tensor, optionally with weights.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_bincount_weighted(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    weights_dtype: Option<DType>,
    input_ptr: u64,
    weights_ptr: Option<u64>,
    output_ptr: u64,
    n: usize,
    minlength: usize,
) -> Result<()> {
    if n == 0 || minlength == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;

        let func_name = match (input_dtype, weights_ptr, weights_dtype) {
            (DType::I32, None, _) => "bincount_i32",
            (DType::I64, None, _) => "bincount_i64",
            (DType::I32, Some(_), Some(DType::F32)) => "bincount_weighted_f32",
            (DType::I32, Some(_), Some(DType::F64)) => "bincount_weighted_f64",
            (DType::I64, Some(_), Some(DType::F32)) => "bincount_i64_weighted_f32",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("bincount requires i32/i64 input, got {:?}", input_dtype),
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;
        let minlength_u32 = minlength as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);

        let weights_ptr_val = weights_ptr.unwrap_or(0);
        if weights_ptr.is_some() {
            builder.arg(&weights_ptr_val);
        }

        builder.arg(&output_ptr);
        builder.arg(&n_u32);
        builder.arg(&minlength_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA bincount kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}
