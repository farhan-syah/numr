//! Basic linear algebra kernel launchers: trace, diag, diagflat, copy, identity, transpose

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch trace kernel to compute sum of diagonal elements.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [n, n] matrix
/// - `output_ptr` must point to a single element (will be atomically added to)
/// - Output should be zero-initialized before launch
pub unsafe fn launch_trace(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    stride: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "trace_f32",
        DType::F64 => "trace_f64",
        DType::F16 => "trace_f16",
        DType::BF16 => "trace_bf16",
        _ => return Err(Error::UnsupportedDType { dtype, op: "trace" }),
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;
    let stride_u32 = stride as u32;

    let shared_mem = BLOCK_SIZE * std::mem::size_of::<f64>() as u32;

    let cfg = launch_config(grid, block, shared_mem);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n_u32);
    builder.arg(&stride_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA trace kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch diag kernel to extract diagonal elements from matrix.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [m, n] matrix
/// - `output_ptr` must have space for min(m, n) elements
pub unsafe fn launch_diag(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    min_dim: usize,
    n_cols: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "diag_f32",
        DType::F64 => "diag_f64",
        DType::F16 => "diag_f16",
        DType::BF16 => "diag_bf16",
        _ => return Err(Error::UnsupportedDType { dtype, op: "diag" }),
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(min_dim)?;
    let block = (BLOCK_SIZE, 1, 1);
    let min_dim_u32 = min_dim as u32;
    let n_cols_u32 = n_cols as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&min_dim_u32);
    builder.arg(&n_cols_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA diag kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch diagflat kernel to create diagonal matrix from vector.
///
/// # Safety
///
/// - `input_ptr` must point to a valid vector of n elements
/// - `output_ptr` must have space for n*n elements
pub unsafe fn launch_diagflat(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "diagflat_f32",
        DType::F64 => "diagflat_f64",
        DType::F16 => "diagflat_f16",
        DType::BF16 => "diagflat_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "diagflat",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = n * n;
    let grid = elementwise_launch_config(total)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA diagflat kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Copy matrix data on device.
///
/// # Safety
///
/// - `src_ptr` and `dst_ptr` must point to n elements
#[allow(dead_code)]
pub unsafe fn launch_matrix_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "matrix_copy_f32",
        DType::F64 => "matrix_copy_f64",
        DType::F16 => "matrix_copy_f16",
        DType::BF16 => "matrix_copy_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "matrix_copy",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_ptr);
    builder.arg(&dst_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA matrix_copy kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Scatter vector into a column of a matrix.
///
/// # Safety
///
/// - `vec_ptr` must point to n elements
/// - `matrix_ptr` must point to n*n matrix
/// - `col` must be < n
pub unsafe fn launch_scatter_column(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    vec_ptr: u64,
    matrix_ptr: u64,
    n: usize,
    col: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "scatter_column_f32",
        DType::F64 => "scatter_column_f64",
        DType::F16 => "scatter_column_f16",
        DType::BF16 => "scatter_column_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_column",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;
    let col_u32 = col as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&vec_ptr);
    builder.arg(&matrix_ptr);
    builder.arg(&n_u32);
    builder.arg(&col_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!("CUDA scatter_column kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Create identity matrix on GPU.
///
/// # Safety
///
/// - `out_ptr` must have space for n*n elements
pub unsafe fn launch_create_identity(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    out_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "create_identity_f32",
        DType::F64 => "create_identity_f64",
        DType::F16 => "create_identity_f16",
        DType::BF16 => "create_identity_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "create_identity",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = n * n;
    let grid = elementwise_launch_config(total)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&out_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA create_identity kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Extract a column from a matrix.
///
/// # Safety
///
/// - `matrix_ptr` must point to m*n_cols matrix
/// - `col_out_ptr` must have space for m elements
/// - `col` must be < n_cols
pub unsafe fn launch_extract_column(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    matrix_ptr: u64,
    col_out_ptr: u64,
    m: usize,
    n_cols: usize,
    col: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "extract_column_f32",
        DType::F64 => "extract_column_f64",
        DType::F16 => "extract_column_f16",
        DType::BF16 => "extract_column_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "extract_column",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(m)?;
    let block = (BLOCK_SIZE, 1, 1);
    let m_u32 = m as u32;
    let n_cols_u32 = n_cols as u32;
    let col_u32 = col as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&matrix_ptr);
    builder.arg(&col_out_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_cols_u32);
    builder.arg(&col_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!("CUDA extract_column kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Launch optimized matrix transpose kernel using shared memory tiling.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [rows, cols] matrix
/// - `output_ptr` must point to allocated [cols, rows] matrix
pub unsafe fn launch_transpose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "transpose_f32",
        DType::F64 => "transpose_f64",
        DType::F16 => "transpose_f16",
        DType::BF16 => "transpose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "transpose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    const TILE_DIM: usize = 32;
    const BLOCK_ROWS: usize = 8;

    let grid_x = (cols + TILE_DIM - 1) / TILE_DIM;
    let grid_y = (rows + TILE_DIM - 1) / TILE_DIM;
    let cfg = launch_config(
        (grid_x as u32, grid_y as u32, 1),
        (TILE_DIM as u32, BLOCK_ROWS as u32, 1),
        0,
    );

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&rows_u32);
    builder.arg(&cols_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA transpose kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch Kronecker product kernel: out = A ⊗ B
///
/// # Safety
///
/// - `a_ptr` must point to a valid [m_a, n_a] matrix
/// - `b_ptr` must point to a valid [m_b, n_b] matrix
/// - `out_ptr` must have space for m_a * m_b * n_a * n_b elements
pub unsafe fn launch_kron(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    m_a: usize,
    n_a: usize,
    m_b: usize,
    n_b: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "kron_f32",
        DType::F64 => "kron_f64",
        DType::F16 => "kron_f16",
        DType::BF16 => "kron_bf16",
        _ => return Err(Error::UnsupportedDType { dtype, op: "kron" }),
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = m_a * m_b * n_a * n_b;
    let grid = elementwise_launch_config(total)?;
    let block = (BLOCK_SIZE, 1, 1);

    let m_a_u32 = m_a as u32;
    let n_a_u32 = n_a as u32;
    let m_b_u32 = m_b as u32;
    let n_b_u32 = n_b as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_ptr);
    builder.arg(&b_ptr);
    builder.arg(&out_ptr);
    builder.arg(&m_a_u32);
    builder.arg(&n_a_u32);
    builder.arg(&m_b_u32);
    builder.arg(&n_b_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA kron kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch Khatri-Rao product kernel: out = A ⊙ B (column-wise Kronecker)
///
/// # Safety
///
/// - `a_ptr` must point to a valid [m, k] matrix
/// - `b_ptr` must point to a valid [n, k] matrix
/// - `out_ptr` must have space for m * n * k elements
pub unsafe fn launch_khatri_rao(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BASIC_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "khatri_rao_f32",
        DType::F64 => "khatri_rao_f64",
        DType::F16 => "khatri_rao_f16",
        DType::BF16 => "khatri_rao_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "khatri_rao",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = m * n * k;
    let grid = elementwise_launch_config(total)?;
    let block = (BLOCK_SIZE, 1, 1);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_ptr);
    builder.arg(&b_ptr);
    builder.arg(&out_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&k_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA khatri_rao kernel launch failed: {:?}", e)))?;

    Ok(())
}
