//! CUDA sparse triangular solve implementation.

use super::super::{CudaClient, CudaRuntime};
use super::common::{
    cast_i64_to_i32_gpu, compute_levels_lower_gpu, compute_levels_upper_gpu, validate_cuda_dtype,
};
use crate::algorithm::sparse_linalg::validate_triangular_solve_dims;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Sparse triangular solve for CUDA.
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs]).
pub fn sparse_solve_triangular_cuda(
    client: &CudaClient,
    l_or_u: &CsrData<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Tensor<CudaRuntime>> {
    let (n, nrhs) = validate_triangular_solve_dims(l_or_u.shape, b.shape())?;
    let dtype = l_or_u.values().dtype();
    validate_cuda_dtype(dtype, "sparse_solve_triangular")?;

    if b.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }

    // Cast CSR structure to i32 on GPU (no CPU transfer of large arrays)
    let row_ptrs_gpu = cast_i64_to_i32_gpu(client, l_or_u.row_ptrs())?;
    let col_indices_gpu = cast_i64_to_i32_gpu(client, l_or_u.col_indices())?;

    // Compute level schedule on GPU
    let (level_ptrs, level_rows_gpu, num_levels) = if lower {
        compute_levels_lower_gpu(client, &row_ptrs_gpu, &col_indices_gpu, n)?
    } else {
        compute_levels_upper_gpu(client, &row_ptrs_gpu, &col_indices_gpu, n)?
    };

    // Allocate output (initialized from b)
    let x = b.clone();

    // Process each level
    for level in 0..num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = (level_end - level_start) as i32;

        if level_size == 0 {
            continue;
        }

        let level_rows_ptr =
            level_rows_gpu.ptr() + (level_start * std::mem::size_of::<i32>()) as u64;

        if nrhs == 1 {
            // Use single RHS kernels for vectors
            if lower {
                launch_trsv_lower(
                    client,
                    level_rows_ptr,
                    level_size,
                    &row_ptrs_gpu,
                    &col_indices_gpu,
                    l_or_u.values(),
                    b,
                    &x,
                    n,
                    unit_diagonal,
                    dtype,
                )?;
            } else {
                launch_trsv_upper(
                    client,
                    level_rows_ptr,
                    level_size,
                    &row_ptrs_gpu,
                    &col_indices_gpu,
                    l_or_u.values(),
                    b,
                    &x,
                    n,
                    dtype,
                )?;
            }
        } else {
            // Use multi-RHS kernels for matrices
            if lower {
                launch_trsv_lower_multi_rhs(
                    client,
                    level_rows_ptr,
                    level_size,
                    nrhs,
                    &row_ptrs_gpu,
                    &col_indices_gpu,
                    l_or_u.values(),
                    b,
                    &x,
                    n,
                    unit_diagonal,
                    dtype,
                )?;
            } else {
                launch_trsv_upper_multi_rhs(
                    client,
                    level_rows_ptr,
                    level_size,
                    nrhs,
                    &row_ptrs_gpu,
                    &col_indices_gpu,
                    l_or_u.values(),
                    b,
                    &x,
                    n,
                    dtype,
                )?;
            }
        }
    }

    client
        .stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    Ok(x)
}

/// Launch lower triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_trsv_lower(
    client: &CudaClient,
    level_rows_ptr: u64,
    level_size: i32,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    x: &Tensor<CudaRuntime>,
    n: usize,
    unit_diagonal: bool,
    dtype: DType,
) -> Result<()> {
    match dtype {
        DType::F32 => unsafe {
            kernels::launch_sparse_trsv_lower_level_f32(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
                unit_diagonal,
            )?;
        },
        DType::F64 => unsafe {
            kernels::launch_sparse_trsv_lower_level_f64(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
                unit_diagonal,
            )?;
        },
        _ => unreachable!(),
    }
    Ok(())
}

/// Launch upper triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_trsv_upper(
    client: &CudaClient,
    level_rows_ptr: u64,
    level_size: i32,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    x: &Tensor<CudaRuntime>,
    n: usize,
    dtype: DType,
) -> Result<()> {
    match dtype {
        DType::F32 => unsafe {
            kernels::launch_sparse_trsv_upper_level_f32(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
            )?;
        },
        DType::F64 => unsafe {
            kernels::launch_sparse_trsv_upper_level_f64(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
            )?;
        },
        _ => unreachable!(),
    }
    Ok(())
}

/// Launch multi-RHS lower triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_trsv_lower_multi_rhs(
    client: &CudaClient,
    level_rows_ptr: u64,
    level_size: i32,
    nrhs: usize,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    x: &Tensor<CudaRuntime>,
    n: usize,
    unit_diagonal: bool,
    dtype: DType,
) -> Result<()> {
    match dtype {
        DType::F32 => unsafe {
            kernels::launch_sparse_trsv_lower_level_multi_rhs_f32(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                nrhs as i32,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
                unit_diagonal,
            )?;
        },
        DType::F64 => unsafe {
            kernels::launch_sparse_trsv_lower_level_multi_rhs_f64(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                nrhs as i32,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
                unit_diagonal,
            )?;
        },
        _ => unreachable!(),
    }
    Ok(())
}

/// Launch multi-RHS upper triangular solve kernel.
#[allow(clippy::too_many_arguments)]
fn launch_trsv_upper_multi_rhs(
    client: &CudaClient,
    level_rows_ptr: u64,
    level_size: i32,
    nrhs: usize,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    x: &Tensor<CudaRuntime>,
    n: usize,
    dtype: DType,
) -> Result<()> {
    match dtype {
        DType::F32 => unsafe {
            kernels::launch_sparse_trsv_upper_level_multi_rhs_f32(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                nrhs as i32,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
            )?;
        },
        DType::F64 => unsafe {
            kernels::launch_sparse_trsv_upper_level_multi_rhs_f64(
                &client.context,
                &client.stream,
                client.device.index,
                level_rows_ptr,
                level_size,
                nrhs as i32,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                x.ptr(),
                n as i32,
            )?;
        },
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
    use crate::runtime::Runtime;

    fn get_client() -> CudaClient {
        let device = CudaRuntime::default_device();
        CudaRuntime::default_client(&device)
    }

    #[test]
    fn test_sparse_solve_triangular_lower() {
        let client = get_client();
        let device = &client.device;

        // Create a simple lower triangular matrix:
        // L = [2 0 0]
        //     [1 3 0]
        //     [0 2 4]
        let row_ptrs =
            Tensor::<CudaRuntime>::try_from_slice(&[0i64, 1, 3, 5], &[4], device).unwrap();
        let col_indices =
            Tensor::<CudaRuntime>::try_from_slice(&[0i64, 0, 1, 1, 2], &[5], device).unwrap();
        let values =
            Tensor::<CudaRuntime>::try_from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device)
                .unwrap();

        let l = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // Solve L*x = b where b = [2, 4, 8]
        let b = Tensor::<CudaRuntime>::try_from_slice(&[2.0f32, 4.0, 8.0], &[3], device).unwrap();

        let x = client
            .sparse_solve_triangular(&l, &b, true, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
        assert!((x_data[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_solve_triangular_upper() {
        let client = get_client();
        let device = &client.device;

        // Create a simple upper triangular matrix:
        // U = [2 1 0]
        //     [0 3 2]
        //     [0 0 4]
        let row_ptrs =
            Tensor::<CudaRuntime>::try_from_slice(&[0i64, 2, 4, 5], &[4], device).unwrap();
        let col_indices =
            Tensor::<CudaRuntime>::try_from_slice(&[0i64, 1, 1, 2, 2], &[5], device).unwrap();
        let values =
            Tensor::<CudaRuntime>::try_from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device)
                .unwrap();

        let u = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // Solve U*x = b
        let b = Tensor::<CudaRuntime>::try_from_slice(&[5.0f32, 7.0, 8.0], &[3], device).unwrap();

        let x = client
            .sparse_solve_triangular(&u, &b, false, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 2.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
        assert!((x_data[2] - 2.0).abs() < 1e-5);
    }
}
