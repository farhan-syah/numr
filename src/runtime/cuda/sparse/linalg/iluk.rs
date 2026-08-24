//! CUDA ILU(k) numeric factorization implementation.
//!
//! The symbolic phase is computed on CPU (inherently sequential, uses HashMaps).
//! This module implements the numeric phase on GPU using level scheduling.

use super::super::{CudaClient, CudaRuntime};
use super::common::{
    cast_i64_to_i32_gpu, compute_levels_lower_gpu, split_lu_cuda, validate_cuda_dtype,
};
use crate::algorithm::sparse_linalg::{
    IluFillLevel, IluMetrics, IlukDecomposition, IlukOptions, IlukSymbolic, validate_square_sparse,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(k) numeric factorization on CUDA using precomputed symbolic data.
///
/// The symbolic phase is computed on CPU via `iluk_symbolic_cpu()`.
/// This function performs the numeric factorization on GPU.
pub fn iluk_numeric_cuda(
    client: &CudaClient,
    a: &CsrData<CudaRuntime>,
    symbolic: &IlukSymbolic,
    opts: &IlukOptions,
) -> Result<IlukDecomposition<CudaRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cuda_dtype(dtype, "iluk")?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Build combined LU sparsity pattern from symbolic L and U
    let (combined_row_ptrs, combined_col_indices, l_map, u_map) =
        build_combined_lu_pattern(symbolic);

    let combined_nnz = combined_col_indices.len();

    // Extract original matrix structure for value initialization
    let orig_row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let orig_col_indices: Vec<i64> = a.col_indices().to_vec();

    // Convert combined pattern to GPU tensors on GPU (no CPU transfer of large arrays)
    let combined_row_ptrs_gpu = Tensor::<CudaRuntime>::from_slice(
        &combined_row_ptrs,
        &[combined_row_ptrs.len()],
        &client.device,
    )?;
    let combined_col_indices_gpu = Tensor::<CudaRuntime>::from_slice(
        &combined_col_indices,
        &[combined_col_indices.len()],
        &client.device,
    )?;
    let row_ptrs_gpu = cast_i64_to_i32_gpu(client, &combined_row_ptrs_gpu)?;
    let col_indices_gpu = cast_i64_to_i32_gpu(client, &combined_col_indices_gpu)?;

    // Compute level schedule on combined pattern on GPU
    let (level_ptrs, level_rows_gpu, num_levels) =
        compute_levels_lower_gpu(client, &row_ptrs_gpu, &col_indices_gpu, n)?;

    let device = &client.device;

    // Initialize combined values array on GPU
    // Start with zeros, then scatter original values to their positions
    let values_gpu = initialize_combined_values_cuda(
        client,
        a,
        &orig_row_ptrs,
        &orig_col_indices,
        &combined_row_ptrs,
        &combined_col_indices,
        combined_nnz,
        dtype,
    )?;

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<CudaRuntime>::zeros(&[n], DType::I32, device)?;

    // Find diagonal indices on GPU
    unsafe {
        kernels::launch_find_diag_indices(
            &client.context,
            &client.stream,
            client.device.index,
            row_ptrs_gpu.ptr(),
            col_indices_gpu.ptr(),
            diag_indices_gpu.ptr(),
            n as i32,
        )?;
    }

    // Process each level using ILU(0) kernel (same algorithm, different pattern)
    for level in 0..num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = (level_end - level_start) as i32;

        if level_size == 0 {
            continue;
        }

        let level_rows_ptr =
            level_rows_gpu.ptr() + (level_start * std::mem::size_of::<i32>()) as u64;

        match dtype {
            DType::F32 => unsafe {
                kernels::launch_ilu0_level_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    level_rows_ptr,
                    level_size,
                    row_ptrs_gpu.ptr(),
                    col_indices_gpu.ptr(),
                    values_gpu.ptr(),
                    diag_indices_gpu.ptr(),
                    n as i32,
                    opts.diagonal_shift as f32,
                )?;
            },
            DType::F64 => unsafe {
                kernels::launch_ilu0_level_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    level_rows_ptr,
                    level_size,
                    row_ptrs_gpu.ptr(),
                    col_indices_gpu.ptr(),
                    values_gpu.ptr(),
                    diag_indices_gpu.ptr(),
                    n as i32,
                    opts.diagonal_shift,
                )?;
            },
            _ => unreachable!(),
        }
    }

    // Synchronize
    client
        .stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    // Split into L and U using the precomputed maps
    let decomp = split_lu_cuda(
        client,
        n,
        &combined_row_ptrs,
        &combined_col_indices,
        &values_gpu,
        dtype,
    )?;

    // Compute metrics
    let original_nnz = a.values().numel();
    let l_nnz = l_map.iter().filter(|&&x| x >= 0).count();
    let u_nnz = u_map.iter().filter(|&&x| x >= 0).count();
    let factored_nnz = l_nnz + u_nnz;

    let metrics = IluMetrics {
        original_nnz,
        factored_nnz,
        fill_ratio: factored_nnz as f64 / original_nnz as f64,
        fill_level: opts.fill_level,
        diagonal_shifts_applied: 0, // GPU doesn't track this
    };

    Ok(IlukDecomposition {
        l: decomp.l,
        u: decomp.u,
        metrics,
    })
}

/// Combined ILU(k) factorization on CUDA (symbolic on CPU + numeric on GPU).
pub fn iluk_cuda(
    client: &CudaClient,
    a: &CsrData<CudaRuntime>,
    opts: IlukOptions,
) -> Result<IlukDecomposition<CudaRuntime>> {
    // Symbolic phase on CPU (unavoidable - uses HashMaps)
    let symbolic = iluk_symbolic_cuda(client, a, opts.fill_level)?;
    iluk_numeric_cuda(client, a, &symbolic, &opts)
}

/// ILU(k) symbolic factorization (runs on CPU, returns result usable by GPU numeric).
pub fn iluk_symbolic_cuda(
    _client: &CudaClient,
    a: &CsrData<CudaRuntime>,
    level: IluFillLevel,
) -> Result<IlukSymbolic> {
    let n = validate_square_sparse(a.shape)?;

    // Extract CSR structure for CPU-based symbolic analysis
    // This transfer is acceptable as symbolic analysis happens once per matrix structure
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Delegate to shared implementation (pure CPU graph analysis)
    crate::algorithm::sparse_linalg::iluk_symbolic_impl(n, &row_ptrs, &col_indices, level)
}

/// Build combined LU sparsity pattern from symbolic L and U patterns.
///
/// Returns:
/// - combined_row_ptrs: CSR row pointers for combined LU
/// - combined_col_indices: CSR column indices for combined LU
/// - l_map: For each combined index, destination in L (-1 if not in L)
/// - u_map: For each combined index, destination in U (-1 if not in U)
fn build_combined_lu_pattern(symbolic: &IlukSymbolic) -> (Vec<i64>, Vec<i64>, Vec<i32>, Vec<i32>) {
    let n = symbolic.n;
    let mut combined_row_ptrs = vec![0i64; n + 1];
    let mut combined_col_indices = Vec::new();
    let mut l_map = Vec::new();
    let mut u_map = Vec::new();

    for i in 0..n {
        // Get L columns for row i (j < i)
        let l_start = symbolic.row_ptrs_l[i] as usize;
        let l_end = symbolic.row_ptrs_l[i + 1] as usize;
        let l_cols: Vec<i64> = symbolic.col_indices_l[l_start..l_end].to_vec();

        // Get U columns for row i (j >= i)
        let u_start = symbolic.row_ptrs_u[i] as usize;
        let u_end = symbolic.row_ptrs_u[i + 1] as usize;
        let u_cols: Vec<i64> = symbolic.col_indices_u[u_start..u_end].to_vec();

        // Merge L and U columns (L columns come first since they're all < i)
        let mut l_idx = 0;
        let mut u_idx = 0;

        while l_idx < l_cols.len() || u_idx < u_cols.len() {
            let l_col = l_cols.get(l_idx).copied();
            let u_col = u_cols.get(u_idx).copied();

            match (l_col, u_col) {
                (Some(lc), Some(uc)) => {
                    if lc < uc {
                        combined_col_indices.push(lc);
                        l_map.push((l_start + l_idx) as i32);
                        u_map.push(-1);
                        l_idx += 1;
                    } else if lc > uc {
                        combined_col_indices.push(uc);
                        l_map.push(-1);
                        u_map.push((u_start + u_idx) as i32);
                        u_idx += 1;
                    } else {
                        // Same column (shouldn't happen since L is strictly lower, U is upper+diag)
                        combined_col_indices.push(lc);
                        l_map.push((l_start + l_idx) as i32);
                        u_map.push((u_start + u_idx) as i32);
                        l_idx += 1;
                        u_idx += 1;
                    }
                }
                (Some(lc), None) => {
                    combined_col_indices.push(lc);
                    l_map.push((l_start + l_idx) as i32);
                    u_map.push(-1);
                    l_idx += 1;
                }
                (None, Some(uc)) => {
                    combined_col_indices.push(uc);
                    l_map.push(-1);
                    u_map.push((u_start + u_idx) as i32);
                    u_idx += 1;
                }
                (None, None) => break,
            }
        }

        combined_row_ptrs[i + 1] = combined_col_indices.len() as i64;
    }

    (combined_row_ptrs, combined_col_indices, l_map, u_map)
}

/// Initialize combined values array from original matrix values.
///
/// Positions that exist in both original and combined get original values.
/// Fill positions (in combined but not in original) get zeros.
#[allow(clippy::too_many_arguments)]
fn initialize_combined_values_cuda(
    client: &CudaClient,
    a: &CsrData<CudaRuntime>,
    orig_row_ptrs: &[i64],
    orig_col_indices: &[i64],
    combined_row_ptrs: &[i64],
    combined_col_indices: &[i64],
    combined_nnz: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let n = orig_row_ptrs.len() - 1;

    // Build mapping from original positions to combined positions
    // For each original entry (i, j), find its position in combined
    let mut init_map = vec![-1i32; combined_nnz];

    for i in 0..n {
        let orig_start = orig_row_ptrs[i] as usize;
        let orig_end = orig_row_ptrs[i + 1] as usize;
        let comb_start = combined_row_ptrs[i] as usize;
        let comb_end = combined_row_ptrs[i + 1] as usize;

        // For each original entry, find matching combined entry
        for orig_idx in orig_start..orig_end {
            let col = orig_col_indices[orig_idx];

            // Binary search in combined row
            for comb_idx in comb_start..comb_end {
                if combined_col_indices[comb_idx] == col {
                    init_map[comb_idx] = orig_idx as i32;
                    break;
                }
            }
        }
    }

    let device = &client.device;

    // Initialize combined values from original matrix on CPU
    // This is acceptable because initialization happens once per factorization
    let orig_values: Vec<f64> = match dtype {
        DType::F32 => a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => a.values().to_vec(),
        _ => unreachable!(),
    };

    let combined_values_cpu: Vec<f64> = init_map
        .iter()
        .map(|&idx| {
            if idx >= 0 {
                orig_values[idx as usize]
            } else {
                0.0
            }
        })
        .collect();

    // Upload to GPU
    let combined_values = match dtype {
        DType::F32 => {
            let vals_f32: Vec<f32> = combined_values_cpu.iter().map(|&x| x as f32).collect();
            Tensor::<CudaRuntime>::from_slice(&vals_f32, &[combined_nnz], device)?
        }
        DType::F64 => {
            Tensor::<CudaRuntime>::from_slice(&combined_values_cpu, &[combined_nnz], device)?
        }
        _ => unreachable!(),
    };

    Ok(combined_values)
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
    fn test_iluk_symbolic() {
        let client = get_client();
        let device = &client.device;

        // Create a simple 4x4 sparse matrix
        let row_ptrs =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3], &[10], device)
                .unwrap();
        let values = Tensor::<CudaRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[10],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [4, 4])
            .expect("CSR creation should succeed");

        let symbolic =
            iluk_symbolic_cuda(&client, &a, IluFillLevel::Zero).expect("symbolic should succeed");

        assert_eq!(symbolic.n, 4);
        assert_eq!(symbolic.fill_level, IluFillLevel::Zero);
    }

    #[test]
    fn test_iluk_numeric() {
        let client = get_client();
        let device = &client.device;

        // Tridiagonal matrix
        let row_ptrs =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3], &[10], device)
                .unwrap();
        let values = Tensor::<CudaRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[10],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [4, 4])
            .expect("CSR creation should succeed");

        let opts = IlukOptions::default();
        let decomp = client.iluk(&a, opts).expect("iluk should succeed");

        assert_eq!(decomp.l.shape, [4, 4]);
        assert_eq!(decomp.u.shape, [4, 4]);
        assert!(decomp.metrics.fill_ratio >= 1.0);
    }

    #[test]
    fn test_iluk_level1() {
        let client = get_client();
        let device = &client.device;

        // Tridiagonal matrix
        let row_ptrs =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3], &[10], device)
                .unwrap();
        let values = Tensor::<CudaRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[10],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [4, 4])
            .expect("CSR creation should succeed");

        let opts = IlukOptions {
            fill_level: IluFillLevel::One,
            ..Default::default()
        };
        let decomp = client.iluk(&a, opts).expect("iluk should succeed");

        assert_eq!(decomp.l.shape, [4, 4]);
        assert_eq!(decomp.u.shape, [4, 4]);
        // ILU(1) should have more or equal fill than ILU(0)
        assert!(decomp.metrics.fill_ratio >= 1.0);
    }
}
