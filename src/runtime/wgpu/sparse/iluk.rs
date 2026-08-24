//! WebGPU ILU(k) numeric factorization implementation.
//!
//! The symbolic phase is computed on CPU (inherently sequential, uses HashMaps).
//! This module implements the numeric phase on GPU using level scheduling.

use super::super::{WgpuClient, WgpuRuntime};
use super::common::{split_lu_wgpu, validate_wgpu_dtype};
use super::ilu0::{launch_find_diag_indices, launch_ilu0_level};
use crate::algorithm::sparse_linalg::{
    IluFillLevel, IluMetrics, IlukDecomposition, IlukOptions, IlukSymbolic, compute_levels_ilu,
    flatten_levels, validate_square_sparse,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(k) numeric factorization on WebGPU using precomputed symbolic data.
///
/// The symbolic phase is computed on CPU via `iluk_symbolic_wgpu()`.
/// This function performs the numeric factorization on GPU.
pub fn iluk_numeric_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    symbolic: &IlukSymbolic,
    opts: &IlukOptions,
) -> Result<IlukDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "iluk")?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Build combined LU sparsity pattern from symbolic L and U (CPU-based, acceptable for symbolic preprocessing)
    let (combined_row_ptrs, combined_col_indices, l_map, u_map) =
        build_combined_lu_pattern(symbolic);

    let combined_nnz = combined_col_indices.len();

    // Extract original matrix structure for value initialization.
    // Note: Symbolic preprocessing (level computation) is O(n+nnz) on CPU, which is fine.
    // Data transfers here are only structure metadata (row_ptrs, col_indices), not matrix values.
    let orig_row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let orig_col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule on combined pattern on CPU
    let schedule = compute_levels_ilu(n, &combined_row_ptrs, &combined_col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Convert all indices to i32 on GPU (eliminates manual CPU conversion)
    let level_rows_i32: Vec<i32> = level_rows.iter().map(|&x| x as i32).collect();
    let row_ptrs_i64_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &combined_row_ptrs,
        &[combined_row_ptrs.len()],
        &client.device_id,
    )?;
    let col_indices_i64_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &combined_col_indices,
        &[combined_col_indices.len()],
        &client.device_id,
    )?;

    // Cast i64→i32 on GPU (native WGSL shader, avoids manual conversion)
    let row_ptrs_gpu = super::common::cast_i64_to_i32_gpu(client, &row_ptrs_i64_gpu)?;
    let col_indices_gpu = super::common::cast_i64_to_i32_gpu(client, &col_indices_i64_gpu)?;

    // Create GPU buffer for level rows
    let level_rows_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    )?;

    // Initialize combined values array
    let values_gpu = initialize_combined_values_wgpu(
        client,
        a,
        &orig_row_ptrs,
        &orig_col_indices,
        &combined_row_ptrs,
        &combined_col_indices,
        combined_nnz,
    )?;

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, &client.device_id)?;

    // Find diagonal indices on GPU
    launch_find_diag_indices(
        client,
        &row_ptrs_gpu,
        &col_indices_gpu,
        &diag_indices_gpu,
        n,
    )?;

    // Process each level using ILU(0) kernel (same algorithm, different pattern)
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        launch_ilu0_level(
            client,
            &level_rows_gpu,
            level_start,
            level_size,
            &row_ptrs_gpu,
            &col_indices_gpu,
            &values_gpu,
            &diag_indices_gpu,
            n,
            opts.diagonal_shift as f32,
        )?;
    }

    // Wait for GPU to complete
    client.poll_wait();

    // Split into L and U
    let decomp = split_lu_wgpu(
        client,
        n,
        &combined_row_ptrs,
        &combined_col_indices,
        &values_gpu,
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

/// Combined ILU(k) factorization on WebGPU (symbolic on CPU + numeric on GPU).
pub fn iluk_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    opts: IlukOptions,
) -> Result<IlukDecomposition<WgpuRuntime>> {
    // Symbolic phase on CPU (unavoidable - uses HashMaps)
    let symbolic = iluk_symbolic_wgpu(client, a, opts.fill_level)?;
    iluk_numeric_wgpu(client, a, &symbolic, &opts)
}

/// ILU(k) symbolic factorization (runs on CPU, returns result usable by GPU numeric).
pub fn iluk_symbolic_wgpu(
    _client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
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
fn build_combined_lu_pattern(symbolic: &IlukSymbolic) -> (Vec<i64>, Vec<i64>, Vec<i32>, Vec<i32>) {
    let n = symbolic.n;
    let mut combined_row_ptrs = vec![0i64; n + 1];
    let mut combined_col_indices = Vec::new();
    let mut l_map = Vec::new();
    let mut u_map = Vec::new();

    for i in 0..n {
        let l_start = symbolic.row_ptrs_l[i] as usize;
        let l_end = symbolic.row_ptrs_l[i + 1] as usize;
        let l_cols: Vec<i64> = symbolic.col_indices_l[l_start..l_end].to_vec();

        let u_start = symbolic.row_ptrs_u[i] as usize;
        let u_end = symbolic.row_ptrs_u[i + 1] as usize;
        let u_cols: Vec<i64> = symbolic.col_indices_u[u_start..u_end].to_vec();

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
fn initialize_combined_values_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    orig_row_ptrs: &[i64],
    orig_col_indices: &[i64],
    combined_row_ptrs: &[i64],
    combined_col_indices: &[i64],
    combined_nnz: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let n = orig_row_ptrs.len() - 1;

    // Build mapping from original positions to combined positions
    let mut init_map = vec![-1i32; combined_nnz];

    for i in 0..n {
        let orig_start = orig_row_ptrs[i] as usize;
        let orig_end = orig_row_ptrs[i + 1] as usize;
        let comb_start = combined_row_ptrs[i] as usize;
        let comb_end = combined_row_ptrs[i + 1] as usize;

        for orig_idx in orig_start..orig_end {
            let col = orig_col_indices[orig_idx];

            for comb_idx in comb_start..comb_end {
                if combined_col_indices[comb_idx] == col {
                    init_map[comb_idx] = orig_idx as i32;
                    break;
                }
            }
        }
    }

    // Copy values on CPU for initialization (happens once per factorization)
    let orig_values: Vec<f32> = a.values().to_vec();

    let combined_values_cpu: Vec<f32> = init_map
        .iter()
        .map(|&idx| {
            if idx >= 0 {
                orig_values[idx as usize]
            } else {
                0.0
            }
        })
        .collect();

    Tensor::<WgpuRuntime>::try_from_slice(&combined_values_cpu, &[combined_nnz], &client.device_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
    use crate::runtime::Runtime;

    fn get_client() -> WgpuClient {
        let device = WgpuRuntime::default_device();
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_iluk_symbolic() {
        let client = get_client();
        let device = &client.device_id;

        let row_ptrs =
            Tensor::<WgpuRuntime>::try_from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices = Tensor::<WgpuRuntime>::try_from_slice(
            &[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            &[10],
            device,
        )
        .unwrap();
        let values = Tensor::<WgpuRuntime>::try_from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[10],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [4, 4])
            .expect("CSR creation should succeed");

        let symbolic =
            iluk_symbolic_wgpu(&client, &a, IluFillLevel::Zero).expect("symbolic should succeed");

        assert_eq!(symbolic.n, 4);
        assert_eq!(symbolic.fill_level, IluFillLevel::Zero);
    }

    #[test]
    fn test_iluk_numeric() {
        let client = get_client();
        let device = &client.device_id;

        let row_ptrs =
            Tensor::<WgpuRuntime>::try_from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices = Tensor::<WgpuRuntime>::try_from_slice(
            &[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            &[10],
            device,
        )
        .unwrap();
        let values = Tensor::<WgpuRuntime>::try_from_slice(
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
        let device = &client.device_id;

        let row_ptrs =
            Tensor::<WgpuRuntime>::try_from_slice(&[0i64, 2, 5, 8, 10], &[5], device).unwrap();
        let col_indices = Tensor::<WgpuRuntime>::try_from_slice(
            &[0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            &[10],
            device,
        )
        .unwrap();
        let values = Tensor::<WgpuRuntime>::try_from_slice(
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
        assert!(decomp.metrics.fill_ratio >= 1.0);
    }
}
