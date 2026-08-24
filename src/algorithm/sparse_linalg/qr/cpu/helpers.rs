//! Helper functions for sparse QR CPU implementation
//!
//! Data extraction and tensor creation utilities.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Extract values as f64 from CSC matrix (sparse QR requires floating-point)
pub(crate) fn extract_values_f64<R: Runtime<DType = DType>>(a: &CscData<R>) -> Result<Vec<f64>> {
    let dtype = a.values().dtype();
    match dtype {
        DType::F32 => Ok(a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect()),
        DType::F64 => Ok(a.values().to_vec()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr",
        }),
    }
}

/// Extract values as f64 from tensor (sparse QR requires floating-point)
pub(crate) fn extract_values_f64_tensor<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
) -> Result<Vec<f64>> {
    let dtype = t.dtype();
    match dtype {
        DType::F32 => Ok(t.to_vec::<f32>().iter().map(|&x| x as f64).collect()),
        DType::F64 => Ok(t.to_vec()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr",
        }),
    }
}

/// Create R tensor in CSC format
pub(crate) fn create_r_tensor<R: Runtime<DType = DType>>(
    m: usize,
    n: usize,
    r_col_ptrs: &[i64],
    r_row_indices: &[i64],
    r_values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> Result<CscData<R>> {
    match dtype {
        DType::F32 => {
            let vals_f32: Vec<f32> = r_values.iter().map(|&x| x as f32).collect();
            CscData::<R>::from_slices(r_col_ptrs, r_row_indices, &vals_f32, [m, n], device)
        }
        DType::F64 => {
            CscData::<R>::from_slices(r_col_ptrs, r_row_indices, r_values, [m, n], device)
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr",
        }),
    }
}

/// Create a vector tensor from f64 data
pub(crate) fn create_vector_tensor<R: Runtime<DType = DType>>(
    data: &[f64],
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let n = data.len();
    match dtype {
        DType::F32 => {
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            Tensor::<R>::try_from_slice(&data_f32, &[n], device)
        }
        DType::F64 => Tensor::<R>::try_from_slice(data, &[n], device),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr",
        }),
    }
}

/// Compute dense Householder vector offset for reflector k in a flat buffer.
///
/// Reflector k has length (m - k), stored at offset `k*m - k*(k-1)/2`.
/// This packs variable-length vectors contiguously: reflector 0 at offset 0
/// with length m, reflector 1 at offset m with length m-1, etc.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn h_offset(k: usize, m: usize) -> usize {
    k * m - k * (k.wrapping_sub(1)) / 2
}

/// Compute R off-diagonal offset for column k in a flat buffer.
///
/// Column k has k off-diagonal entries, stored at offset `k*(k-1)/2`.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn r_offdiag_offset(k: usize) -> usize {
    k * (k.wrapping_sub(1)) / 2
}

/// Build R factor in CSC format from flat off-diagonal and diagonal buffers.
///
/// Off-diagonal entries for column k are stored at `r_offdiag_offset(k)` with
/// k entries. Diagonal entries are in a separate `diag` array. Near-zero
/// off-diagonal entries are dropped.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn build_r_csc(
    r_offdiag: &[f64],
    diag: &[f64],
    min_mn: usize,
    n: usize,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let mut r_col_ptrs = vec![0i64; n + 1];
    let mut r_row_indices: Vec<i64> = Vec::new();
    let mut r_values: Vec<f64> = Vec::new();

    for k in 0..min_mn {
        let ro = r_offdiag_offset(k);
        for row in 0..k {
            let val = r_offdiag[ro + row];
            if val.abs() > 1e-15 {
                r_row_indices.push(row as i64);
                r_values.push(val);
            }
        }
        r_row_indices.push(k as i64);
        r_values.push(diag[k]);
        r_col_ptrs[k + 1] = r_row_indices.len() as i64;
    }
    for k in min_mn..n {
        r_col_ptrs[k + 1] = r_col_ptrs[min_mn];
    }

    (r_col_ptrs, r_row_indices, r_values)
}

/// Detect numerical rank from R diagonal entries.
///
/// Returns the index of the first diagonal entry whose absolute value is
/// below `rank_tolerance`, or `min_mn` if all entries are above tolerance.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn detect_rank(diag: &[f64], min_mn: usize, rank_tolerance: f64) -> usize {
    for k in 0..min_mn {
        if diag[k].abs() < rank_tolerance {
            return k;
        }
    }
    min_mn
}
