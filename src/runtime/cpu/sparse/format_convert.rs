//! Sparse format conversion implementations (COO↔CSR↔CSC) for CPU

use super::CpuRuntime;
use crate::dtype::Element;
use crate::error::Result;
use crate::tensor::Tensor;

pub fn coo_to_csr<T: Element>(
    row_indices: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [nrows, _ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let row_idx: Vec<i64> = row_indices.to_vec();
    let col_idx: Vec<i64> = col_indices.to_vec();
    let vals: Vec<T> = values.to_vec();

    let mut perm: Vec<usize> = (0..nnz).collect();
    perm.sort_by_key(|&i| (row_idx[i], col_idx[i]));

    let mut sorted_col_indices = Vec::with_capacity(nnz);
    let mut sorted_values = Vec::with_capacity(nnz);
    for &i in &perm {
        sorted_col_indices.push(col_idx[i]);
        sorted_values.push(vals[i]);
    }

    let mut row_ptrs = vec![0i64; nrows + 1];
    for &i in &perm {
        let row = row_idx[i] as usize;
        row_ptrs[row + 1] += 1;
    }
    for i in 1..=nrows {
        row_ptrs[i] += row_ptrs[i - 1];
    }

    let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[nrows + 1], device)?;
    let col_indices_tensor = Tensor::from_slice(&sorted_col_indices, &[nnz], device)?;
    let values_tensor = Tensor::from_slice(&sorted_values, &[nnz], device)?;

    Ok((row_ptrs_tensor, col_indices_tensor, values_tensor))
}

pub fn coo_to_csc<T: Element>(
    row_indices: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [_nrows, ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let row_idx: Vec<i64> = row_indices.to_vec();
    let col_idx: Vec<i64> = col_indices.to_vec();
    let vals: Vec<T> = values.to_vec();

    let mut perm: Vec<usize> = (0..nnz).collect();
    perm.sort_by_key(|&i| (col_idx[i], row_idx[i]));

    let mut sorted_row_indices = Vec::with_capacity(nnz);
    let mut sorted_values = Vec::with_capacity(nnz);
    for &i in &perm {
        sorted_row_indices.push(row_idx[i]);
        sorted_values.push(vals[i]);
    }

    let mut col_ptrs = vec![0i64; ncols + 1];
    for &i in &perm {
        let col = col_idx[i] as usize;
        col_ptrs[col + 1] += 1;
    }
    for i in 1..=ncols {
        col_ptrs[i] += col_ptrs[i - 1];
    }

    let col_ptrs_tensor = Tensor::from_slice(&col_ptrs, &[ncols + 1], device)?;
    let row_indices_tensor = Tensor::from_slice(&sorted_row_indices, &[nnz], device)?;
    let values_tensor = Tensor::from_slice(&sorted_values, &[nnz], device)?;

    Ok((col_ptrs_tensor, row_indices_tensor, values_tensor))
}

pub fn csr_to_coo<T: Element>(
    row_ptrs: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [nrows, _ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let ptrs: Vec<i64> = row_ptrs.to_vec();

    let mut row_indices = Vec::with_capacity(nnz);
    for row in 0..nrows {
        let start = ptrs[row] as usize;
        let end = ptrs[row + 1] as usize;
        for _ in start..end {
            row_indices.push(row as i64);
        }
    }

    let row_indices_tensor = Tensor::from_slice(&row_indices, &[nnz], device)?;
    Ok((row_indices_tensor, col_indices.clone(), values.clone()))
}

pub fn csc_to_coo<T: Element>(
    col_ptrs: &Tensor<CpuRuntime>,
    row_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [_nrows, ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let ptrs: Vec<i64> = col_ptrs.to_vec();

    let mut col_indices_vec = Vec::with_capacity(nnz);
    for col in 0..ncols {
        let start = ptrs[col] as usize;
        let end = ptrs[col + 1] as usize;
        for _ in start..end {
            col_indices_vec.push(col as i64);
        }
    }

    let col_indices_tensor = Tensor::from_slice(&col_indices_vec, &[nnz], device)?;
    Ok((row_indices.clone(), col_indices_tensor, values.clone()))
}

pub fn csr_to_csc<T: Element>(
    row_ptrs: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [nrows, ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let row_ptr: Vec<i64> = row_ptrs.to_vec();
    let col_idx: Vec<i64> = col_indices.to_vec();
    let vals: Vec<T> = values.to_vec();

    let mut col_counts = vec![0usize; ncols];
    for &col in &col_idx {
        col_counts[col as usize] += 1;
    }

    let mut col_ptrs = vec![0i64; ncols + 1];
    for col in 0..ncols {
        col_ptrs[col + 1] = col_ptrs[col] + col_counts[col] as i64;
    }

    let mut new_row_indices = vec![0i64; nnz];
    let mut new_values = vec![T::from_f64(0.0); nnz];
    let mut col_positions = col_ptrs[..ncols].to_vec();

    for row in 0..nrows {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for idx in start..end {
            let col = col_idx[idx] as usize;
            let pos = col_positions[col] as usize;
            new_row_indices[pos] = row as i64;
            new_values[pos] = vals[idx];
            col_positions[col] += 1;
        }
    }

    let col_ptrs_tensor = Tensor::from_slice(&col_ptrs, &[ncols + 1], device)?;
    let row_indices_tensor = Tensor::from_slice(&new_row_indices, &[nnz], device)?;
    let values_tensor = Tensor::from_slice(&new_values, &[nnz], device)?;

    Ok((col_ptrs_tensor, row_indices_tensor, values_tensor))
}

pub fn csc_to_csr<T: Element>(
    col_ptrs: &Tensor<CpuRuntime>,
    row_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let [nrows, ncols] = shape;
    let nnz = values.numel();
    let device = values.device();

    let col_ptr: Vec<i64> = col_ptrs.to_vec();
    let row_idx: Vec<i64> = row_indices.to_vec();
    let vals: Vec<T> = values.to_vec();

    let mut row_counts = vec![0usize; nrows];
    for &row in &row_idx {
        row_counts[row as usize] += 1;
    }

    let mut row_ptrs = vec![0i64; nrows + 1];
    for row in 0..nrows {
        row_ptrs[row + 1] = row_ptrs[row] + row_counts[row] as i64;
    }

    let mut new_col_indices = vec![0i64; nnz];
    let mut new_values = vec![T::from_f64(0.0); nnz];
    let mut row_positions = row_ptrs[..nrows].to_vec();

    for col in 0..ncols {
        let start = col_ptr[col] as usize;
        let end = col_ptr[col + 1] as usize;
        for idx in start..end {
            let row = row_idx[idx] as usize;
            let pos = row_positions[row] as usize;
            new_col_indices[pos] = col as i64;
            new_values[pos] = vals[idx];
            row_positions[row] += 1;
        }
    }

    let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[nrows + 1], device)?;
    let col_indices_tensor = Tensor::from_slice(&new_col_indices, &[nnz], device)?;
    let values_tensor = Tensor::from_slice(&new_values, &[nnz], device)?;

    Ok((row_ptrs_tensor, col_indices_tensor, values_tensor))
}

pub fn extract_diagonal_csr<T: Element>(
    row_ptrs: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<Tensor<CpuRuntime>> {
    let [nrows, ncols] = shape;
    let n = nrows.min(ncols);
    let device = values.device();

    if n == 0 {
        return Tensor::empty(&[0], values.dtype(), device);
    }

    let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
    let col_indices_data: Vec<i64> = col_indices.to_vec();
    let values_data: Vec<T> = values.to_vec();

    let mut diag = vec![T::zero(); n];
    for row in 0..n {
        let start = row_ptrs_data[row] as usize;
        let end = row_ptrs_data[row + 1] as usize;
        for pos in start..end {
            if col_indices_data[pos] as usize == row {
                diag[row] = values_data[pos];
                break;
            }
        }
    }

    Tensor::from_slice(&diag, &[n], device)
}
