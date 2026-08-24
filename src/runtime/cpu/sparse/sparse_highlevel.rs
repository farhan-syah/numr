//! High-level sparse operations (format-agnostic wrappers, reductions, conversions)

use super::{CpuClient, CpuRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::sparse::SparseTensor;
use crate::tensor::Tensor;

/// Sparse-matrix vector multiply (format-agnostic)
pub fn spmv(
    client: &CpuClient,
    a: &SparseTensor<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    use crate::sparse::SparseOps;

    let csr = match a {
        SparseTensor::Csr(data) => data.clone(),
        SparseTensor::Coo(data) => data.to_csr()?,
        SparseTensor::Csc(data) => data.to_csr()?,
    };

    let shape = csr.shape;
    let dtype = csr.values.dtype();

    crate::dispatch_dtype!(dtype, T => {
        client.spmv_csr::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, x, shape)
    }, "spmv")
}

/// Sparse-matrix dense-matrix multiply (format-agnostic)
pub fn spmm(
    client: &CpuClient,
    a: &SparseTensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    use crate::sparse::SparseOps;

    let csr = match a {
        SparseTensor::Csr(data) => data.clone(),
        SparseTensor::Coo(data) => data.to_csr()?,
        SparseTensor::Csc(data) => data.to_csr()?,
    };

    let shape = csr.shape;
    let dtype = csr.values.dtype();

    crate::dispatch_dtype!(dtype, T => {
        client.spmm_csr::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, b, shape)
    }, "spmm")
}

/// Dense-matrix sparse-matrix multiply
pub fn dsmm(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    use crate::algorithm::sparse::SparseAlgorithms;

    let csc = match b {
        SparseTensor::Csc(data) => data.clone(),
        SparseTensor::Coo(data) => data.to_csc()?,
        SparseTensor::Csr(data) => data.to_csc()?,
    };

    client.column_parallel_dsmm(a, &csc)
}

/// Sparse + Sparse (format-agnostic)
pub fn sparse_add(
    a: &SparseTensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<SparseTensor<CpuRuntime>> {
    validate_shapes_and_dtypes(a, b)?;

    let csr_a = to_csr(a)?;
    let csr_b = to_csr(b)?;
    let result = csr_a.add(&csr_b)?;
    Ok(SparseTensor::Csr(result))
}

/// Sparse - Sparse (format-agnostic)
pub fn sparse_sub(
    a: &SparseTensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<SparseTensor<CpuRuntime>> {
    validate_shapes_and_dtypes(a, b)?;

    let csr_a = to_csr(a)?;
    let csr_b = to_csr(b)?;
    let result = csr_a.sub(&csr_b)?;
    Ok(SparseTensor::Csr(result))
}

/// Sparse matmul (SpGEMM)
pub fn sparse_matmul(
    client: &CpuClient,
    a: &SparseTensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<SparseTensor<CpuRuntime>> {
    use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};

    let csr_a = to_csr(a)?;
    let csr_b = to_csr(b)?;
    validate_dtype_match(csr_a.values.dtype(), csr_b.values.dtype())?;
    let result_csr = client.esc_spgemm_csr(&csr_a, &csr_b)?;
    Ok(SparseTensor::Csr(result_csr))
}

/// Element-wise sparse multiply
pub fn sparse_mul(
    a: &SparseTensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<SparseTensor<CpuRuntime>> {
    validate_shapes_and_dtypes(a, b)?;

    let csr_a = to_csr(a)?;
    let csr_b = to_csr(b)?;
    let result = csr_a.mul(&csr_b)?;
    Ok(SparseTensor::Csr(result))
}

/// Scale sparse tensor by scalar
pub fn sparse_scale(
    client: &CpuClient,
    a: &SparseTensor<CpuRuntime>,
    scalar: f64,
) -> Result<SparseTensor<CpuRuntime>> {
    use crate::ops::ScalarOps;

    if a.nnz() == 0 {
        return Ok(a.clone());
    }

    match a {
        SparseTensor::Csr(data) => {
            let scaled_values = client.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Csr(crate::sparse::CsrData {
                row_ptrs: data.row_ptrs.clone(),
                col_indices: data.col_indices.clone(),
                values: scaled_values,
                shape: data.shape,
            }))
        }
        SparseTensor::Csc(data) => {
            let scaled_values = client.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Csc(crate::sparse::CscData {
                col_ptrs: data.col_ptrs.clone(),
                row_indices: data.row_indices.clone(),
                values: scaled_values,
                shape: data.shape,
            }))
        }
        SparseTensor::Coo(data) => {
            let scaled_values = client.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Coo(crate::sparse::CooData {
                row_indices: data.row_indices.clone(),
                col_indices: data.col_indices.clone(),
                values: scaled_values,
                shape: data.shape,
                sorted: data.sorted,
            }))
        }
    }
}

/// Sum all non-zero values
pub fn sparse_sum(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let device = values_ref(a).device();

    crate::dispatch_dtype!(dtype, T => {
        let values_vec: Vec<T> = values_ref(a).to_vec();
        let sum: f64 = values_vec.iter().map(|v| v.to_f64()).sum();
        Tensor::from_slice(&[T::from_f64(sum)], &[1], device)
    }, "sparse_sum")
}

/// Sum each row
pub fn sparse_sum_rows(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let csr = to_csr(a)?;
    let [nrows, _] = csr.shape;
    let dtype = csr.values.dtype();
    let device = csr.values.device();

    crate::dispatch_dtype!(dtype, T => {
        let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
        let values: Vec<T> = csr.values.to_vec();
        let mut row_sums: Vec<T> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let start = row_ptrs[row] as usize;
            let end = row_ptrs[row + 1] as usize;
            let sum: f64 = values[start..end].iter().map(|v| v.to_f64()).sum();
            row_sums.push(T::from_f64(sum));
        }
        Tensor::from_slice(&row_sums, &[nrows], device)
    }, "sparse_sum_rows")
}

/// Sum each column
pub fn sparse_sum_cols(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let csc = to_csc(a)?;
    let [_, ncols] = csc.shape;
    let dtype = csc.values.dtype();
    let device = csc.values.device();

    crate::dispatch_dtype!(dtype, T => {
        let col_ptrs: Vec<i64> = csc.col_ptrs.to_vec();
        let values: Vec<T> = csc.values.to_vec();
        let mut col_sums: Vec<T> = Vec::with_capacity(ncols);
        for col in 0..ncols {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;
            let sum: f64 = values[start..end].iter().map(|v| v.to_f64()).sum();
            col_sums.push(T::from_f64(sum));
        }
        Tensor::from_slice(&col_sums, &[ncols], device)
    }, "sparse_sum_cols")
}

/// Non-zeros per row
pub fn sparse_nnz_per_row(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let csr = to_csr(a)?;
    let [nrows, _] = csr.shape;
    let device = csr.values.device();

    let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
    let mut nnz_counts: Vec<i64> = Vec::with_capacity(nrows);
    for row in 0..nrows {
        nnz_counts.push(row_ptrs[row + 1] - row_ptrs[row]);
    }

    Tensor::from_slice(&nnz_counts, &[nrows], device)
}

/// Non-zeros per column
pub fn sparse_nnz_per_col(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let csc = to_csc(a)?;
    let [_, ncols] = csc.shape;
    let device = csc.values.device();

    let col_ptrs: Vec<i64> = csc.col_ptrs.to_vec();
    let mut nnz_counts: Vec<i64> = Vec::with_capacity(ncols);
    for col in 0..ncols {
        nnz_counts.push(col_ptrs[col + 1] - col_ptrs[col]);
    }

    Tensor::from_slice(&nnz_counts, &[ncols], device)
}

/// Convert sparse to dense
pub fn sparse_to_dense(a: &SparseTensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let csr = to_csr(a)?;
    let [nrows, ncols] = csr.shape;
    let dtype = csr.values.dtype();
    let device = csr.values.device();

    crate::dispatch_dtype!(dtype, T => {
        let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
        let col_indices: Vec<i64> = csr.col_indices.to_vec();
        let values: Vec<T> = csr.values.to_vec();
        let mut dense: Vec<T> = vec![T::zero(); nrows * ncols];
        for row in 0..nrows {
            let start = row_ptrs[row] as usize;
            let end = row_ptrs[row + 1] as usize;
            for idx in start..end {
                let col = col_indices[idx] as usize;
                dense[row * ncols + col] = values[idx];
            }
        }
        Tensor::from_slice(&dense, &[nrows, ncols], device)
    }, "sparse_to_dense")
}

/// Convert dense to sparse (COO format)
pub fn dense_to_sparse(a: &Tensor<CpuRuntime>, threshold: f64) -> Result<SparseTensor<CpuRuntime>> {
    if a.ndim() != 2 {
        return Err(Error::Internal(format!(
            "Expected 2D tensor for dense_to_sparse, got {}D",
            a.ndim()
        )));
    }

    let shape_vec = a.shape();
    let nrows = shape_vec[0];
    let ncols = shape_vec[1];
    let dtype = a.dtype();
    let device = a.device();

    crate::dispatch_dtype!(dtype, T => {
        let data: Vec<T> = a.to_vec();
        let mut row_indices: Vec<i64> = Vec::new();
        let mut col_indices: Vec<i64> = Vec::new();
        let mut values: Vec<T> = Vec::new();
        for row in 0..nrows {
            for col in 0..ncols {
                let val = data[row * ncols + col];
                if val.to_f64().abs() >= threshold {
                    row_indices.push(row as i64);
                    col_indices.push(col as i64);
                    values.push(val);
                }
            }
        }
        let mut coo = crate::sparse::CooData::from_slices(
            &row_indices, &col_indices, &values, [nrows, ncols], device,
        )?;
        // SAFETY: row-major iteration guarantees sorted output
        unsafe { coo.set_sorted(true); }
        Ok(SparseTensor::Coo(coo))
    }, "dense_to_sparse")
}

/// Transpose sparse tensor
pub fn sparse_transpose(a: &SparseTensor<CpuRuntime>) -> Result<SparseTensor<CpuRuntime>> {
    match a {
        SparseTensor::Csr(data) => {
            let csc = data.to_csc()?;
            Ok(SparseTensor::Csc(csc))
        }
        SparseTensor::Csc(data) => {
            let csr = data.to_csr()?;
            Ok(SparseTensor::Csr(csr))
        }
        SparseTensor::Coo(data) => {
            let [nrows, ncols] = data.shape;
            let transposed = crate::sparse::CooData {
                row_indices: data.col_indices.clone(),
                col_indices: data.row_indices.clone(),
                values: data.values.clone(),
                shape: [ncols, nrows],
                sorted: false,
            };
            Ok(SparseTensor::Coo(transposed))
        }
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn validate_shapes_and_dtypes(
    a: &SparseTensor<CpuRuntime>,
    b: &SparseTensor<CpuRuntime>,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]],
            got: vec![b.shape()[0], b.shape()[1]],
        });
    }
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(())
}

fn to_csr(a: &SparseTensor<CpuRuntime>) -> Result<crate::sparse::CsrData<CpuRuntime>> {
    match a {
        SparseTensor::Csr(data) => Ok(data.clone()),
        SparseTensor::Coo(data) => data.to_csr(),
        SparseTensor::Csc(data) => data.to_csr(),
    }
}

fn to_csc(a: &SparseTensor<CpuRuntime>) -> Result<crate::sparse::CscData<CpuRuntime>> {
    match a {
        SparseTensor::Csc(data) => Ok(data.clone()),
        SparseTensor::Coo(data) => data.to_csc(),
        SparseTensor::Csr(data) => data.to_csc(),
    }
}

fn values_ref(a: &SparseTensor<CpuRuntime>) -> &Tensor<CpuRuntime> {
    match a {
        SparseTensor::Csr(data) => &data.values,
        SparseTensor::Csc(data) => &data.values,
        SparseTensor::Coo(data) => &data.values,
    }
}
