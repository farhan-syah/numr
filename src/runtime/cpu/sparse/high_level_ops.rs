//! SparseOps trait implementation for CPU runtime
//!
//! Thin delegation layer — actual implementations live in:
//! - `format_convert.rs` — COO↔CSR↔CSC format conversions
//! - `sparse_highlevel.rs` — high-level ops, reductions, dense conversion

use super::format_convert;
use super::merge::{
    MergeStrategy, OperationSemantics, intersect_coo_impl, merge_coo_impl, merge_csc_impl,
    merge_csr_impl,
};
use super::sparse_highlevel;
use super::{CpuClient, CpuRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

impl SparseOps<CpuRuntime> for CpuClient {
    // =========================================================================
    // CSR low-level operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        let [nrows, ncols] = shape;
        let device = values.device();

        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<T> = values.to_vec();
        let x_data: Vec<T> = x.to_vec();

        let mut y_data: Vec<T> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let start = row_ptrs_data[row] as usize;
            let end = row_ptrs_data[row + 1] as usize;
            let mut sum: f64 = 0.0;
            for j in start..end {
                let col = col_indices_data[j] as usize;
                sum += values_data[j].to_f64() * x_data[col].to_f64();
            }
            y_data.push(T::from_f64(sum));
        }

        Tensor::try_from_slice(&y_data, &[nrows], device)
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        let [m, k] = shape;
        let device = values.device();

        if b.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for SpMM, got {}D",
                b.ndim()
            )));
        }

        let b_shape = b.shape();
        let b_k = b_shape[0];
        let n = b_shape[1];

        if b_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                got: vec![b_k],
            });
        }

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let a_values: Vec<T> = values.to_vec();
        let b_data: Vec<T> = b.to_vec();

        let mut c_data: Vec<T> = vec![T::zero(); m * n];
        for row in 0..m {
            let start = row_ptrs_data[row] as usize;
            let end = row_ptrs_data[row + 1] as usize;
            for j in start..end {
                let col = col_indices_data[j] as usize;
                let a_val = a_values[j].to_f64();
                for col_b in 0..n {
                    let c_idx = row * n + col_b;
                    let current = c_data[c_idx].to_f64();
                    c_data[c_idx] = T::from_f64(current + a_val * b_data[col * n + col_b].to_f64());
                }
            }
        }

        Tensor::try_from_slice(&c_data, &[m, n], device)
    }

    fn add_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Multiply,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn div_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    // =========================================================================
    // CSC low-level operations
    // =========================================================================

    fn add_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Multiply,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn div_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    // =========================================================================
    // COO low-level operations
    // =========================================================================

    fn add_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_ri,
            a_ci,
            a_v,
            b_ri,
            b_ci,
            b_v,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_ri,
            a_ci,
            a_v,
            b_ri,
            b_ci,
            b_v,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(a_ri, a_ci, a_v, b_ri, b_ci, b_v, |a, b| {
            T::from_f64(a.to_f64() * b.to_f64())
        })
    }

    fn div_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(a_ri, a_ci, a_v, b_ri, b_ci, b_v, |a, b| {
            T::from_f64(a.to_f64() / b.to_f64())
        })
    }

    // =========================================================================
    // High-level operations — delegate to sparse_highlevel module
    // =========================================================================

    fn spmv(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::spmv(self, a, x)
    }

    fn spmm(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::spmm(self, a, b)
    }

    fn dsmm(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::dsmm(self, a, b)
    }

    fn sparse_add(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_add(a, b)
    }

    fn sparse_sub(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_sub(a, b)
    }

    fn sparse_matmul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_matmul(self, a, b)
    }

    fn sparse_mul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_mul(a, b)
    }

    fn sparse_scale(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_scale(self, a, scalar)
    }

    fn sparse_add_scalar(
        &self,
        _a: &crate::sparse::SparseTensor<CpuRuntime>,
        _scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum(a)
    }

    fn sparse_sum_rows(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum_rows(a)
    }

    fn sparse_sum_cols(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum_cols(a)
    }

    fn sparse_nnz_per_row(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_nnz_per_row(a)
    }

    fn sparse_nnz_per_col(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_nnz_per_col(a)
    }

    fn sparse_to_dense(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_to_dense(a)
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<CpuRuntime>,
        threshold: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::dense_to_sparse(a, threshold)
    }

    // =========================================================================
    // Format conversions — delegate to format_convert module
    // =========================================================================

    fn coo_to_csr<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::coo_to_csr::<T>(row_indices, col_indices, values, shape)
    }

    fn coo_to_csc<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::coo_to_csc::<T>(row_indices, col_indices, values, shape)
    }

    fn csr_to_coo<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csr_to_coo::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_coo<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csc_to_coo::<T>(col_ptrs, row_indices, values, shape)
    }

    fn csr_to_csc<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csr_to_csc::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_csr<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csc_to_csr::<T>(col_ptrs, row_indices, values, shape)
    }

    fn extract_diagonal_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        format_convert::extract_diagonal_csr::<T>(row_ptrs, col_indices, values, shape)
    }

    fn sparse_transpose(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_transpose(a)
    }
}
