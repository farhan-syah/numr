//! Native ESC (Exact Symbolic Computation) + Hash Accumulation SpGEMM implementation.
//!
//! This implements the SAME algorithm as the CPU backend for perfect backend parity.
//!
//! Algorithm:
//! Phase 1: Symbolic - Count unique columns per output row using bitmap
//! Phase 2: Numeric - Compute values with pre-sized hash accumulator (no resizing)

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    exclusive_scan_i32_gpu, launch_cast, spgemm_numeric_phase, spgemm_symbolic_phase,
};
use crate::sparse::CsrData;
use crate::tensor::Tensor;
use cudarc::driver::DeviceRepr;
use cudarc::types::CudaTypeName;

/// Public function to be called from the combined trait implementation
pub(super) fn esc_spgemm_csr(
    client: &CudaClient,
    a_csr: &CsrData<CudaRuntime>,
    b_csr: &CsrData<CudaRuntime>,
) -> Result<CsrData<CudaRuntime>> {
    use crate::algorithm::sparse::validate_spgemm_shapes;

    // Validate shapes
    let ([_m, _n], _k) = validate_spgemm_shapes(a_csr.shape, b_csr.shape)?;

    let dtype = a_csr.values.dtype();

    // Dispatch to typed implementation
    crate::dispatch_dtype!(dtype, T => {
        client.sparse_matmul_csr_esc::<T>(
            &a_csr.row_ptrs,
            &a_csr.col_indices,
            &a_csr.values,
            &b_csr.row_ptrs,
            &b_csr.col_indices,
            &b_csr.values,
            a_csr.shape,
            b_csr.shape,
        )
    }, "esc_spgemm")
}

impl CudaClient {
    /// Sparse × Sparse matrix multiplication using native ESC+Hash algorithm
    ///
    /// This uses the SAME algorithm as the CPU backend:
    /// - Phase 1: Symbolic - count NNZ per row using bitmap
    /// - Phase 2: Numeric - compute values with pre-sized hash accumulator
    ///
    /// Backend parity is guaranteed by algorithmic consistency.
    pub(super) fn sparse_matmul_csr_esc<T: Element + CudaTypeName + Copy + DeviceRepr>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        a_shape: [usize; 2],
        b_shape: [usize; 2],
    ) -> Result<crate::sparse::CsrData<CudaRuntime>> {
        use crate::runtime::common::sparse_utils::zero_tolerance;

        let [m, _k] = a_shape;
        let [_, n] = b_shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        // ====================================================================
        // PHASE 1: Symbolic (Count NNZ per output row)
        // ====================================================================
        let row_nnz = unsafe {
            spgemm_symbolic_phase(
                &self.context,
                &self.stream,
                self.device.index,
                device,
                a_row_ptrs,
                a_col_indices,
                b_row_ptrs,
                b_col_indices,
                m,
                n,
            )?
        };

        // Build row_ptrs via exclusive scan
        let (c_row_ptrs_i32, total_nnz_i32) = unsafe {
            exclusive_scan_i32_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                device,
                &row_nnz,
            )?
        };

        // Cast row_ptrs from I32 to I64 (numr internal format)
        let c_row_ptrs = unsafe {
            let output = Tensor::try_zeros(&[m + 1], DType::I64, device)?;
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                DType::I32,
                DType::I64,
                c_row_ptrs_i32.ptr(),
                output.ptr(),
                m + 1,
            )?;
            output
        };

        // Get total NNZ from the last element of row_ptrs
        let total_nnz = total_nnz_i32;

        // Allocate output arrays
        let c_col_indices = Tensor::try_zeros(&[total_nnz], crate::dtype::DType::I64, device)?;
        let c_values = Tensor::try_zeros(&[total_nnz], dtype, device)?;

        // ====================================================================
        // PHASE 2: Numeric (Compute values with pre-sized hash accumulator)
        // ====================================================================
        let threshold = zero_tolerance::<T>();
        unsafe {
            spgemm_numeric_phase::<T>(
                &self.context,
                &self.stream,
                self.device.index,
                a_row_ptrs,
                a_col_indices,
                a_values,
                b_row_ptrs,
                b_col_indices,
                b_values,
                &c_row_ptrs,
                &c_col_indices,
                &c_values,
                m,
                n,
                T::from_f64(threshold),
            )?;
        }

        // Synchronize to ensure kernels complete
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("CUDA stream synchronization failed: {:?}", e)))?;

        Ok(crate::sparse::CsrData::new(
            c_row_ptrs,
            c_col_indices,
            c_values,
            [m, n],
        )?)
    }
}
