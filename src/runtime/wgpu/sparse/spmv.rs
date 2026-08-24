//! Sparse matrix-vector and sparse matrix-matrix multiplication for WebGPU.
//!
//! Implements CSR format SpMV and SpMM using WGSL compute shaders.
//! Uses row-parallel algorithms that don't require atomic operations.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::{launch_csr_spmm, launch_csr_spmv};
use super::super::{WgpuClient, WgpuRuntime};
use super::common::validate_wgpu_dtype;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// SpMV parameters uniform buffer layout.
/// Must match the WGSL struct `SpmvParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpmvParams {
    pub nrows: u32,
    pub ncols: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// SpMM parameters uniform buffer layout.
/// Must match the WGSL struct `SpmmParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpmmParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _pad: u32,
}

impl WgpuClient {
    /// CSR sparse matrix-vector multiplication: y = A * x
    ///
    /// Row-parallel implementation where each GPU thread processes one row.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers [nrows + 1] (I32)
    /// * `col_indices` - CSR column indices [nnz] (I32)
    /// * `values` - CSR values [nnz] (dtype)
    /// * `x` - Dense input vector [ncols] (dtype)
    /// * `shape` - Matrix shape [nrows, ncols]
    ///
    /// # Returns
    ///
    /// Dense output vector [nrows]
    pub(crate) fn spmv_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        let [nrows, ncols] = shape;
        let dtype = values.dtype();

        // Validate dtype is supported by WebGPU sparse operations
        validate_wgpu_dtype(dtype, "spmv_csr")?;

        // Validate input shapes
        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        // Allocate output on GPU
        let y = Tensor::<WgpuRuntime>::try_zeros(&[nrows], dtype, &self.device_id)?;

        // Create params buffer
        let params = SpmvParams {
            nrows: nrows as u32,
            ncols: ncols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.create_uniform_buffer("spmv_params", 16);
        self.write_buffer(
            &params_buffer,
            &[params.nrows, params.ncols, params._pad0, params._pad1],
        );

        // Get buffers
        let row_ptrs_buffer = get_tensor_buffer(row_ptrs)?;
        let col_indices_buffer = get_tensor_buffer(col_indices)?;
        let values_buffer = get_tensor_buffer(values)?;
        let x_buffer = get_tensor_buffer(x)?;
        let y_buffer = get_tensor_buffer(&y)?;

        // Launch kernel
        launch_csr_spmv(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buffer,
            &col_indices_buffer,
            &values_buffer,
            &x_buffer,
            &y_buffer,
            &params_buffer,
            nrows,
            dtype,
        )?;

        Ok(y)
    }

    /// CSR sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// Element-parallel implementation where each thread computes one output element.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers [m + 1] (I32)
    /// * `col_indices` - CSR column indices [nnz] (I32)
    /// * `values` - CSR values [nnz] (dtype)
    /// * `b` - Dense input matrix [k, n] (dtype, row-major)
    /// * `shape` - Sparse matrix shape [m, k]
    ///
    /// # Returns
    ///
    /// Dense output matrix [m, n]
    pub(crate) fn spmm_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        let [m, k] = shape;
        let dtype = values.dtype();

        // Validate dtype is supported by WebGPU sparse operations
        validate_wgpu_dtype(dtype, "spmm_csr")?;

        // Validate B is 2D
        if b.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for SpMM, got {}D",
                b.ndim()
            )));
        }

        let b_shape = b.shape();
        let b_k = b_shape[0];
        let n = b_shape[1];

        // Validate dimensions match
        if b_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                got: vec![b_k],
            });
        }

        // Allocate output on GPU
        let c = Tensor::<WgpuRuntime>::try_zeros(&[m, n], dtype, &self.device_id)?;

        // Create params buffer
        let params = SpmmParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };
        let params_buffer = self.create_uniform_buffer("spmm_params", 16);
        self.write_buffer(&params_buffer, &[params.m, params.k, params.n, params._pad]);

        // Get buffers
        let row_ptrs_buffer = get_tensor_buffer(row_ptrs)?;
        let col_indices_buffer = get_tensor_buffer(col_indices)?;
        let values_buffer = get_tensor_buffer(values)?;
        let b_buffer = get_tensor_buffer(b)?;
        let c_buffer = get_tensor_buffer(&c)?;

        // Launch kernel
        launch_csr_spmm(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buffer,
            &col_indices_buffer,
            &values_buffer,
            &b_buffer,
            &c_buffer,
            &params_buffer,
            m,
            n,
            dtype,
        )?;

        Ok(c)
    }
}
