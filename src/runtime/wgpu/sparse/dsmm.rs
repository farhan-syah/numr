//! Dense × Sparse Matrix Multiplication (DSMM) - WebGPU implementation
//!
//! Implements column-parallel DSMM algorithm for WebGPU backend.
//!
//! # Algorithm
//!
//! Column-parallel CSC iteration (matches CPU/CUDA implementation):
//! ```text
//! For each column j in B (0..N):
//!   For each non-zero B[k,j] in column j:
//!     C[:,j] += A[:,k] * B[k,j]
//! ```
//!
//! WebGPU parallelization:
//! - Each thread computes one element C[row, col]
//! - Thread iterates over sparse column of B to accumulate result
//!
//! This algorithm matches the CPU/CUDA implementation for backend parity.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::launch_dsmm_csc;
use super::common::validate_wgpu_dtype;
use crate::algorithm::sparse::validate_dsmm_shapes;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// DSMM parameters uniform buffer layout.
/// Must match the WGSL struct `DsmmParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DsmmParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _pad: u32,
}

/// Public function to be called from the combined trait implementation
pub(super) fn column_parallel_dsmm(
    client: &WgpuClient,
    dense_a: &Tensor<WgpuRuntime>,
    sparse_b_csc: &CscData<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = dense_a.dtype();
    let device = dense_a.device();

    // Validate dtype
    validate_wgpu_dtype(dtype, "column_parallel_dsmm")?;

    // Validate dimensions
    let ([m, n], k) = validate_dsmm_shapes(dense_a.shape(), sparse_b_csc.shape)?;

    // Ensure A is contiguous for kernel
    let a_contig = dense_a.contiguous()?;

    // Allocate output C [M, N]
    let output = Tensor::<WgpuRuntime>::try_zeros(&[m, n], dtype, device)?;

    // Convert col_ptrs and row_indices from I64 to I32 for WGSL shader
    let col_ptrs_i32 = client.cast(&sparse_b_csc.col_ptrs, DType::I32)?;
    let row_indices_i32 = client.cast(&sparse_b_csc.row_indices, DType::I32)?;

    // Create params buffer
    let params = DsmmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buffer = client.create_uniform_buffer("dsmm_params", 16);
    client.write_buffer(&params_buffer, &[params.m, params.k, params.n, params._pad]);

    // Get buffers
    let a_buffer = get_tensor_buffer(&a_contig)?;
    let col_ptrs_buffer = get_tensor_buffer(&col_ptrs_i32)?;
    let row_indices_buffer = get_tensor_buffer(&row_indices_i32)?;
    let b_values_buffer = get_tensor_buffer(&sparse_b_csc.values)?;
    let c_buffer = get_tensor_buffer(&output)?;

    // Launch WGSL kernel
    launch_dsmm_csc(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buffer,
        &col_ptrs_buffer,
        &row_indices_buffer,
        &b_values_buffer,
        &c_buffer,
        &params_buffer,
        m,
        n,
        dtype,
    )?;

    Ok(output)
}
