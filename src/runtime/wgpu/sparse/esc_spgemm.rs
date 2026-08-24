//! ESC SpGEMM (Exact Symbolic Computation) - WebGPU implementation
//!
//! Implements the ESC algorithm for Sparse × Sparse matrix multiplication.
//!
//! # Algorithm
//!
//! ## Phase 1: Symbolic (Count NNZ per output row)
//! ```text
//! For each output row i:
//!   Use bitmap to find unique columns in C[i,:]
//!   Store count in row_nnz[i]
//! Build row_ptrs via exclusive scan
//! ```
//!
//! ## Phase 2: Numeric (Compute values with dense accumulator)
//! ```text
//! For each output row i:
//!   Use dense accumulator array (size N)
//!   For each non-zero A[i,k]:
//!     For each non-zero B[k,j]:
//!       Accumulate: accum[j] += A[i,k] * B[k,j]
//!   Filter values below zero_tolerance
//!   Write to output arrays (sorted by column)
//! ```
//!
//! This algorithm matches the CPU/CUDA implementation for backend parity.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::{
    launch_exclusive_scan_i32, launch_spgemm_accumulate, launch_spgemm_scatter,
    launch_spgemm_symbolic,
};
use super::common::validate_wgpu_dtype;
use super::merge::ScanParams;
use crate::algorithm::sparse::validate_spgemm_shapes;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// SpGEMM symbolic parameters uniform buffer layout.
/// Must match the WGSL struct `SymbolicParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpgemmSymbolicParams {
    pub m: u32,
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// SpGEMM numeric pass parameters uniform buffer layout.
/// Must match the WGSL struct `SpgemmParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpgemmParams {
    pub m: u32,
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Public function to be called from the combined trait implementation
pub(super) fn esc_spgemm_csr(
    client: &WgpuClient,
    a_csr: &CsrData<WgpuRuntime>,
    b_csr: &CsrData<WgpuRuntime>,
) -> Result<CsrData<WgpuRuntime>> {
    // Validate shapes
    let ([m, n], _k) = validate_spgemm_shapes(a_csr.shape, b_csr.shape)?;

    let dtype = a_csr.values.dtype();
    let device = a_csr.values.device();

    // Validate dtype
    validate_wgpu_dtype(dtype, "esc_spgemm_csr")?;

    // For large matrices, the dense accumulator approach uses O(M*N) memory
    // which may be prohibitive. Check reasonable limits.
    const MAX_DENSE_ELEMENTS: usize = 64 * 1024 * 1024; // 64M elements (~256MB for f32)
    if m * n > MAX_DENSE_ELEMENTS {
        return Err(Error::Internal(format!(
            "SpGEMM output matrix {}x{} = {} elements exceeds WebGPU limit of {} elements",
            m,
            n,
            m * n,
            MAX_DENSE_ELEMENTS
        )));
    }

    // Convert indices to I32 for WGSL (I64 not fully supported in WGSL)
    let a_row_ptrs_i32 = client.cast(&a_csr.row_ptrs, DType::I32)?;
    let a_col_indices_i32 = client.cast(&a_csr.col_indices, DType::I32)?;
    let b_row_ptrs_i32 = client.cast(&b_csr.row_ptrs, DType::I32)?;
    let b_col_indices_i32 = client.cast(&b_csr.col_indices, DType::I32)?;

    // ========================================================================
    // PHASE 1: Symbolic (Count NNZ per output row)
    // ========================================================================

    // Allocate row_nnz array [M]
    let row_nnz = Tensor::<WgpuRuntime>::zeros(&[m], DType::I32, device)?;

    // Allocate bitmap: M rows × ((N+31)/32) u32 words per row
    let words_per_row = (n + 31) / 32;
    let bitmap_size = (m * words_per_row * 4) as u64; // bytes
    let bitmap = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("spgemm_bitmap"),
        size: bitmap_size.max(4), // Ensure non-zero size
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Zero the bitmap
    if bitmap_size > 0 {
        client
            .wgpu_queue()
            .write_buffer(&bitmap, 0, &vec![0u8; bitmap_size as usize]);
    }

    // Create symbolic params buffer
    let symbolic_params = SpgemmSymbolicParams {
        m: m as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let symbolic_params_buffer = client.create_uniform_buffer("spgemm_symbolic_params", 16);
    client.write_buffer(
        &symbolic_params_buffer,
        &[
            symbolic_params.m,
            symbolic_params.n,
            symbolic_params._pad0,
            symbolic_params._pad1,
        ],
    );

    // Get buffers for symbolic phase
    let a_row_ptrs_buffer = get_tensor_buffer(&a_row_ptrs_i32)?;
    let a_col_indices_buffer = get_tensor_buffer(&a_col_indices_i32)?;
    let b_row_ptrs_buffer = get_tensor_buffer(&b_row_ptrs_i32)?;
    let b_col_indices_buffer = get_tensor_buffer(&b_col_indices_i32)?;
    let row_nnz_buffer = get_tensor_buffer(&row_nnz)?;

    // Launch symbolic phase kernel
    launch_spgemm_symbolic(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_row_ptrs_buffer,
        &a_col_indices_buffer,
        &b_row_ptrs_buffer,
        &b_col_indices_buffer,
        &row_nnz_buffer,
        &symbolic_params_buffer,
        &bitmap,
        m,
        dtype,
    )?;

    // Build row_ptrs via exclusive scan
    let c_row_ptrs_i32 = Tensor::<WgpuRuntime>::zeros(&[m + 1], DType::I32, device)?;
    let scan_params = ScanParams {
        n: m as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let scan_params_buffer = client.create_uniform_buffer("spgemm_scan_params", 16);
    client.write_buffer(
        &scan_params_buffer,
        &[
            scan_params.n,
            scan_params._pad0,
            scan_params._pad1,
            scan_params._pad2,
        ],
    );

    let c_row_ptrs_buffer = get_tensor_buffer(&c_row_ptrs_i32)?;

    launch_exclusive_scan_i32(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &row_nnz_buffer,
        &c_row_ptrs_buffer,
        &scan_params_buffer,
    )?;

    // Read back total NNZ from row_ptrs[m]
    // This requires GPU → CPU synchronization
    let staging = client.create_staging_buffer("total_nnz_staging", 4);
    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("total_nnz_copy"),
            });
    // Copy from row_ptrs[m] (offset = m * sizeof(i32))
    let offset = (m * 4) as u64;
    encoder.copy_buffer_to_buffer(&c_row_ptrs_buffer, offset, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut total_nnz_arr = [0i32; 1];
    client.read_buffer(&staging, &mut total_nnz_arr)?;
    let total_nnz = total_nnz_arr[0];

    // Handle empty result
    if total_nnz == 0 {
        let c_row_ptrs = Tensor::<WgpuRuntime>::zeros(&[m + 1], DType::I64, device)?;
        let c_col_indices = Tensor::<WgpuRuntime>::zeros(&[0], DType::I64, device)?;
        let c_values = Tensor::<WgpuRuntime>::zeros(&[0], dtype, device)?;
        return CsrData::new(c_row_ptrs, c_col_indices, c_values, [m, n]);
    }

    // ========================================================================
    // PHASE 2: Numeric (Compute values with dense accumulator)
    // ========================================================================

    // Allocate output arrays
    let c_col_indices = Tensor::<WgpuRuntime>::zeros(&[total_nnz as usize], DType::I32, device)?;
    let c_values = Tensor::<WgpuRuntime>::zeros(&[total_nnz as usize], dtype, device)?;

    // Allocate dense accumulator and flags arrays [M * N]
    let elem_size = match dtype {
        DType::F32 => 4,
        DType::F16 => 2,
        _ => 4, // default to f32 size
    };
    let accum_size = (m * n * elem_size) as u64;
    let accum = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("spgemm_accum"),
        size: accum_size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let flags_size = (m * n * 4) as u64; // u32 per element
    let flags = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("spgemm_flags"),
        size: flags_size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create params buffer for numeric passes.
    let numeric_params_buffer = client.create_uniform_buffer("spgemm_numeric_params", 16);
    let numeric_params = SpgemmParams {
        m: m as u32,
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
    };
    client.write_buffer(
        &numeric_params_buffer,
        &[
            numeric_params.m,
            numeric_params.n,
            numeric_params._pad0,
            numeric_params._pad1,
        ],
    );

    // Get buffers for numeric phase
    let a_values_buffer = get_tensor_buffer(&a_csr.values)?;
    let b_values_buffer = get_tensor_buffer(&b_csr.values)?;
    let c_col_indices_buffer = get_tensor_buffer(&c_col_indices)?;
    let c_values_buffer = get_tensor_buffer(&c_values)?;

    // Launch numeric phase A: accumulate into dense per-row buffers.
    launch_spgemm_accumulate(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_row_ptrs_buffer,
        &a_col_indices_buffer,
        &a_values_buffer,
        &b_row_ptrs_buffer,
        &b_col_indices_buffer,
        &b_values_buffer,
        &numeric_params_buffer,
        &accum,
        &flags,
        m,
        dtype,
    )?;

    // Launch numeric phase B: scatter compact CSR outputs.
    launch_spgemm_scatter(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &c_row_ptrs_buffer,
        &accum,
        &flags,
        &c_col_indices_buffer,
        &c_values_buffer,
        &numeric_params_buffer,
        m,
        dtype,
    )?;

    // Convert row_ptrs and col_indices back to I64
    let c_row_ptrs = client.cast(&c_row_ptrs_i32, DType::I64)?;
    let c_col_indices_i64 = client.cast(&c_col_indices, DType::I64)?;

    CsrData::new(c_row_ptrs, c_col_indices_i64, c_values, [m, n])
}
