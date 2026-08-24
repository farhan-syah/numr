//! CUDA GPU factorization loop for sparse Householder QR
//!
//! Keeps ALL data on GPU with zero intermediate transfers:
//! 1. Structure (col_ptrs, col_perm) on CPU drives the column loop
//! 2. Matrix values and dense work buffers on GPU
//! 3. Householder vectors stored as dense sub-vectors on GPU (kept GPU-resident)
//! 4. Only R structural data (diag, off-diag) transferred to CPU for CSC construction

use crate::algorithm::sparse_linalg::qr::cpu::helpers::{
    build_r_csc, create_r_tensor, detect_rank, h_offset,
};
use crate::algorithm::sparse_linalg::qr::types::{QrFactors, QrOptions, QrSymbolic};
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_sparse_qr_apply_reflector_f32, launch_sparse_qr_apply_reflector_f64,
    launch_sparse_qr_clear_f32, launch_sparse_qr_clear_f64, launch_sparse_qr_extract_r_f32,
    launch_sparse_qr_extract_r_f64, launch_sparse_qr_householder_f32,
    launch_sparse_qr_householder_f64, launch_sparse_qr_norm_f32, launch_sparse_qr_norm_f64,
    launch_sparse_scatter_f32, launch_sparse_scatter_f64,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Run the GPU factorization for a specific dtype
pub(super) fn run_factorization<T: GpuQrScalar>(
    client: &CudaClient,
    a: &CscData<CudaRuntime>,
    symbolic: &QrSymbolic,
    options: &QrOptions,
    m: usize,
    n: usize,
) -> Result<QrFactors<CudaRuntime>> {
    let dtype = a.values().dtype();
    let min_mn = m.min(n);
    let device = a.values().device();
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();

    // A's row_indices as i32 for CUDA kernels
    let a_row_indices_i32: Vec<i32> = a
        .row_indices()
        .to_vec::<i64>()
        .iter()
        .map(|&x| x as i32)
        .collect();
    let a_row_indices_gpu = Tensor::<CudaRuntime>::try_from_slice(
        &a_row_indices_i32,
        &[a_row_indices_i32.len()],
        &device,
    )?;

    // Pre-compute buffer sizes
    let total_h_size = if min_mn > 0 {
        h_offset(min_mn - 1, m) + (m - (min_mn - 1))
    } else {
        0
    };
    let total_r_offdiag = min_mn * min_mn.saturating_sub(1) / 2;

    // Allocate GPU buffers
    let work_gpu = Tensor::<CudaRuntime>::try_zeros(&[m], dtype, &device)?;
    let h_values_gpu = Tensor::<CudaRuntime>::try_zeros(&[total_h_size.max(1)], dtype, &device)?;
    let tau_gpu = Tensor::<CudaRuntime>::try_zeros(&[min_mn.max(1)], dtype, &device)?;
    let diag_gpu = Tensor::<CudaRuntime>::try_zeros(&[min_mn.max(1)], dtype, &device)?;
    let r_offdiag_gpu =
        Tensor::<CudaRuntime>::try_zeros(&[total_r_offdiag.max(1)], dtype, &device)?;
    let norm_sq_gpu = Tensor::<CudaRuntime>::try_zeros(&[1], dtype, &device)?;

    let context = &client.context;
    let stream = &client.stream;
    let device_index = client.device.index;

    let elem_size = T::ELEM_SIZE as u64;
    let idx_size = std::mem::size_of::<i32>() as u64;

    let work_ptr = work_gpu.ptr();
    let h_values_ptr = h_values_gpu.ptr();
    let tau_ptr = tau_gpu.ptr();
    let diag_ptr = diag_gpu.ptr();
    let r_offdiag_ptr = r_offdiag_gpu.ptr();
    let norm_sq_ptr = norm_sq_gpu.ptr();
    let a_values_ptr = a.values().ptr();
    let a_indices_ptr = a_row_indices_gpu.ptr();

    for k in 0..min_mn {
        // Step 1: Clear work vector
        unsafe { T::launch_clear(context, stream, device_index, work_ptr, m as i32)? };

        // Step 2: Scatter permuted column into work
        let orig_col = symbolic.col_perm[k];
        let a_col_start = col_ptrs[orig_col] as usize;
        let a_col_end = col_ptrs[orig_col + 1] as usize;
        let a_col_nnz = a_col_end - a_col_start;

        if a_col_nnz > 0 {
            let values_offset = a_values_ptr + (a_col_start as u64) * elem_size;
            let indices_offset = a_indices_ptr + (a_col_start as u64) * idx_size;

            unsafe {
                T::launch_scatter(
                    context,
                    stream,
                    device_index,
                    values_offset,
                    indices_offset,
                    work_ptr,
                    a_col_nnz as i32,
                )?;
            }
        }

        // Step 3: Apply previous Householder reflectors
        for j in 0..k {
            let v_offset = h_values_ptr + (h_offset(j, m) as u64) * elem_size;
            let tau_j_ptr = tau_ptr + (j as u64) * elem_size;

            unsafe {
                T::launch_apply_reflector(
                    context,
                    stream,
                    device_index,
                    v_offset,
                    j as i32,
                    (m - j) as i32,
                    tau_j_ptr,
                    work_ptr,
                    m as i32,
                )?;
            }
        }

        // Step 4: Extract R off-diagonal entries (work[0..k])
        if k > 0 {
            let r_out = r_offdiag_ptr
                + (crate::algorithm::sparse_linalg::qr::cpu::helpers::r_offdiag_offset(k) as u64)
                    * elem_size;
            unsafe {
                T::launch_extract_r(context, stream, device_index, work_ptr, k as i32, r_out)?;
            }
        }

        // Step 5: Compute norm ||work[k..m]||^2
        unsafe {
            T::launch_norm(
                context,
                stream,
                device_index,
                work_ptr,
                k as i32,
                (m - k) as i32,
                norm_sq_ptr,
            )?;
        }

        // Step 6: Compute Householder vector
        let h_out = h_values_ptr + (h_offset(k, m) as u64) * elem_size;
        let tau_k_ptr = tau_ptr + (k as u64) * elem_size;
        let diag_k_ptr = diag_ptr + (k as u64) * elem_size;

        unsafe {
            T::launch_householder(
                context,
                stream,
                device_index,
                work_ptr,
                k as i32,
                m as i32,
                norm_sq_ptr,
                h_out,
                tau_k_ptr,
                diag_k_ptr,
            )?;
        }
    }

    // Synchronize
    client
        .stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    // Transfer ONLY R structural data (diag + off-diag) for CSC construction.
    // Householder vectors and tau stay GPU-resident — no GPU→CPU transfer.
    let diag_cpu = T::structural_to_f64(&diag_gpu, min_mn);
    let r_offdiag_cpu = T::structural_to_f64(&r_offdiag_gpu, total_r_offdiag);

    // Build R factor on CPU (small structural data)
    let (r_col_ptrs, r_row_indices, r_values) = build_r_csc(&r_offdiag_cpu, &diag_cpu, min_mn, n);
    let rank = detect_rank(&diag_cpu, min_mn, options.rank_tolerance);
    let r = create_r_tensor::<CudaRuntime>(
        m,
        n,
        &r_col_ptrs,
        &r_row_indices,
        &r_values,
        dtype,
        &device,
    )?;

    Ok(QrFactors {
        // GPU factorization keeps Householder data GPU-resident only.
        // CPU sparse representation is empty; use gpu_householder_values for solve.
        householder_vectors: Vec::new(),
        tau: Vec::new(),
        r,
        col_perm: symbolic.col_perm.clone(),
        rank,
        gpu_householder_values: Some(h_values_gpu),
        gpu_tau: Some(tau_gpu),
    })
}

/// Trait for dtype-specific GPU kernel dispatch.
///
/// Eliminates f32/f64 code duplication by providing a uniform interface
/// to dtype-specific CUDA kernel launchers.
pub(super) trait GpuQrScalar: Sized {
    const ELEM_SIZE: usize;

    unsafe fn launch_clear(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        n: i32,
    ) -> Result<()>;

    unsafe fn launch_scatter(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        values: u64,
        indices: u64,
        work: u64,
        nnz: i32,
    ) -> Result<()>;

    unsafe fn launch_apply_reflector(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        v: u64,
        v_start: i32,
        v_len: i32,
        tau_ptr: u64,
        work: u64,
        m: i32,
    ) -> Result<()>;

    unsafe fn launch_norm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        count: i32,
        result: u64,
    ) -> Result<()>;

    unsafe fn launch_householder(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        m: i32,
        norm_sq: u64,
        out_v: u64,
        out_tau: u64,
        out_diag: u64,
    ) -> Result<()>;

    unsafe fn launch_extract_r(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        count: i32,
        output: u64,
    ) -> Result<()>;

    /// Extract small structural data (diag, off-diag) as f64 for R CSC construction.
    /// Only used for O(n) / O(n²) structural buffers, NOT for large data tensors.
    fn structural_to_f64(tensor: &Tensor<CudaRuntime>, count: usize) -> Vec<f64>;
}

impl GpuQrScalar for f32 {
    const ELEM_SIZE: usize = 4;

    unsafe fn launch_clear(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        n: i32,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_clear_f32(ctx, stream, dev, work, n) }
    }
    unsafe fn launch_scatter(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        values: u64,
        indices: u64,
        work: u64,
        nnz: i32,
    ) -> Result<()> {
        unsafe { launch_sparse_scatter_f32(ctx, stream, dev, values, indices, work, nnz) }
    }
    unsafe fn launch_apply_reflector(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        v: u64,
        v_start: i32,
        v_len: i32,
        tau_ptr: u64,
        work: u64,
        m: i32,
    ) -> Result<()> {
        unsafe {
            launch_sparse_qr_apply_reflector_f32(
                ctx, stream, dev, v, v_start, v_len, tau_ptr, work, m,
            )
        }
    }
    unsafe fn launch_norm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        count: i32,
        result: u64,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_norm_f32(ctx, stream, dev, work, start, count, result) }
    }
    unsafe fn launch_householder(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        m: i32,
        norm_sq: u64,
        out_v: u64,
        out_tau: u64,
        out_diag: u64,
    ) -> Result<()> {
        unsafe {
            launch_sparse_qr_householder_f32(
                ctx, stream, dev, work, start, m, norm_sq, out_v, out_tau, out_diag,
            )
        }
    }
    unsafe fn launch_extract_r(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        count: i32,
        output: u64,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_extract_r_f32(ctx, stream, dev, work, count, output) }
    }
    fn structural_to_f64(tensor: &Tensor<CudaRuntime>, count: usize) -> Vec<f64> {
        if count == 0 {
            return vec![];
        }
        tensor
            .to_vec::<f32>()
            .iter()
            .take(count)
            .map(|&x| x as f64)
            .collect()
    }
}

impl GpuQrScalar for f64 {
    const ELEM_SIZE: usize = 8;

    unsafe fn launch_clear(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        n: i32,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_clear_f64(ctx, stream, dev, work, n) }
    }
    unsafe fn launch_scatter(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        values: u64,
        indices: u64,
        work: u64,
        nnz: i32,
    ) -> Result<()> {
        unsafe { launch_sparse_scatter_f64(ctx, stream, dev, values, indices, work, nnz) }
    }
    unsafe fn launch_apply_reflector(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        v: u64,
        v_start: i32,
        v_len: i32,
        tau_ptr: u64,
        work: u64,
        m: i32,
    ) -> Result<()> {
        unsafe {
            launch_sparse_qr_apply_reflector_f64(
                ctx, stream, dev, v, v_start, v_len, tau_ptr, work, m,
            )
        }
    }
    unsafe fn launch_norm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        count: i32,
        result: u64,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_norm_f64(ctx, stream, dev, work, start, count, result) }
    }
    unsafe fn launch_householder(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        start: i32,
        m: i32,
        norm_sq: u64,
        out_v: u64,
        out_tau: u64,
        out_diag: u64,
    ) -> Result<()> {
        unsafe {
            launch_sparse_qr_householder_f64(
                ctx, stream, dev, work, start, m, norm_sq, out_v, out_tau, out_diag,
            )
        }
    }
    unsafe fn launch_extract_r(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        work: u64,
        count: i32,
        output: u64,
    ) -> Result<()> {
        unsafe { launch_sparse_qr_extract_r_f64(ctx, stream, dev, work, count, output) }
    }
    fn structural_to_f64(tensor: &Tensor<CudaRuntime>, count: usize) -> Vec<f64> {
        if count == 0 {
            return vec![];
        }
        tensor.to_vec::<f64>().iter().copied().take(count).collect()
    }
}
