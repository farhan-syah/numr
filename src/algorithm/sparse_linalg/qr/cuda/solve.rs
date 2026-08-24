//! GPU-resident QR solve for CUDA
//!
//! Solves A*x = b using precomputed QR factors entirely on GPU.
//! No CPU↔GPU data transfers except final result retrieval by the caller.
//!
//! Steps:
//! 1. Q^T * b: apply Householder reflectors via `apply_reflector` kernels
//! 2. R \ (Q^T b): level-scheduled upper triangular solve on GPU
//! 3. Column permutation: scatter kernel with inverse permutation

use crate::algorithm::sparse_linalg::qr::cpu::helpers::h_offset;
use crate::algorithm::sparse_linalg::qr::types::QrFactors;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_apply_row_perm_f32, launch_apply_row_perm_f64, launch_find_diag_indices_csc,
    launch_sparse_qr_apply_reflector_f32, launch_sparse_qr_apply_reflector_f64,
    launch_sparse_trsv_csc_upper_level_f32, launch_sparse_trsv_csc_upper_level_f64,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Solve A*x = b using precomputed QR factors, fully on GPU.
///
/// Requires `factors.gpu_householder_values` and `factors.gpu_tau` to be populated
/// (they are set automatically by `sparse_qr_cuda`).
pub fn sparse_qr_solve_cuda(
    client: &CudaClient,
    factors: &QrFactors<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let [m, n] = factors.r.shape;
    let b_shape = b.shape();

    if b_shape.is_empty() || b_shape[0] != m {
        return Err(Error::ShapeMismatch {
            expected: vec![m],
            got: b_shape.to_vec(),
        });
    }

    if factors.rank < n {
        return Err(Error::Internal(format!(
            "sparse_qr_solve: matrix is rank-deficient (rank {} < n {})",
            factors.rank, n
        )));
    }

    let dtype = b.dtype();
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr_solve_cuda",
        });
    }

    let gpu_h = factors.gpu_householder_values.as_ref().ok_or_else(|| {
        Error::Internal("sparse_qr_solve_cuda: GPU Householder vectors not available".to_string())
    })?;
    let gpu_tau = factors.gpu_tau.as_ref().ok_or_else(|| {
        Error::Internal("sparse_qr_solve_cuda: GPU tau not available".to_string())
    })?;

    match dtype {
        DType::F32 => solve_impl::<f32>(client, factors, b, gpu_h, gpu_tau, m, n),
        DType::F64 => solve_impl::<f64>(client, factors, b, gpu_h, gpu_tau, m, n),
        _ => unreachable!(),
    }
}

trait SolveScalar: Sized {
    const ELEM_SIZE: usize;

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

    unsafe fn launch_trsv_upper_level(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        level_cols: u64,
        level_size: i32,
        col_ptrs: u64,
        row_indices: u64,
        values: u64,
        diag_ptr: u64,
        x: u64,
        n: i32,
    ) -> Result<()>;

    unsafe fn launch_perm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        b: u64,
        perm: u64,
        y: u64,
        n: i32,
    ) -> Result<()>;
}

impl SolveScalar for f32 {
    const ELEM_SIZE: usize = 4;

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

    unsafe fn launch_trsv_upper_level(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        level_cols: u64,
        level_size: i32,
        col_ptrs: u64,
        row_indices: u64,
        values: u64,
        diag_ptr: u64,
        x: u64,
        n: i32,
    ) -> Result<()> {
        unsafe {
            launch_sparse_trsv_csc_upper_level_f32(
                ctx,
                stream,
                dev,
                level_cols,
                level_size,
                col_ptrs,
                row_indices,
                values,
                diag_ptr,
                x,
                n,
            )
        }
    }

    unsafe fn launch_perm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        b: u64,
        perm: u64,
        y: u64,
        n: i32,
    ) -> Result<()> {
        unsafe { launch_apply_row_perm_f32(ctx, stream, dev, b, perm, y, n) }
    }
}

impl SolveScalar for f64 {
    const ELEM_SIZE: usize = 8;

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

    unsafe fn launch_trsv_upper_level(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        level_cols: u64,
        level_size: i32,
        col_ptrs: u64,
        row_indices: u64,
        values: u64,
        diag_ptr: u64,
        x: u64,
        n: i32,
    ) -> Result<()> {
        unsafe {
            launch_sparse_trsv_csc_upper_level_f64(
                ctx,
                stream,
                dev,
                level_cols,
                level_size,
                col_ptrs,
                row_indices,
                values,
                diag_ptr,
                x,
                n,
            )
        }
    }

    unsafe fn launch_perm(
        ctx: &std::sync::Arc<cudarc::driver::safe::CudaContext>,
        stream: &cudarc::driver::safe::CudaStream,
        dev: usize,
        b: u64,
        perm: u64,
        y: u64,
        n: i32,
    ) -> Result<()> {
        unsafe { launch_apply_row_perm_f64(ctx, stream, dev, b, perm, y, n) }
    }
}

fn solve_impl<T: SolveScalar>(
    client: &CudaClient,
    factors: &QrFactors<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    gpu_h: &Tensor<CudaRuntime>,
    gpu_tau: &Tensor<CudaRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    use crate::algorithm::sparse_linalg::levels::{compute_levels_csc_upper, flatten_levels};

    let min_mn = m.min(n);
    let dtype = b.dtype();
    let device = b.device();
    let context = &client.context;
    let stream = &client.stream;
    let dev = client.device.index;
    let elem_size = T::ELEM_SIZE as u64;

    // ========================================================================
    // Step 1: Copy b into work buffer (GPU-to-GPU)
    // ========================================================================
    let work = b.clone();
    let work_ptr = work.ptr();

    let h_ptr = gpu_h.ptr();
    let tau_ptr = gpu_tau.ptr();

    // ========================================================================
    // Step 2: Apply Q^T by launching reflector kernels (CPU drives loop)
    // ========================================================================
    for k in 0..min_mn {
        let v_offset = h_ptr + (h_offset(k, m) as u64) * elem_size;
        let tau_k_ptr = tau_ptr + (k as u64) * elem_size;

        unsafe {
            T::launch_apply_reflector(
                context,
                stream,
                dev,
                v_offset,
                k as i32,
                (m - k) as i32,
                tau_k_ptr,
                work_ptr,
                m as i32,
            )?;
        }
    }

    // ========================================================================
    // Step 3: Upper triangular solve R * x = (Q^T b)[0:n]
    // ========================================================================
    let r_col_ptrs: Vec<i64> = factors.r.col_ptrs().to_vec();
    let r_row_indices: Vec<i64> = factors.r.row_indices().to_vec();

    let u_schedule = compute_levels_csc_upper(n, &r_col_ptrs, &r_row_indices)?;
    let (u_level_ptrs, u_level_cols) = flatten_levels(&u_schedule);

    // Upload structure to GPU
    let r_col_ptrs_i32: Vec<i32> = r_col_ptrs.iter().map(|&x| x as i32).collect();
    let r_row_indices_i32: Vec<i32> = r_row_indices.iter().map(|&x| x as i32).collect();
    let r_col_ptrs_gpu =
        Tensor::<CudaRuntime>::try_from_slice(&r_col_ptrs_i32, &[r_col_ptrs_i32.len()], &device)?;
    let r_row_indices_gpu = Tensor::<CudaRuntime>::try_from_slice(
        &r_row_indices_i32,
        &[r_row_indices_i32.len()],
        &device,
    )?;
    let u_level_cols_gpu =
        Tensor::<CudaRuntime>::try_from_slice(&u_level_cols, &[u_level_cols.len()], &device)?;

    // Find diagonal indices on GPU
    let u_diag_ptr_gpu = Tensor::<CudaRuntime>::try_zeros(&[n], DType::I32, &device)?;
    unsafe {
        launch_find_diag_indices_csc(
            context,
            stream,
            dev,
            r_col_ptrs_gpu.ptr(),
            r_row_indices_gpu.ptr(),
            u_diag_ptr_gpu.ptr(),
            n as i32,
        )?;
    }

    // Launch level-scheduled upper triangular solve
    // work[0:n] = R^{-1} * work[0:n]
    let idx_size = std::mem::size_of::<i32>() as u64;
    for level in 0..u_level_ptrs.len().saturating_sub(1) {
        let offset = u_level_ptrs[level];
        let size = (u_level_ptrs[level + 1] - u_level_ptrs[level]) as i32;
        if size == 0 {
            continue;
        }

        // Offset the level_cols pointer to point at this level's columns
        let level_cols_ptr = u_level_cols_gpu.ptr() + (offset as u64) * idx_size;

        unsafe {
            T::launch_trsv_upper_level(
                context,
                stream,
                dev,
                level_cols_ptr,
                size,
                r_col_ptrs_gpu.ptr(),
                r_row_indices_gpu.ptr(),
                factors.r.values().ptr(),
                u_diag_ptr_gpu.ptr(),
                work_ptr,
                n as i32,
            )?;
        }
    }

    // ========================================================================
    // Step 4: Apply column permutation x_out[col_perm[k]] = work[k]
    // ========================================================================
    let mut inv_perm = vec![0i32; n];
    for (k, &orig_col) in factors.col_perm.iter().enumerate() {
        inv_perm[orig_col] = k as i32;
    }
    let inv_perm_gpu = Tensor::<CudaRuntime>::try_from_slice(&inv_perm, &[n], &device)?;

    let result = Tensor::<CudaRuntime>::try_zeros(&[n], dtype, &device)?;
    unsafe {
        T::launch_perm(
            context,
            stream,
            dev,
            work_ptr,
            inv_perm_gpu.ptr(),
            result.ptr(),
            n as i32,
        )?;
    }

    client
        .stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    Ok(result)
}
