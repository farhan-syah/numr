//! GPU-resident QR solve for WebGPU (F32 only)
//!
//! Solves A*x = b using precomputed QR factors entirely on GPU.
//! No CPU↔GPU data transfers except final result retrieval by the caller.
//!
//! Steps:
//! 1. Q^T * b: apply Householder reflectors via `apply_reflector` shaders
//! 2. R \ (Q^T b): level-scheduled upper triangular solve on GPU
//! 3. Column permutation: permutation shader with inverse permutation

#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::levels::{compute_levels_csc_upper, flatten_levels};
#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::qr::cpu::helpers::h_offset;
#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::qr::types::QrFactors;
#[cfg(feature = "wgpu")]
use crate::dtype::DType;
#[cfg(feature = "wgpu")]
use crate::error::{Error, Result};
#[cfg(feature = "wgpu")]
use crate::runtime::wgpu::client::get_buffer;
#[cfg(feature = "wgpu")]
use crate::runtime::wgpu::shaders::{LayoutKey, workgroup_count};
#[cfg(feature = "wgpu")]
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
#[cfg(feature = "wgpu")]
use crate::tensor::Tensor;
#[cfg(feature = "wgpu")]
use wgpu::{BufferDescriptor, BufferUsages};

/// Solve A*x = b using precomputed QR factors, fully on GPU (F32 only).
///
/// Requires `factors.gpu_householder_values` and `factors.gpu_tau` to be populated
/// (they are set automatically by `sparse_qr_wgpu`).
#[cfg(feature = "wgpu")]
pub fn sparse_qr_solve_wgpu(
    client: &WgpuClient,
    factors: &QrFactors<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr_solve_wgpu",
        });
    }

    let gpu_h = factors.gpu_householder_values.as_ref().ok_or_else(|| {
        Error::Internal("sparse_qr_solve_wgpu: GPU Householder vectors not available".to_string())
    })?;
    let gpu_tau = factors.gpu_tau.as_ref().ok_or_else(|| {
        Error::Internal("sparse_qr_solve_wgpu: GPU tau not available".to_string())
    })?;

    let min_mn = m.min(n);
    let device = b.device();
    let wgpu_device = &client.wgpu_device;
    let queue = &client.queue;
    let cache = &client.pipeline_cache;
    let elem_size: u64 = 4;

    let shader_source = include_str!("../../../../runtime/wgpu/shaders/sparse_linalg.wgsl");

    // Get GPU buffers
    let h_buf = get_buffer(gpu_h.ptr())
        .ok_or_else(|| Error::Internal("Invalid h_values buffer".to_string()))?;
    let tau_buf = get_buffer(gpu_tau.ptr())
        .ok_or_else(|| Error::Internal("Invalid tau buffer".to_string()))?;

    // Copy b into work buffer (GPU-to-GPU)
    let work = b.clone();
    let work_buf =
        get_buffer(work.ptr()).ok_or_else(|| Error::Internal("Invalid work buffer".to_string()))?;

    // ========================================================================
    // Step 1: Apply Q^T via Householder reflectors
    // ========================================================================
    let make = |name: &str, entry: &str, num_storage: u32, num_readonly: u32| {
        let module = cache.get_or_create_module_from_source(name, shader_source);
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: num_storage,
            num_uniform_buffers: 1,
            num_readonly_storage: num_readonly,
        });
        let pipeline = cache.get_or_create_dynamic_pipeline(name, entry, &module, &layout);
        (pipeline, layout)
    };

    let (reflector_pipeline, reflector_layout) =
        make("sparse_qr_reflector", "sparse_qr_apply_reflector_f32", 3, 2);

    // Temp buffer for scalar tau value
    let tau_scalar_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_solve_tau_scalar"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffer for reflector params
    let reflector_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_solve_reflector_params"),
        size: 8,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for k in 0..min_mn {
        // Copy tau[k] to scalar buffer
        let tau_byte_offset = (k as u64) * elem_size;
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&tau_buf, tau_byte_offset, &tau_scalar_buf, 0, 4);
        queue.submit(std::iter::once(enc.finish()));

        // Copy v sub-range to temp buffer
        let v_byte_offset = (h_offset(k, m) as u64) * elem_size;
        let v_len = m - k;
        let v_byte_len = (v_len as u64) * elem_size;

        let v_temp_buf = wgpu_device.create_buffer(&BufferDescriptor {
            label: Some("qr_solve_v_temp"),
            size: v_byte_len.max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&h_buf, v_byte_offset, &v_temp_buf, 0, v_byte_len);
        queue.submit(std::iter::once(enc.finish()));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ReflectorParams {
            v_start: u32,
            v_len: u32,
        }
        queue.write_buffer(
            &reflector_params_buf,
            0,
            bytemuck::bytes_of(&ReflectorParams {
                v_start: k as u32,
                v_len: v_len as u32,
            }),
        );

        let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_solve_reflector_bg"),
            layout: &reflector_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: v_temp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tau_scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: work_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: reflector_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&reflector_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // ========================================================================
    // Step 2: Upper triangular solve R * x = (Q^T b)[0:n]
    // ========================================================================
    let r_col_ptrs: Vec<i64> = factors.r.col_ptrs().to_vec();
    let r_row_indices: Vec<i64> = factors.r.row_indices().to_vec();

    let u_schedule = compute_levels_csc_upper(n, &r_col_ptrs, &r_row_indices)?;
    let (u_level_ptrs, u_level_cols) = flatten_levels(&u_schedule);

    // Upload structure to GPU
    let r_col_ptrs_i32: Vec<i32> = r_col_ptrs.iter().map(|&x| x as i32).collect();
    let r_row_indices_i32: Vec<i32> = r_row_indices.iter().map(|&x| x as i32).collect();
    let r_col_ptrs_gpu =
        Tensor::<WgpuRuntime>::from_slice(&r_col_ptrs_i32, &[r_col_ptrs_i32.len()], &device)?;
    let r_row_indices_gpu =
        Tensor::<WgpuRuntime>::from_slice(&r_row_indices_i32, &[r_row_indices_i32.len()], &device)?;
    let u_level_cols_gpu =
        Tensor::<WgpuRuntime>::from_slice(&u_level_cols, &[u_level_cols.len()], &device)?;

    let r_col_ptrs_buf = get_buffer(r_col_ptrs_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid r_col_ptrs buffer".to_string()))?;
    let r_row_indices_buf = get_buffer(r_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid r_row_indices buffer".to_string()))?;
    let u_level_cols_buf = get_buffer(u_level_cols_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid u_level_cols buffer".to_string()))?;
    let r_values_buf = get_buffer(factors.r.values().ptr())
        .ok_or_else(|| Error::Internal("Invalid r_values buffer".to_string()))?;

    // Find diagonal indices
    let u_diag_gpu = Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &device)?;
    let u_diag_buf = get_buffer(u_diag_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid u_diag buffer".to_string()))?;

    let find_diag_module =
        cache.get_or_create_module_from_source("sparse_find_diag_csc", shader_source);
    let find_diag_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let find_diag_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_find_diag_csc",
        "find_diag_indices_csc_f32",
        &find_diag_module,
        &find_diag_layout,
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct FindDiagParams {
        n: u32,
        _p1: u32,
        _p2: u32,
        _p3: u32,
    }

    let find_diag_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_solve_find_diag_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &find_diag_params_buf,
        0,
        bytemuck::bytes_of(&FindDiagParams {
            n: n as u32,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        }),
    );

    {
        let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_solve_find_diag_bg"),
            layout: &find_diag_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_diag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: find_diag_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&find_diag_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // Level-scheduled upper triangular solve
    let upper_module =
        cache.get_or_create_module_from_source("sparse_trsv_csc_upper", shader_source);
    let upper_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 5,
    });
    let upper_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_trsv_csc_upper",
        "sparse_trsv_csc_upper_level_f32",
        &upper_module,
        &upper_layout,
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct TrsvParams {
        level_offset: u32,
        level_size: u32,
        n: u32,
        _pad: u32,
    }

    let trsv_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_solve_trsv_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for level in 0..u_level_ptrs.len().saturating_sub(1) {
        let level_start = u_level_ptrs[level] as u32;
        let level_end = u_level_ptrs[level + 1] as u32;
        let level_size = level_end - level_start;
        if level_size == 0 {
            continue;
        }

        queue.write_buffer(
            &trsv_params_buf,
            0,
            bytemuck::bytes_of(&TrsvParams {
                level_offset: level_start,
                level_size,
                n: n as u32,
                _pad: 0,
            }),
        );

        let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_solve_trsv_bg"),
            layout: &upper_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_level_cols_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: r_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: r_values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: u_diag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: work_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: trsv_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&upper_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(level_size as usize), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // ========================================================================
    // Step 3: Apply column permutation
    // ========================================================================
    let mut inv_perm = vec![0i32; n];
    for (k, &orig_col) in factors.col_perm.iter().enumerate() {
        inv_perm[orig_col] = k as i32;
    }
    let inv_perm_gpu = Tensor::<WgpuRuntime>::from_slice(&inv_perm, &[n], &device)?;
    let inv_perm_buf = get_buffer(inv_perm_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid inv_perm buffer".to_string()))?;

    let result = Tensor::<WgpuRuntime>::zeros(&[n], dtype, &device)?;
    let result_buf = get_buffer(result.ptr())
        .ok_or_else(|| Error::Internal("Invalid result buffer".to_string()))?;

    let perm_module = cache.get_or_create_module_from_source("sparse_apply_perm", shader_source);
    let perm_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let perm_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_apply_perm",
        "apply_row_perm_f32",
        &perm_module,
        &perm_layout,
    );

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct PermParams {
        n: u32,
        _p1: u32,
        _p2: u32,
        _p3: u32,
    }

    let perm_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_solve_perm_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &perm_params_buf,
        0,
        bytemuck::bytes_of(&PermParams {
            n: n as u32,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        }),
    );

    {
        let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_solve_perm_bg"),
            layout: &perm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: work_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inv_perm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perm_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = wgpu_device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&perm_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroup_count(n), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // Wait for completion
    let _ = wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });

    Ok(result)
}
