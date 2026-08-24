//! WebGPU GPU factorization loop for sparse Householder QR
//!
//! F32 only. Same architecture as CUDA: dense Householder vectors on GPU,
//! structure-driven column loop on CPU. Householder vectors and tau stay
//! GPU-resident; only R structural data transferred to CPU for CSC construction.

use wgpu::{BufferDescriptor, BufferUsages};

use crate::algorithm::sparse_linalg::qr::cpu::helpers::{
    build_r_csc, create_r_tensor, detect_rank, h_offset, r_offdiag_offset,
};
use crate::algorithm::sparse_linalg::qr::types::{QrFactors, QrOptions, QrSymbolic};
use crate::error::{Error, Result};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::shaders::{LayoutKey, workgroup_count};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Run the WebGPU factorization for f32
pub(super) fn run_factorization_wgpu(
    client: &WgpuClient,
    a: &CscData<WgpuRuntime>,
    symbolic: &QrSymbolic,
    options: &QrOptions,
) -> Result<QrFactors<WgpuRuntime>> {
    let [m, n] = a.shape;
    let dtype = a.values().dtype();
    let min_mn = m.min(n);
    let device = a.values().device();
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();

    // A's row_indices as i32
    let a_row_indices_i32: Vec<i32> = a
        .row_indices()
        .to_vec::<i64>()
        .iter()
        .map(|&x| x as i32)
        .collect();
    let a_row_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &a_row_indices_i32,
        &[a_row_indices_i32.len()],
        &device,
    )?;

    // Buffer sizes
    let total_h_size = if min_mn > 0 {
        h_offset(min_mn - 1, m) + (m - (min_mn - 1))
    } else {
        0
    };
    let total_r_offdiag = min_mn * min_mn.saturating_sub(1) / 2;

    // Allocate GPU buffers
    let work_gpu = Tensor::<WgpuRuntime>::try_zeros(&[m], dtype, &device)?;
    let h_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[total_h_size.max(1)], dtype, &device)?;
    let tau_gpu = Tensor::<WgpuRuntime>::try_zeros(&[min_mn.max(1)], dtype, &device)?;
    let diag_gpu = Tensor::<WgpuRuntime>::try_zeros(&[min_mn.max(1)], dtype, &device)?;
    let r_offdiag_gpu =
        Tensor::<WgpuRuntime>::try_zeros(&[total_r_offdiag.max(1)], dtype, &device)?;
    let norm_sq_gpu = Tensor::<WgpuRuntime>::try_zeros(&[1], dtype, &device)?;

    // Get buffer references
    let work_buf = get_buffer(work_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid work buffer".to_string()))?;
    let h_values_buf = get_buffer(h_values_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid h_values buffer".to_string()))?;
    let tau_buf = get_buffer(tau_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid tau buffer".to_string()))?;
    let diag_buf = get_buffer(diag_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid diag buffer".to_string()))?;
    let r_offdiag_buf = get_buffer(r_offdiag_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid r_offdiag buffer".to_string()))?;
    let norm_sq_buf = get_buffer(norm_sq_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid norm_sq buffer".to_string()))?;
    let a_values_buf = get_buffer(a.values().ptr())
        .ok_or_else(|| Error::Internal("Invalid A values buffer".to_string()))?;
    let a_indices_buf = get_buffer(a_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid A indices buffer".to_string()))?;

    let cache = &client.pipeline_cache;
    let queue = &client.queue;
    let wgpu_device = &client.wgpu_device;

    let shader_source = include_str!("../../../../runtime/wgpu/shaders/sparse_linalg.wgsl");

    // Create pipelines
    let pipelines = create_pipelines(cache, shader_source);

    // Create reusable uniform buffers
    let uniform_bufs = create_uniform_buffers(wgpu_device);

    // Tau scalar buffer for per-reflector access (WGPU doesn't support buffer offsets)
    let tau_scalar_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("qr_tau_scalar"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let elem_size = 4u64; // f32

    // Column loop
    for k in 0..min_mn {
        dispatch_clear(wgpu_device, queue, &pipelines, &uniform_bufs, &work_buf, m);

        dispatch_scatter(
            wgpu_device,
            queue,
            &pipelines,
            &uniform_bufs,
            &a_values_buf,
            &a_indices_buf,
            &work_buf,
            &col_ptrs,
            &symbolic.col_perm,
            k,
        );

        dispatch_apply_reflectors(
            wgpu_device,
            queue,
            &pipelines,
            &uniform_bufs,
            &h_values_buf,
            &tau_buf,
            &tau_scalar_buf,
            &work_buf,
            k,
            m,
            elem_size,
        );

        dispatch_extract_r(
            wgpu_device,
            queue,
            &pipelines,
            &uniform_bufs,
            &work_buf,
            &r_offdiag_buf,
            k,
            elem_size,
        );

        dispatch_norm(
            wgpu_device,
            queue,
            &pipelines,
            &uniform_bufs,
            &work_buf,
            &norm_sq_buf,
            k,
            m,
        );

        dispatch_householder(
            wgpu_device,
            queue,
            &pipelines,
            &uniform_bufs,
            &work_buf,
            &norm_sq_buf,
            &h_values_buf,
            &tau_buf,
            &diag_buf,
            k,
            m,
            elem_size,
        );
    }

    // Wait for completion
    let _ = wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });

    // Transfer ONLY R structural data (diag + off-diag) for CSC construction.
    // Householder vectors and tau stay GPU-resident — no GPU→CPU transfer.
    let diag_cpu_f32: Vec<f32> = diag_gpu.to_vec();
    let r_offdiag_cpu_f32: Vec<f32> = r_offdiag_gpu.to_vec();

    let diag_cpu: Vec<f64> = diag_cpu_f32
        .iter()
        .take(min_mn)
        .map(|&x| x as f64)
        .collect();
    let r_offdiag_cpu: Vec<f64> = r_offdiag_cpu_f32.iter().map(|&x| x as f64).collect();

    // Build R factor on CPU (small structural data)
    let (r_col_ptrs, r_row_indices, r_values) = build_r_csc(&r_offdiag_cpu, &diag_cpu, min_mn, n);
    let rank = detect_rank(&diag_cpu, min_mn, options.rank_tolerance);
    let r = create_r_tensor::<WgpuRuntime>(
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

// ============================================================================
// Pipeline and buffer setup
// ============================================================================

struct Pipelines {
    scatter: std::sync::Arc<wgpu::ComputePipeline>,
    scatter_layout: std::sync::Arc<wgpu::BindGroupLayout>,
    reflector: std::sync::Arc<wgpu::ComputePipeline>,
    reflector_layout: std::sync::Arc<wgpu::BindGroupLayout>,
    norm: std::sync::Arc<wgpu::ComputePipeline>,
    norm_layout: std::sync::Arc<wgpu::BindGroupLayout>,
    householder: std::sync::Arc<wgpu::ComputePipeline>,
    hh_layout: std::sync::Arc<wgpu::BindGroupLayout>,
    extract_r: std::sync::Arc<wgpu::ComputePipeline>,
    extract_layout: std::sync::Arc<wgpu::BindGroupLayout>,
    clear: std::sync::Arc<wgpu::ComputePipeline>,
    clear_layout: std::sync::Arc<wgpu::BindGroupLayout>,
}

struct UniformBuffers {
    scatter: wgpu::Buffer,
    reflector: wgpu::Buffer,
    norm: wgpu::Buffer,
    householder: wgpu::Buffer,
    extract_r: wgpu::Buffer,
    clear: wgpu::Buffer,
}

fn create_pipelines(
    cache: &crate::runtime::wgpu::shaders::PipelineCache,
    shader_source: &str,
) -> Pipelines {
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

    let (scatter, scatter_layout) = make("sparse_qr_scatter", "sparse_scatter_offset_f32", 3, 2);
    let (reflector, reflector_layout) =
        make("sparse_qr_reflector", "sparse_qr_apply_reflector_f32", 3, 2);
    let (norm, norm_layout) = make("sparse_qr_norm", "sparse_qr_norm_f32", 2, 1);
    let (householder, hh_layout) = make("sparse_qr_householder", "sparse_qr_householder_f32", 5, 2);
    let (extract_r, extract_layout) = make("sparse_qr_extract", "sparse_qr_extract_r_f32", 2, 1);
    let (clear, clear_layout) = make("sparse_qr_clear", "sparse_qr_clear_f32", 1, 0);

    Pipelines {
        scatter,
        scatter_layout,
        reflector,
        reflector_layout,
        norm,
        norm_layout,
        householder,
        hh_layout,
        extract_r,
        extract_layout,
        clear,
        clear_layout,
    }
}

fn create_uniform_buffers(dev: &wgpu::Device) -> UniformBuffers {
    let make = |label| {
        dev.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    UniformBuffers {
        scatter: make("qr_scatter_params"),
        reflector: make("qr_reflector_params"),
        norm: make("qr_norm_params"),
        householder: make("qr_hh_params"),
        extract_r: make("qr_extract_params"),
        clear: make("qr_clear_params"),
    }
}

fn dispatch_clear(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    work_buf: &wgpu::Buffer,
    m: usize,
) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n: u32,
        _alignment: u32,
    }
    queue.write_buffer(
        &u.clear,
        0,
        bytemuck::bytes_of(&Params {
            n: m as u32,
            _alignment: 0,
        }),
    );

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qr_clear_bg"),
        layout: &p.clear_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.clear.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.clear);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

fn dispatch_scatter(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    a_values_buf: &wgpu::Buffer,
    a_indices_buf: &wgpu::Buffer,
    work_buf: &wgpu::Buffer,
    col_ptrs: &[i64],
    col_perm: &[usize],
    k: usize,
) {
    let orig_col = col_perm[k];
    let a_col_start = col_ptrs[orig_col] as u32;
    let a_col_end = col_ptrs[orig_col + 1] as u32;
    let a_col_nnz = a_col_end - a_col_start;

    if a_col_nnz == 0 {
        return;
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        offset: u32,
        count: u32,
    }
    queue.write_buffer(
        &u.scatter,
        0,
        bytemuck::bytes_of(&Params {
            offset: a_col_start,
            count: a_col_nnz,
        }),
    );

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qr_scatter_bg"),
        layout: &p.scatter_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_values_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: work_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: u.scatter.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.scatter);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroup_count(a_col_nnz as usize), 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_apply_reflectors(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    h_values_buf: &wgpu::Buffer,
    tau_buf: &wgpu::Buffer,
    tau_scalar_buf: &wgpu::Buffer,
    work_buf: &wgpu::Buffer,
    k: usize,
    m: usize,
    elem_size: u64,
) {
    for j in 0..k {
        // Copy tau[j] to scalar buffer (GPU-to-GPU)
        let tau_byte_offset = (j as u64) * elem_size;
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(tau_buf, tau_byte_offset, tau_scalar_buf, 0, 4);
        queue.submit(std::iter::once(enc.finish()));

        // Extract v sub-range into temp buffer (GPU-to-GPU copy, not CPU transfer)
        let v_byte_offset = (h_offset(j, m) as u64) * elem_size;
        let v_len = m - j;
        let v_byte_len = (v_len as u64) * elem_size;

        let v_temp_buf = dev.create_buffer(&BufferDescriptor {
            label: Some("qr_v_temp"),
            size: v_byte_len.max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(h_values_buf, v_byte_offset, &v_temp_buf, 0, v_byte_len);
        queue.submit(std::iter::once(enc.finish()));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            v_start: u32,
            v_len: u32,
        }
        queue.write_buffer(
            &u.reflector,
            0,
            bytemuck::bytes_of(&Params {
                v_start: j as u32,
                v_len: v_len as u32,
            }),
        );

        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qr_reflector_bg"),
            layout: &p.reflector_layout,
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
                    resource: u.reflector.as_entire_binding(),
                },
            ],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p.reflector);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }
}

fn dispatch_extract_r(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    work_buf: &wgpu::Buffer,
    r_offdiag_buf: &wgpu::Buffer,
    k: usize,
    elem_size: u64,
) {
    if k == 0 {
        return;
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        count: u32,
        _alignment: u32,
    }
    queue.write_buffer(
        &u.extract_r,
        0,
        bytemuck::bytes_of(&Params {
            count: k as u32,
            _alignment: 0,
        }),
    );

    let r_byte_offset = (r_offdiag_offset(k) as u64) * elem_size;
    let r_byte_len = (k as u64) * elem_size;
    let r_temp_buf = dev.create_buffer(&BufferDescriptor {
        label: Some("qr_r_temp"),
        size: r_byte_len.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qr_extract_bg"),
        layout: &p.extract_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: r_temp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: u.extract_r.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.extract_r);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(workgroup_count(k), 1, 1);
    }
    enc.copy_buffer_to_buffer(&r_temp_buf, 0, r_offdiag_buf, r_byte_offset, r_byte_len);
    queue.submit(std::iter::once(enc.finish()));
}

fn dispatch_norm(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    work_buf: &wgpu::Buffer,
    norm_sq_buf: &wgpu::Buffer,
    k: usize,
    m: usize,
) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        start: u32,
        count: u32,
    }
    queue.write_buffer(
        &u.norm,
        0,
        bytemuck::bytes_of(&Params {
            start: k as u32,
            count: (m - k) as u32,
        }),
    );

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qr_norm_bg"),
        layout: &p.norm_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: norm_sq_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: u.norm.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.norm);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_householder(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    p: &Pipelines,
    u: &UniformBuffers,
    work_buf: &wgpu::Buffer,
    norm_sq_buf: &wgpu::Buffer,
    h_values_buf: &wgpu::Buffer,
    tau_buf: &wgpu::Buffer,
    diag_buf: &wgpu::Buffer,
    k: usize,
    m: usize,
    elem_size: u64,
) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        start: u32,
        m: u32,
    }
    queue.write_buffer(
        &u.householder,
        0,
        bytemuck::bytes_of(&Params {
            start: k as u32,
            m: m as u32,
        }),
    );

    let v_len = m - k;
    let v_byte_len = (v_len as u64) * elem_size;
    let v_byte_offset = (h_offset(k, m) as u64) * elem_size;

    let v_out_buf = dev.create_buffer(&BufferDescriptor {
        label: Some("qr_hh_v_out"),
        size: v_byte_len.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tau_out_buf = dev.create_buffer(&BufferDescriptor {
        label: Some("qr_hh_tau_out"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let diag_out_buf = dev.create_buffer(&BufferDescriptor {
        label: Some("qr_hh_diag_out"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qr_hh_bg"),
        layout: &p.hh_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: norm_sq_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tau_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: diag_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: u.householder.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.householder);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&v_out_buf, 0, h_values_buf, v_byte_offset, v_byte_len);
    enc.copy_buffer_to_buffer(&tau_out_buf, 0, tau_buf, (k as u64) * elem_size, 4);
    enc.copy_buffer_to_buffer(&diag_out_buf, 0, diag_buf, (k as u64) * elem_size, 4);
    queue.submit(std::iter::once(enc.finish()));
}
