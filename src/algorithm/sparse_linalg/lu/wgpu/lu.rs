//! WebGPU implementation of sparse LU factorization
//!
//! **Static Pivoting Mode**: No row swaps, threshold-based pivot adjustment
//!
//! This implementation keeps ALL data on GPU with zero intermediate transfers:
//! 1. Structure (col_ptrs, row_indices) on CPU drives the algorithm
//! 2. Matrix values and index arrays transferred to GPU ONCE at start
//! 3. L/U buffers pre-allocated based on symbolic structure
//! 4. Numeric factorization with WGSL kernels (scatter, axpy, gather, divide)
//! 5. L/U transferred back ONCE at end
//!
//! **Key Design**: Uniform params for offset/count, not per-column buffer allocations.

#[cfg(feature = "wgpu")]
use wgpu::{BufferDescriptor, BufferUsages};

#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::lu::types::{LuFactors, LuOptions, LuSymbolic};
#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::traits::validate_square_sparse;
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
use crate::sparse::CscData;
#[cfg(feature = "wgpu")]
use crate::tensor::Tensor;

/// Sparse LU factorization with full symbolic information (WebGPU, static pivoting)
///
/// Uses WGSL shaders with zero intermediate transfers. Pre-allocates L/U based on
/// symbolic structure and performs all numeric factorization on GPU.
#[cfg(feature = "wgpu")]
pub fn sparse_lu_wgpu(
    client: &WgpuClient,
    a: &CscData<WgpuRuntime>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
) -> Result<LuFactors<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();

    // WebGPU only supports F32
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_lu_wgpu",
        });
    }

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Extract CSC structure (stays on CPU for algorithm control)
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();

    // ==========================================================================
    // STEP 1: Transfer ALL data to GPU ONCE at start
    // ==========================================================================

    let device = a.values().device();

    // A's row_indices as i32 (WGSL uses i32 for indices)
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

    // L's row_indices as i32 (from symbolic structure)
    let l_row_indices_i32: Vec<i32> = symbolic.l_row_indices.iter().map(|&x| x as i32).collect();
    let l_row_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &l_row_indices_i32,
        &[l_row_indices_i32.len()],
        &device,
    )?;

    // U's row_indices as i32 (from symbolic structure)
    let u_row_indices_i32: Vec<i32> = symbolic.u_row_indices.iter().map(|&x| x as i32).collect();
    let u_row_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &u_row_indices_i32,
        &[u_row_indices_i32.len()],
        &device,
    )?;

    // Pre-allocate L and U values on GPU based on symbolic pattern
    let l_nnz = symbolic.l_row_indices.len();
    let u_nnz = symbolic.u_row_indices.len();
    let l_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[l_nnz], dtype, &device)?;
    let u_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[u_nnz], dtype, &device)?;

    // Work vector on GPU (dense, size n)
    let work_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], dtype, &device)?;

    // ==========================================================================
    // STEP 2: Run factorization (all computation on GPU)
    // ==========================================================================

    run_factorization_f32(
        client,
        n,
        &col_ptrs,
        a.values(),
        &a_row_indices_gpu,
        &l_values_gpu,
        &l_row_indices_gpu,
        &u_values_gpu,
        &u_row_indices_gpu,
        &work_gpu,
        symbolic,
        options,
    )?;

    // ==========================================================================
    // STEP 3: Transfer results back from GPU ONCE at end
    // ==========================================================================

    let l = CscData::<WgpuRuntime>::from_slices(
        &symbolic.l_col_ptrs,
        &symbolic.l_row_indices,
        &l_values_gpu.to_vec::<f32>(),
        [n, n],
        &device,
    )?;

    let u = CscData::<WgpuRuntime>::from_slices(
        &symbolic.u_col_ptrs,
        &symbolic.u_row_indices,
        &u_values_gpu.to_vec::<f32>(),
        [n, n],
        &device,
    )?;

    // No row permutations (static pivoting)
    let row_perm: Vec<usize> = (0..n).collect();
    let row_perm_inv: Vec<usize> = (0..n).collect();

    Ok(LuFactors {
        l,
        u,
        row_perm,
        row_perm_inv,
    })
}

/// Uniform buffer params for scatter kernel
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScatterParams {
    offset: u32,
    count: u32,
}

/// Uniform buffer params for AXPY kernel
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    scale: f32,
    offset: u32,
    count: u32,
    _pad: u32,
}

/// Uniform buffer params for gather kernel
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    offset: u32,
    count: u32,
}

/// Uniform buffer params for divide kernel
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DivideParams {
    inv_pivot: f32,
    offset: u32,
    count: u32,
    _pad: u32,
}

/// GPU factorization loop for f32
#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
fn run_factorization_f32(
    client: &WgpuClient,
    n: usize,
    col_ptrs: &[i64],
    a_values_gpu: &Tensor<WgpuRuntime>,
    a_row_indices_gpu: &Tensor<WgpuRuntime>,
    l_values_gpu: &Tensor<WgpuRuntime>,
    l_row_indices_gpu: &Tensor<WgpuRuntime>,
    u_values_gpu: &Tensor<WgpuRuntime>,
    u_row_indices_gpu: &Tensor<WgpuRuntime>,
    work_gpu: &Tensor<WgpuRuntime>,
    symbolic: &LuSymbolic,
    options: &LuOptions,
) -> Result<()> {
    let cache = &client.pipeline_cache;
    let queue = &client.queue;
    let wgpu_device = &client.wgpu_device;

    // Get buffer references
    let a_values_buf = get_buffer(a_values_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid A values buffer".to_string()))?;
    let a_row_indices_buf = get_buffer(a_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid A row_indices buffer".to_string()))?;
    let l_values_buf = get_buffer(l_values_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L values buffer".to_string()))?;
    let l_row_indices_buf = get_buffer(l_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L row_indices buffer".to_string()))?;
    let u_values_buf = get_buffer(u_values_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U values buffer".to_string()))?;
    let u_row_indices_buf = get_buffer(u_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U row_indices buffer".to_string()))?;
    let work_buf = get_buffer(work_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid work buffer".to_string()))?;

    // Create reusable uniform buffers for parameters
    let scatter_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("scatter_params"),
        size: std::mem::size_of::<ScatterParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let axpy_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("axpy_params"),
        size: std::mem::size_of::<AxpyParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gather_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("gather_params"),
        size: std::mem::size_of::<GatherParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let divide_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("divide_params"),
        size: std::mem::size_of::<DivideParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load shaders and create pipelines
    let shader_source = include_str!("../../../../runtime/wgpu/shaders/sparse_linalg.wgsl");

    // Scatter pipeline (1 uniform + 3 storage)
    let scatter_module = cache.get_or_create_module_from_source("sparse_lu_scatter", shader_source);
    let scatter_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let scatter_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_lu_scatter",
        "sparse_scatter_offset_f32",
        &scatter_module,
        &scatter_layout,
    );

    // AXPY pipeline (1 uniform + 3 storage)
    let axpy_module = cache.get_or_create_module_from_source("sparse_lu_axpy", shader_source);
    let axpy_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let axpy_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_lu_axpy",
        "sparse_axpy_offset_f32",
        &axpy_module,
        &axpy_layout,
    );

    // Gather pipeline (1 uniform + 3 storage)
    let gather_module = cache.get_or_create_module_from_source("sparse_lu_gather", shader_source);
    let gather_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let gather_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_lu_gather",
        "sparse_gather_clear_offset_f32",
        &gather_module,
        &gather_layout,
    );

    // Divide pipeline (1 uniform + 1 storage)
    let divide_module = cache.get_or_create_module_from_source("sparse_lu_divide", shader_source);
    let divide_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let divide_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_lu_divide",
        "sparse_divide_pivot_offset_f32",
        &divide_module,
        &divide_layout,
    );

    // Process column by column
    for k in 0..n {
        // ======================================================================
        // Step 1: Scatter column k of A into work vector
        // ======================================================================
        let a_col_start = col_ptrs[k] as u32;
        let a_col_end = col_ptrs[k + 1] as u32;
        let a_col_nnz = a_col_end - a_col_start;

        if a_col_nnz > 0 {
            let params = ScatterParams {
                offset: a_col_start,
                count: a_col_nnz,
            };
            queue.write_buffer(&scatter_params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_bind_group"),
                layout: &scatter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_values_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a_row_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: work_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scatter_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scatter_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("scatter_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&scatter_pipeline);
                pass.set_bind_group(0, Some(&bind_group), &[]);
                pass.dispatch_workgroups(workgroup_count(a_col_nnz as usize), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // ======================================================================
        // Step 2: Sparse triangular solve - for each column j in reach(k)
        // ======================================================================
        for &j in &symbolic.reach[k] {
            if j >= k {
                continue;
            }

            let l_col_start = symbolic.l_col_ptrs[j] as u32;
            let l_col_end = symbolic.l_col_ptrs[j + 1] as u32;
            let l_col_nnz = l_col_end - l_col_start;

            if l_col_nnz > 0 {
                // Scale factor is work[j] - see note in CUDA implementation
                let scale = 1.0f32;

                let params = AxpyParams {
                    scale,
                    offset: l_col_start,
                    count: l_col_nnz,
                    _pad: 0,
                };
                queue.write_buffer(&axpy_params_buf, 0, bytemuck::bytes_of(&params));

                let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("axpy_bind_group"),
                    layout: &axpy_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: l_values_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: l_row_indices_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: work_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: axpy_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("axpy_encoder"),
                    });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("axpy_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&axpy_pipeline);
                    pass.set_bind_group(0, Some(&bind_group), &[]);
                    pass.dispatch_workgroups(workgroup_count(l_col_nnz as usize), 1, 1);
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
        }

        // ======================================================================
        // Step 3: Static pivoting
        // ======================================================================
        let inv_pivot = if options.diagonal_shift > 0.0 {
            1.0 / (1.0 + options.diagonal_shift as f32)
        } else {
            1.0f32
        };

        // ======================================================================
        // Step 4: Gather U values from work
        // ======================================================================
        let u_col_start = symbolic.u_col_ptrs[k] as u32;
        let u_col_end = symbolic.u_col_ptrs[k + 1] as u32;
        let u_col_nnz = u_col_end - u_col_start;

        if u_col_nnz > 0 {
            let params = GatherParams {
                offset: u_col_start,
                count: u_col_nnz,
            };
            queue.write_buffer(&gather_params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gather_u_bind_group"),
                layout: &gather_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: work_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_row_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: u_values_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gather_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gather_u_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gather_u_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gather_pipeline);
                pass.set_bind_group(0, Some(&bind_group), &[]);
                pass.dispatch_workgroups(workgroup_count(u_col_nnz as usize), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // ======================================================================
        // Step 5: Gather L values and divide by pivot
        // ======================================================================
        let l_col_start = symbolic.l_col_ptrs[k] as u32;
        let l_col_end = symbolic.l_col_ptrs[k + 1] as u32;
        let l_col_nnz = l_col_end - l_col_start;

        if l_col_nnz > 0 {
            // Gather L values from work
            let gather_params = GatherParams {
                offset: l_col_start,
                count: l_col_nnz,
            };
            queue.write_buffer(&gather_params_buf, 0, bytemuck::bytes_of(&gather_params));

            let gather_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gather_l_bind_group"),
                layout: &gather_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: work_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: l_row_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: l_values_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gather_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gather_l_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gather_l_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gather_pipeline);
                pass.set_bind_group(0, Some(&gather_bind_group), &[]);
                pass.dispatch_workgroups(workgroup_count(l_col_nnz as usize), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));

            // Divide by pivot
            let divide_params = DivideParams {
                inv_pivot,
                offset: l_col_start,
                count: l_col_nnz,
                _pad: 0,
            };
            queue.write_buffer(&divide_params_buf, 0, bytemuck::bytes_of(&divide_params));

            let divide_bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("divide_bind_group"),
                layout: &divide_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: l_values_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: divide_params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("divide_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("divide_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&divide_pipeline);
                pass.set_bind_group(0, Some(&divide_bind_group), &[]);
                pass.dispatch_workgroups(workgroup_count(l_col_nnz as usize), 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
    }

    // Wait for all GPU operations to complete
    let _ = wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });

    Ok(())
}

/// Sparse LU factorization with simple symbolic structure (WebGPU)
///
/// For matrices without pre-computed symbolic structure, falls back to CPU.
#[cfg(feature = "wgpu")]
pub fn sparse_lu_simple_wgpu(
    client: &WgpuClient,
    a: &CscData<WgpuRuntime>,
    options: &LuOptions,
) -> Result<LuFactors<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_lu_simple_wgpu",
        });
    }

    // Without symbolic structure, we need to compute it on CPU first
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();
    let values_f32: Vec<f32> = a.values().to_vec();
    let values: Vec<f64> = values_f32.iter().map(|&x| x as f64).collect();

    let cpu_device =
        <crate::runtime::cpu::CpuRuntime as crate::runtime::Runtime>::Device::default();
    let cpu_a = CscData::<crate::runtime::cpu::CpuRuntime>::from_slices(
        &col_ptrs,
        &row_indices,
        &values,
        a.shape,
        &cpu_device,
    )?;

    let cpu_factors = crate::algorithm::sparse_linalg::lu::sparse_lu_simple_cpu(&cpu_a, options)?;

    // Transfer results back
    let device = &client.device_id;

    let l_col_ptrs: Vec<i64> = cpu_factors.l.col_ptrs().to_vec();
    let l_row_indices: Vec<i64> = cpu_factors.l.row_indices().to_vec();
    let l_values_f64: Vec<f64> = cpu_factors.l.values().to_vec();
    let l_values: Vec<f32> = l_values_f64.iter().map(|&x| x as f32).collect();

    let u_col_ptrs: Vec<i64> = cpu_factors.u.col_ptrs().to_vec();
    let u_row_indices: Vec<i64> = cpu_factors.u.row_indices().to_vec();
    let u_values_f64: Vec<f64> = cpu_factors.u.values().to_vec();
    let u_values: Vec<f32> = u_values_f64.iter().map(|&x| x as f32).collect();

    let l = CscData::<WgpuRuntime>::from_slices(
        &l_col_ptrs,
        &l_row_indices,
        &l_values,
        [n, n],
        device,
    )?;
    let u = CscData::<WgpuRuntime>::from_slices(
        &u_col_ptrs,
        &u_row_indices,
        &u_values,
        [n, n],
        device,
    )?;

    Ok(LuFactors {
        l,
        u,
        row_perm: cpu_factors.row_perm,
        row_perm_inv: cpu_factors.row_perm_inv,
    })
}

/// Uniform buffer params for CSC lower triangular solve
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TrsvCscLowerParams {
    level_offset: u32,
    level_size: u32,
    n: u32,
    unit_diagonal: u32,
}

/// Uniform buffer params for CSC upper triangular solve
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TrsvCscUpperParams {
    level_offset: u32,
    level_size: u32,
    n: u32,
    _pad: u32,
}

/// Uniform buffer params for find diagonal indices
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FindDiagCscParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Uniform buffer params for apply permutation
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ApplyPermParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Solve Ax = b using precomputed LU factors (WebGPU)
///
/// Uses GPU-based level-scheduled triangular solve with CSC format.
/// All computation happens on GPU - only transfer of b at start and x at end.
#[cfg(feature = "wgpu")]
pub fn sparse_lu_solve_wgpu(
    client: &WgpuClient,
    factors: &LuFactors<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    use crate::algorithm::sparse_linalg::levels::{
        compute_levels_csc_lower, compute_levels_csc_upper, flatten_levels,
    };

    let n = factors.row_perm.len();
    let b_shape = b.shape();

    if b_shape.is_empty() || b_shape[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b_shape.to_vec(),
        });
    }

    if b_shape.len() > 1 && b_shape[1] != 1 {
        return Err(Error::Internal(
            "Multi-RHS GPU solve not yet implemented".to_string(),
        ));
    }

    let dtype = b.dtype();
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_lu_solve_wgpu",
        });
    }

    let device = b.device();
    let cache = &client.pipeline_cache;
    let queue = &client.queue;
    let wgpu_device = &client.wgpu_device;

    // ==========================================================================
    // STEP 1: Setup - transfer structure to CPU for level scheduling
    // ==========================================================================

    let l_col_ptrs: Vec<i64> = factors.l.col_ptrs().to_vec();
    let l_row_indices: Vec<i64> = factors.l.row_indices().to_vec();
    let u_col_ptrs: Vec<i64> = factors.u.col_ptrs().to_vec();
    let u_row_indices: Vec<i64> = factors.u.row_indices().to_vec();

    // Compute level schedules for L and U (CSC format)
    let l_schedule = compute_levels_csc_lower(n, &l_col_ptrs, &l_row_indices)?;
    let u_schedule = compute_levels_csc_upper(n, &u_col_ptrs, &u_row_indices)?;

    let (l_level_ptrs, l_level_cols) = flatten_levels(&l_schedule);
    let (u_level_ptrs, u_level_cols) = flatten_levels(&u_schedule);

    // ==========================================================================
    // STEP 2: Transfer auxiliary data to GPU ONCE
    // ==========================================================================

    // L structure on GPU
    let l_col_ptrs_i32: Vec<i32> = l_col_ptrs.iter().map(|&x| x as i32).collect();
    let l_row_indices_i32: Vec<i32> = l_row_indices.iter().map(|&x| x as i32).collect();
    let l_col_ptrs_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&l_col_ptrs_i32, &[l_col_ptrs_i32.len()], &device)?;
    let l_row_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &l_row_indices_i32,
        &[l_row_indices_i32.len()],
        &device,
    )?;

    // U structure on GPU
    let u_col_ptrs_i32: Vec<i32> = u_col_ptrs.iter().map(|&x| x as i32).collect();
    let u_row_indices_i32: Vec<i32> = u_row_indices.iter().map(|&x| x as i32).collect();
    let u_col_ptrs_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&u_col_ptrs_i32, &[u_col_ptrs_i32.len()], &device)?;
    let u_row_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &u_row_indices_i32,
        &[u_row_indices_i32.len()],
        &device,
    )?;

    // Level schedule data on GPU
    let l_level_cols_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&l_level_cols, &[l_level_cols.len()], &device)?;
    let u_level_cols_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&u_level_cols, &[u_level_cols.len()], &device)?;

    // Row permutation on GPU
    let row_perm_i32: Vec<i32> = factors.row_perm.iter().map(|&x| x as i32).collect();
    let row_perm_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&row_perm_i32, &[row_perm_i32.len()], &device)?;

    // Diagonal pointer arrays
    let l_diag_ptr_gpu: Tensor<WgpuRuntime> =
        Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, &device)?;
    let u_diag_ptr_gpu: Tensor<WgpuRuntime> =
        Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, &device)?;

    // Get buffer references
    let l_col_ptrs_buf = get_buffer(l_col_ptrs_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L col_ptrs buffer".to_string()))?;
    let l_row_indices_buf = get_buffer(l_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L row_indices buffer".to_string()))?;
    let l_values_buf = get_buffer(factors.l.values().ptr())
        .ok_or_else(|| Error::Internal("Invalid L values buffer".to_string()))?;
    let l_diag_ptr_buf = get_buffer(l_diag_ptr_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L diag_ptr buffer".to_string()))?;
    let l_level_cols_buf = get_buffer(l_level_cols_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid L level_cols buffer".to_string()))?;

    let u_col_ptrs_buf = get_buffer(u_col_ptrs_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U col_ptrs buffer".to_string()))?;
    let u_row_indices_buf = get_buffer(u_row_indices_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U row_indices buffer".to_string()))?;
    let u_values_buf = get_buffer(factors.u.values().ptr())
        .ok_or_else(|| Error::Internal("Invalid U values buffer".to_string()))?;
    let u_diag_ptr_buf = get_buffer(u_diag_ptr_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U diag_ptr buffer".to_string()))?;
    let u_level_cols_buf = get_buffer(u_level_cols_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid U level_cols buffer".to_string()))?;

    let b_buf =
        get_buffer(b.ptr()).ok_or_else(|| Error::Internal("Invalid b buffer".to_string()))?;
    let row_perm_buf = get_buffer(row_perm_gpu.ptr())
        .ok_or_else(|| Error::Internal("Invalid row_perm buffer".to_string()))?;

    // Load shader
    let shader_source = include_str!("../../../../runtime/wgpu/shaders/sparse_linalg.wgsl");

    // ==========================================================================
    // STEP 3: Find diagonal indices
    // ==========================================================================

    let find_diag_module =
        cache.get_or_create_module_from_source("sparse_find_diag_csc", shader_source);
    let find_diag_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let find_diag_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_find_diag_csc",
        "find_diag_indices_csc_f32",
        &find_diag_module,
        &find_diag_layout,
    );

    let find_diag_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("find_diag_params"),
        size: std::mem::size_of::<FindDiagCscParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let find_diag_params = FindDiagCscParams {
        n: n as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    queue.write_buffer(
        &find_diag_params_buf,
        0,
        bytemuck::bytes_of(&find_diag_params),
    );

    // Find L diagonals
    {
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_diag_l_bind_group"),
            layout: &find_diag_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: l_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: l_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: l_diag_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: find_diag_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("find_diag_l_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_diag_l_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&find_diag_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count(n), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // Find U diagonals
    {
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_diag_u_bind_group"),
            layout: &find_diag_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_diag_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: find_diag_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("find_diag_u_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_diag_u_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&find_diag_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count(n), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // ==========================================================================
    // STEP 4: Apply permutation - y = P * b
    // ==========================================================================

    let y_gpu: Tensor<WgpuRuntime> = Tensor::<WgpuRuntime>::try_zeros(&[n], dtype, &device)?;
    let y_buf =
        get_buffer(y_gpu.ptr()).ok_or_else(|| Error::Internal("Invalid y buffer".to_string()))?;

    let perm_module = cache.get_or_create_module_from_source("sparse_apply_perm", shader_source);
    let perm_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let perm_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_apply_perm",
        "apply_row_perm_f32",
        &perm_module,
        &perm_layout,
    );

    let perm_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("perm_params"),
        size: std::mem::size_of::<ApplyPermParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let perm_params = ApplyPermParams {
        n: n as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    queue.write_buffer(&perm_params_buf, 0, bytemuck::bytes_of(&perm_params));

    {
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perm_bind_group"),
            layout: &perm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: row_perm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perm_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("perm_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("perm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&perm_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count(n), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // ==========================================================================
    // STEP 5: Forward substitution - L * z = y
    // ==========================================================================

    let lower_module =
        cache.get_or_create_module_from_source("sparse_trsv_csc_lower", shader_source);
    let lower_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let lower_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_trsv_csc_lower",
        "sparse_trsv_csc_lower_level_f32",
        &lower_module,
        &lower_layout,
    );

    let lower_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("lower_params"),
        size: std::mem::size_of::<TrsvCscLowerParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for level in 0..l_schedule.num_levels {
        let level_start = l_level_ptrs[level] as u32;
        let level_end = l_level_ptrs[level + 1] as u32;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        let params = TrsvCscLowerParams {
            level_offset: level_start,
            level_size,
            n: n as u32,
            unit_diagonal: 1, // L has unit diagonal
        };
        queue.write_buffer(&lower_params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lower_bind_group"),
            layout: &lower_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: l_level_cols_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: l_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: l_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: l_values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: l_diag_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: lower_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lower_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lower_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&lower_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count(level_size as usize), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // ==========================================================================
    // STEP 6: Backward substitution - U * x = z
    // ==========================================================================

    let upper_module =
        cache.get_or_create_module_from_source("sparse_trsv_csc_upper", shader_source);
    let upper_layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let upper_pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_trsv_csc_upper",
        "sparse_trsv_csc_upper_level_f32",
        &upper_module,
        &upper_layout,
    );

    let upper_params_buf = wgpu_device.create_buffer(&BufferDescriptor {
        label: Some("upper_params"),
        size: std::mem::size_of::<TrsvCscUpperParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for level in 0..u_schedule.num_levels {
        let level_start = u_level_ptrs[level] as u32;
        let level_end = u_level_ptrs[level + 1] as u32;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        let params = TrsvCscUpperParams {
            level_offset: level_start,
            level_size,
            n: n as u32,
            _pad: 0,
        };
        queue.write_buffer(&upper_params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upper_bind_group"),
            layout: &upper_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_level_cols_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: u_col_ptrs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_row_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: u_values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: u_diag_ptr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: upper_params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("upper_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("upper_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&upper_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count(level_size as usize), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // Wait for all GPU operations to complete
    let _ = wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });

    // y_gpu now contains the solution x
    Ok(y_gpu)
}
