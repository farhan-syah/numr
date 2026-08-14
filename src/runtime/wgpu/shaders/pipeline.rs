//! WGSL compute pipeline infrastructure
//!
//! Provides pipeline caching and dispatch utilities for WGSL compute shaders.
//! Follows the same pattern as CUDA kernel launchers for DRY consistency.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Workgroup size for compute shaders (matches CUDA BLOCK_SIZE)
pub const WORKGROUP_SIZE: u32 = 256;

// ============================================================================
// Pipeline Cache
// ============================================================================

/// Cache for compute pipelines keyed by (shader_name, dtype)
pub struct PipelineCache {
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    /// Cached shader modules by name
    modules: Mutex<HashMap<&'static str, Arc<ShaderModule>>>,
    /// Cached shader modules by dynamic name (for generated shaders)
    dynamic_modules: Mutex<HashMap<String, Arc<ShaderModule>>>,
    /// Cached pipelines by (shader_name, entry_point)
    pipelines: Mutex<HashMap<(&'static str, &'static str), Arc<ComputePipeline>>>,
    /// Cached pipelines with dynamic keys (for generated shaders)
    dynamic_pipelines: Mutex<HashMap<(String, String), Arc<ComputePipeline>>>,
    /// Cached bind group layouts by layout key
    layouts: Mutex<HashMap<LayoutKey, Arc<BindGroupLayout>>>,
    /// Layouts whose final uniform binding uses a dynamic offset.
    dynamic_uniform_layouts: Mutex<HashMap<(u32, u32), Arc<BindGroupLayout>>>,
}

/// Key for bind group layout cache
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutKey {
    /// Number of storage buffers in the layout
    pub num_storage_buffers: u32,
    /// Number of uniform buffers in the layout
    pub num_uniform_buffers: u32,
    /// Number of leading storage buffers that are read-only.
    /// Bindings 0..num_readonly_storage are read-only,
    /// bindings num_readonly_storage..num_storage_buffers are read-write.
    /// Default: 0 (all read-write, backwards compatible).
    pub num_readonly_storage: u32,
}

impl PipelineCache {
    /// Create a new pipeline cache
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            modules: Mutex::new(HashMap::new()),
            dynamic_modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            dynamic_pipelines: Mutex::new(HashMap::new()),
            layouts: Mutex::new(HashMap::new()),
            dynamic_uniform_layouts: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a shader module
    pub fn get_or_create_module(&self, name: &'static str, source: &str) -> Arc<ShaderModule> {
        let mut modules = self.modules.lock();
        if let Some(module) = modules.get(name) {
            return module.clone();
        }

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name),
            source: ShaderSource::Wgsl(source.into()),
        });

        let module = Arc::new(module);
        modules.insert(name, module.clone());
        module
    }

    /// Get or create a shader module from dynamically generated source
    ///
    /// This variant accepts owned strings for cases where shader source is generated
    /// at runtime (e.g., multi-dtype shader generation).
    pub fn get_or_create_module_from_source(&self, name: &str, source: &str) -> Arc<ShaderModule> {
        let mut modules = self.dynamic_modules.lock();
        if let Some(module) = modules.get(name) {
            return module.clone();
        }

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name),
            source: ShaderSource::Wgsl(source.into()),
        });

        let module = Arc::new(module);
        modules.insert(name.to_string(), module.clone());
        module
    }

    /// Get or create a compute pipeline
    pub fn get_or_create_pipeline(
        &self,
        shader_name: &'static str,
        entry_point: &'static str,
        module: &ShaderModule,
        layout: &BindGroupLayout,
    ) -> Arc<ComputePipeline> {
        let key = (shader_name, entry_point);
        let mut pipelines = self.pipelines.lock();

        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", shader_name)),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0, // Not using push constants
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("{}_{}", shader_name, entry_point)),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline = Arc::new(pipeline);
        pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// Get or create a compute pipeline from dynamically named shader
    ///
    /// This variant accepts owned strings for cases where shader/entry point names
    /// are generated at runtime (e.g., multi-dtype shader generation).
    pub fn get_or_create_dynamic_pipeline(
        &self,
        shader_name: &str,
        entry_point: &str,
        module: &ShaderModule,
        layout: &BindGroupLayout,
    ) -> Arc<ComputePipeline> {
        let key = (shader_name.to_string(), entry_point.to_string());
        let mut pipelines = self.dynamic_pipelines.lock();

        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{}_layout", shader_name)),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("{}_{}", shader_name, entry_point)),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline = Arc::new(pipeline);
        pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// Get or create a bind group layout for storage buffers
    pub fn get_or_create_layout(&self, key: LayoutKey) -> Arc<BindGroupLayout> {
        let mut layouts = self.layouts.lock();

        if let Some(layout) = layouts.get(&key) {
            return layout.clone();
        }

        let mut entries = Vec::new();

        // Storage buffers: first num_readonly_storage are read-only, rest are read-write
        for i in 0..key.num_storage_buffers {
            let read_only = i < key.num_readonly_storage;
            entries.push(BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Uniform buffers (read-only params)
        for i in 0..key.num_uniform_buffers {
            entries.push(BindGroupLayoutEntry {
                binding: key.num_storage_buffers + i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("linalg_layout"),
                entries: &entries,
            });

        let layout = Arc::new(layout);
        layouts.insert(key, layout.clone());
        layout
    }

    /// Create a bind group from buffers
    pub fn create_bind_group(&self, layout: &BindGroupLayout, buffers: &[&Buffer]) -> BindGroup {
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout,
            entries: &entries,
        })
    }

    /// Get or create a storage-buffer layout followed by one dynamic uniform.
    pub fn get_or_create_dynamic_uniform_layout(
        &self,
        num_storage_buffers: u32,
        num_readonly_storage: u32,
    ) -> Arc<BindGroupLayout> {
        let key = (num_storage_buffers, num_readonly_storage);
        let mut layouts = self.dynamic_uniform_layouts.lock();
        if let Some(layout) = layouts.get(&key) {
            return layout.clone();
        }

        let mut entries = Vec::with_capacity(num_storage_buffers as usize + 1);
        for binding in 0..num_storage_buffers {
            entries.push(BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage {
                        read_only: binding < num_readonly_storage,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(BindGroupLayoutEntry {
            binding: num_storage_buffers,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });

        let layout = Arc::new(
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("dynamic_uniform_compute_layout"),
                    entries: &entries,
                }),
        );
        layouts.insert(key, layout.clone());
        layout
    }

    /// Create a bind group whose final binding is a dynamically offset uniform.
    pub fn create_bind_group_with_dynamic_uniform(
        &self,
        layout: &BindGroupLayout,
        storage_buffers: &[&Buffer],
        uniform: &Buffer,
        uniform_binding_size: u64,
    ) -> BindGroup {
        let mut entries: Vec<BindGroupEntry<'_>> = storage_buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(BindGroupEntry {
            binding: storage_buffers.len() as u32,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: uniform,
                offset: 0,
                size: NonZeroU64::new(uniform_binding_size),
            }),
        });

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("dynamic_uniform_compute_bind_group"),
            layout,
            entries: &entries,
        })
    }

    /// Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ============================================================================
// Dispatch Helpers
// ============================================================================

/// Compute number of workgroups for n elements
#[inline]
pub fn workgroup_count(n: usize) -> u32 {
    ((n as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

/// Map DType to WGSL entry point suffix
#[allow(dead_code)]
pub fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Err(Error::UnsupportedDType {
            dtype,
            op: "WGSL (f64 not supported in WebGPU)",
        }),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "linalg",
        }),
    }
}

/// Get entry point name for operation and dtype
#[allow(dead_code)]
pub fn entry_point(op: &str, dtype: DType) -> Result<String> {
    let suffix = dtype_suffix(dtype)?;
    Ok(format!("{}_{}", op, suffix))
}
