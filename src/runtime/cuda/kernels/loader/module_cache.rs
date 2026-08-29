//! PTX module loading and per-device module caching.
//!
//! PTX files are compiled by `build.rs`; modules are loaded on first use and
//! cached per-device. The cache uses `OnceLock<Mutex<HashMap>>` so concurrent
//! CUDA streams can share a loaded module safely.

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

/// Directory containing compiled PTX files (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load PTX from compiled file.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.ptx", KERNEL_DIR, name);
    Ptx::from_file(path)
}

/// Cache for loaded CUDA modules, keyed by (device_index, module_name)
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>> =
    OnceLock::new();

/// Get or load a CUDA module from PTX.
///
/// Modules are cached per-device to avoid repeated loading. This is thread-safe
/// and can be called concurrently from multiple streams.
///
/// # Arguments
///
/// * `context` - CUDA context for the target device
/// * `device_index` - Index of the target device (used as cache key)
/// * `module_name` - Name of the PTX file (without extension)
///
/// # Errors
///
/// Returns an error if the PTX file cannot be loaded or the module cannot be created.
pub fn get_or_load_module(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_name: &'static str,
) -> Result<Arc<CudaModule>> {
    let cache = MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let key = (device_index, module_name);
    if let Some(module) = guard.get(&key) {
        return Ok(module.clone());
    }

    // Load PTX and create module
    let ptx = load_ptx(module_name);
    let module = context.load_module(ptx).map_err(|e| {
        Error::Internal(format!(
            "Failed to load CUDA module '{}': {:?}. \
             Ensure CUDA kernels were compiled correctly by build.rs.",
            module_name, e
        ))
    })?;

    guard.insert(key, module.clone());

    Ok(module)
}

/// Pre-load a list of CUDA modules to avoid JIT compilation latency on first use.
///
/// This is useful for inference warmup: call this once with all module names
/// that will be used during inference to front-load all PTX→SASS compilation.
pub fn preload_modules(
    context: &Arc<CudaContext>,
    device_index: usize,
    module_names: &[&'static str],
) -> Result<()> {
    for name in module_names {
        get_or_load_module(context, device_index, name)?;
    }
    Ok(())
}

/// Get a kernel function from a loaded module.
///
/// # Arguments
///
/// * `module` - Loaded CUDA module
/// * `kernel_name` - Name of the kernel function (e.g., "add_f32")
///
/// # Errors
///
/// Returns an error if the kernel function is not found in the module.
pub fn get_kernel_function(module: &Arc<CudaModule>, kernel_name: &str) -> Result<CudaFunction> {
    module.load_function(kernel_name).map_err(|e| {
        Error::Internal(format!(
            "Failed to get kernel '{}': {:?}. \
             Check that the kernel name matches the CUDA source.",
            kernel_name, e
        ))
    })
}
