//! CUDA kernel loading, caching, and launching infrastructure
//!
//! This module provides utilities for loading PTX kernels compiled by build.rs,
//! caching the modules per-device, and launching kernels with type-safe wrappers.
//!
//! # Architecture
//!
//! - PTX files are compiled by `build.rs` using nvcc
//! - Modules are loaded on first use and cached per-device
//! - Generic launch helpers reduce boilerplate across kernel types
//!
//! # Thread Safety
//!
//! The module cache uses `OnceLock<Mutex<HashMap>>` for thread-safe initialization
//! and concurrent access from multiple CUDA streams.

use cudarc::driver::PushKernelArg;
pub use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// PTX Sources (compiled by build.rs)
// ============================================================================

/// Directory containing compiled PTX files (set by build.rs)
const KERNEL_DIR: &str = env!("CUDA_KERNEL_DIR");

/// Load PTX from compiled file.
fn load_ptx(name: &str) -> Ptx {
    let path = format!("{}/{}.ptx", KERNEL_DIR, name);
    Ptx::from_file(path)
}

// ============================================================================
// Kernel Module Cache
// ============================================================================

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

// ============================================================================
// Launch Configuration
// ============================================================================

/// Block size for element-wise operations (256 threads is optimal for most GPUs)
pub const BLOCK_SIZE: u32 = 256;

/// Calculate optimal grid dimensions for element-wise operations.
///
/// Uses a 1D grid with blocks of `BLOCK_SIZE` threads each.
#[inline]
pub fn elementwise_launch_config(numel: usize) -> (u32, u32, u32) {
    // Floored at 1: a grid extent of 0 is itself a launch error, even though
    // `numel == 0` needs no work. Callers that reach this helper with an
    // empty tensor rely on the kernel's own bounds guard (`idx < n`) to no-op.
    let grid_size = (((numel as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
    (grid_size, 1, 1)
}

/// Calculate launch configuration for global reduction kernels.
///
/// Limits grid size to prevent excessive block overhead for small inputs.
#[inline]
#[allow(dead_code)] // Kept for potential future optimization of global reductions
pub fn reduce_launch_config(numel: usize) -> (u32, u32) {
    let block_size = BLOCK_SIZE;
    let grid_size = ((numel as u32) + block_size - 1) / block_size;
    // Limit grid size to ensure we don't launch too many blocks
    let grid_size = grid_size.min(1024);
    (grid_size, block_size)
}

/// Maximum CUDA grid extent along `x`, for every compute capability numr targets.
const MAX_GRID_DIM_X: u32 = 2_147_483_647;

/// Maximum CUDA grid extent along `y` and `z`, for every compute capability numr targets.
const MAX_GRID_DIM_YZ: u32 = 65_535;

/// Calculate launch configuration for dimension-wise reduction.
///
/// Uses a 2D grid over the `[outer, inner]` output plane: `outer` on `x`,
/// `inner` on `y`. Each axis is clamped to its architectural maximum, and the
/// dim-reduction kernels grid-stride over both axes to cover the remainder — so
/// an `inner` past 65535 (e.g. summing `[1, 32, 4096, 64]` over dim 1, where
/// `inner` is 262144) launches and computes every output instead of being
/// rejected with `CUDA_ERROR_INVALID_VALUE`.
///
/// Each axis is also floored at 1: a grid extent of 0 is itself a launch error,
/// and callers pass sizes already clamped to at least 1.
#[inline]
pub fn reduce_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), u32) {
    let grid_x = (outer.min(MAX_GRID_DIM_X as usize) as u32).max(1);
    let grid_y = (inner.min(MAX_GRID_DIM_YZ as usize) as u32).max(1);
    let block = BLOCK_SIZE;
    ((grid_x, grid_y, 1), block)
}

/// Calculate launch configuration for softmax over the last dimension.
///
/// One block per row, with threads cooperating to compute the softmax.
/// Returns (grid_size, block_size, shared_memory_bytes).
#[inline]
pub fn softmax_launch_config(outer: usize, dim_size: usize) -> (u32, u32, u32) {
    // One block per row, threads handle the dimension
    // Block size must be a power of 2 for the shared-memory tree reduction to work correctly
    let block_size = BLOCK_SIZE.min(dim_size as u32).next_power_of_two();
    let block_size = block_size.min(BLOCK_SIZE);
    let grid_size = outer as u32;
    // Shared memory: 2 arrays of block_size floats (for max and sum reduction)
    let shared_mem = 2 * block_size * 4; // f32
    (grid_size, block_size, shared_mem)
}

/// Calculate launch configuration for softmax over a non-last dimension.
///
/// Uses a 2D grid to process all (outer, inner) pairs in parallel.
/// Each thread processes one element position across the reduction dimension.
#[inline]
#[allow(dead_code)] // Available for future optimized softmax_dim kernel
pub fn softmax_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    // Use 2D grid: one thread per (outer, inner) pair
    // Each thread sequentially processes the dim_size elements
    let total_elements = (outer * inner) as u32;
    let grid_x = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, 1, 1);
    let block = (BLOCK_SIZE, 1, 1);
    (grid, block)
}

/// Create a launch configuration from grid, block, and shared memory sizes.
#[inline]
pub fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    }
}

// ============================================================================
// Kernel Naming
// ============================================================================

/// Kernel operation categories for consistent naming.
pub mod kernel_names {
    /// Binary operations (two tensor inputs)
    pub const BINARY_MODULE: &str = "binary";
    /// Unary operations (one tensor input)
    pub const UNARY_MODULE: &str = "unary";
    /// Scalar operations (tensor + scalar input)
    pub const SCALAR_MODULE: &str = "scalar";
    /// Reduction operations (sum, max, min)
    pub const REDUCE_MODULE: &str = "reduce";
    /// Integer reduction operations, which accumulate in `Numr128`
    pub const REDUCE_INT_MODULE: &str = "reduce_int";
    /// Comparison operations (eq, ne, lt, le, gt, ge)
    pub const COMPARE_MODULE: &str = "compare";
    /// Element-wise activation functions (relu, sigmoid, silu, gelu, leaky_relu, elu)
    pub const ACTIVATION_MODULE: &str = "activation";
    /// Softmax forward + backward kernels
    pub const SOFTMAX_MODULE: &str = "softmax";
    /// Normalization operations (rms_norm, layer_norm)
    pub const NORM_MODULE: &str = "norm";
    /// Fused add + normalization operations
    pub const FUSED_ADD_NORM_MODULE: &str = "fused_add_norm";
    /// Type casting operations (cast between dtypes)
    pub const CAST_MODULE: &str = "cast";
    /// Utility operations (fill, arange, linspace, eye)
    pub const UTILITY_MODULE: &str = "utility";
    /// Random sampling operations (rand, randn, randint, multinomial)
    pub const UTILITY_RANDOM_MODULE: &str = "utility_random";
    /// Coordinate-addressed indexing (gather_nd, gather_2d, slice_assign)
    pub const INDEX_ND_MODULE: &str = "index_nd";
    /// Scatter with reduction (sum, prod, max, min, mean)
    pub const SCATTER_REDUCE_MODULE: &str = "scatter_reduce";
    /// Ternary operations (where)
    pub const TERNARY_MODULE: &str = "ternary";
    /// Prefix sum operations (exclusive scan)
    #[cfg(feature = "sparse")]
    pub const SCAN_MODULE: &str = "scan";
    /// Sparse matrix operations (SpMV, SpMM)
    #[cfg(feature = "sparse")]
    pub const SPARSE_SPMV_MODULE: &str = "sparse_spmv";
    /// Sparse matrix element-wise operations (add, sub, mul)
    #[cfg(feature = "sparse")]
    pub const SPARSE_MERGE_MODULE: &str = "sparse_merge";
    /// Sparse format conversion operations (COO↔CSR↔CSC)
    #[cfg(feature = "sparse")]
    pub const SPARSE_CONVERT_MODULE: &str = "sparse_convert";
    /// COO sparse element-wise operations with CUB sort
    #[cfg(feature = "sparse")]
    pub const SPARSE_COO_MODULE: &str = "sparse_coo";
    /// Dense × Sparse matrix multiplication (DSMM / SpMM)
    #[cfg(feature = "sparse")]
    pub const DSMM_MODULE: &str = "dsmm";
    /// Linear algebra basic operations (trace, diag, diagflat, identity, transpose)
    pub const LINALG_BASIC_MODULE: &str = "linalg_basic";
    /// Banded linear system solvers (Thomas, banded LU)
    pub const LINALG_BANDED_MODULE: &str = "linalg_banded";
    /// Linear algebra solvers (forward_sub, backward_sub, det_from_lu, apply_permutation)
    pub const LINALG_SOLVERS_MODULE: &str = "linalg_solvers";
    /// Matrix decompositions (LU, Cholesky, QR)
    pub const LINALG_DECOMP_MODULE: &str = "linalg_decomp";
    /// SVD decomposition (Jacobi algorithm)
    pub const LINALG_SVD_MODULE: &str = "linalg_svd";
    /// Symmetric eigenvalue decomposition (Jacobi algorithm)
    pub const LINALG_EIGEN_MODULE: &str = "linalg_eigen";
    /// Schur decomposition (Hessenberg + QR iteration)
    pub const LINALG_SCHUR_MODULE: &str = "linalg_schur";
    /// General eigenvalue decomposition
    pub const LINALG_EIGEN_GENERAL_MODULE: &str = "linalg_eigen_general";
    /// Advanced decompositions (rsf2csf)
    pub const LINALG_ADVANCED_MODULE: &str = "linalg_advanced";
    /// QZ decomposition (generalized Schur - double-shift algorithm)
    pub const LINALG_QZ_MODULE: &str = "linalg_qz";
    /// Matrix functions (exp, log, sqrt on quasi-triangular matrices)
    pub const LINALG_MATRIX_FUNCS_MODULE: &str = "linalg_matrix_funcs";
    /// Matrix multiplication operations (native tiled GEMM)
    pub const MATMUL_MODULE: &str = "matmul";
    /// Tensor-core WMMA GEMM for F16/BF16 (sm_70+)
    pub const MATMUL_WMMA_MODULE: &str = "matmul_wmma";
    /// GEMV operations (matrix-vector multiply for small M)
    pub const GEMV_MODULE: &str = "gemv";
    /// Integer GEMV operations, which accumulate in `Numr128`
    pub const GEMV_INT_MODULE: &str = "gemv_int";
    /// Compile-time-tiled integer GEMM
    pub const MATMUL_INT_MODULE: &str = "matmul_int";
    /// Compile-time-tiled FP8 GEMM (FP8E4M3, FP8E5M2), F32 accumulation
    pub const MATMUL_FP8_MODULE: &str = "matmul_fp8";
    /// Cumulative operations (cumsum, cumprod, logsumexp)
    pub const CUMULATIVE_MODULE: &str = "cumulative";
    /// Distribution sampling operations (bernoulli, beta, gamma, etc.)
    pub const DISTRIBUTIONS_MODULE: &str = "distributions";
    /// Quasi-random sequence generation (sobol, halton, latin_hypercube)
    pub const QUASIRANDOM_MODULE: &str = "quasirandom";
    /// Advanced PRNGs (philox, threefry, pcg64, xoshiro256)
    pub const ADVANCED_RANDOM_MODULE: &str = "advanced_random";
    /// Statistics operations (mode)
    pub const STATISTICS_MODULE: &str = "statistics";
    /// Semiring matrix multiplication operations
    pub const SEMIRING_MATMUL_MODULE: &str = "semiring_matmul";

    /// Generate kernel name for reduction operations.
    #[inline]
    pub fn reduce_kernel(op: &str) -> String {
        format!("reduce_{}", op)
    }

    /// Generate kernel name for dimension-wise reduction operations.
    #[inline]
    pub fn reduce_dim_kernel(op: &str) -> String {
        format!("reduce_{}_dim", op)
    }
}

/// Get the kernel name suffix for a given dtype.
pub fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::FP8E4M3 => "fp8_e4m3",
        DType::FP8E5M2 => "fp8_e5m2",
        DType::I64 => "i64",
        DType::I32 => "i32",
        DType::I16 => "i16",
        DType::I8 => "i8",
        DType::U64 => "u64",
        DType::U32 => "u32",
        DType::U16 => "u16",
        DType::U8 => "u8",
        DType::Bool => "bool",
        DType::Complex64 => "c64",
        DType::Complex128 => "c128",
    }
}

/// Generate a kernel name with dtype suffix.
///
/// # Example
///
/// ```ignore
/// let name = kernel_name("add", DType::F32); // "add_f32"
/// ```
#[inline]
pub fn kernel_name(base: &str, dtype: DType) -> String {
    format!("{}_{}", base, dtype_suffix(dtype))
}

/// The PTX module holding this dtype's GEMV kernels.
///
/// Integer GEMV lives in its own translation unit: it accumulates in `Numr128`
/// instead of a float register, and `gemv.cu` is already at its size limit.
/// Every dtype that reaches a GEMV launcher has kernels in one module or the
/// other, so there is no dtype gate on the small-M fast path: `matmul` admits
/// only dtypes that `int_matmul_has_kernel` or `gemv.cu` covers.
#[inline]
fn gemv_module(dtype: DType) -> &'static str {
    if dtype.is_int() {
        kernel_names::GEMV_INT_MODULE
    } else {
        kernel_names::GEMV_MODULE
    }
}

/// Whether `matmul_int.cu` and `gemv_int.cu` instantiate this dtype.
///
/// Both files instantiate the same list, so one predicate gates the tiled GEMM
/// and the GEMV fast paths together, and `ops/cuda/matmul.rs` gates the public
/// entry point on it. Every integer dtype is covered except I8: CPU `matmul` on
/// I8 returns an I32 tensor (quantized accumulation, see `ops/cpu/matmul.rs`),
/// so an I8-in/I8-out CUDA kernel would disagree with the reference on both the
/// output dtype and the value.
#[inline]
pub fn int_matmul_has_kernel(dtype: DType) -> bool {
    dtype.is_int() && dtype != DType::I8
}

/// The PTX module holding this dtype's reduction kernels.
///
/// Integer reductions live in their own translation unit: they accumulate in
/// `Numr128` instead of a float register, and `reduce.cu` is already at its size
/// limit. `reduce_int.cu` instantiates every integer dtype and every reduction
/// name that `reduce.cu` does, so this is a straight swap of module, never of
/// kernel name.
#[inline]
pub(crate) fn reduce_module(dtype: DType) -> &'static str {
    if dtype.is_int() {
        kernel_names::REDUCE_INT_MODULE
    } else {
        kernel_names::REDUCE_MODULE
    }
}

// ============================================================================
// Generic Kernel Launch Helpers
// ============================================================================

/// Launch an element-wise unary kernel (one input, one output).
///
/// This handles the common pattern for operations like neg, abs, sqrt, exp, etc.
///
/// # Safety
///
/// `input_ptr` and `output_ptr` must be valid device memory pointers with at least
/// `numel` elements of the appropriate dtype.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "unary", "activation")
/// * `op` - Operation name (e.g., "neg", "relu")
/// * `dtype` - Data type of the tensors
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_unary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch an element-wise binary kernel (two inputs, one output).
///
/// This handles the common pattern for operations like add, sub, mul, div, etc.
///
/// # Safety
///
/// All pointers must be valid device memory with at least `numel` elements.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "binary", "compare")
/// * `op` - Operation name (e.g., "add", "eq")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// Matrix Multiplication Launch Helpers
// ============================================================================

use crate::algorithm::TileConfig;

/// Calculate launch configuration for register-tiled matrix multiplication.
///
/// Uses configurable tile sizes - no hardcoded values.
/// Grid: ceil(N/block_n) × ceil(M/block_m)
/// Block: (block_n/thread_n) × (block_m/thread_m) threads
#[inline]
pub fn matmul_launch_config(
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    // Dynamic shared memory: As[block_m][block_k] + Bs[block_k][block_n]
    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Calculate launch configuration for batched register-tiled matrix multiplication.
///
/// Uses 3D grid: (tiles_x, tiles_y, batch)
#[inline]
pub fn matmul_batched_launch_config(
    batch: usize,
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let grid_z = batch as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Get default tile configuration for a dtype.
///
/// These are reasonable defaults; can be overridden via autotuning.
#[inline]
pub fn default_tile_config(dtype: DType) -> TileConfig {
    match dtype {
        // F64 uses smaller tiles due to larger element size
        DType::F64 => TileConfig {
            block_m: 64,
            block_n: 64,
            block_k: 8,
            thread_m: 4,
            thread_n: 4,
        },
        // F32/F16/BF16 use larger tiles
        _ => TileConfig::CUDA,
    }
}

/// Shape-aware tile configuration for F32 batched matmul.
///
/// The default 128×128×8 tile is badly inefficient when N or M is small (e.g.
/// N=64 in the context-attention path): half the columns in every block are
/// wasted, and block_k=8 forces 64+ __syncthreads barriers for K=512.
///
/// Rules (all tiles keep smem ≤ 24KB per buffer, 48KB total for double-buffer):
/// - Small-N (N ≤ 64): use 64×64 block tile, block_k=32, thread_m=8, thread_n=4
///   (64×32 + 32×64 = 4096 floats × 2 buffers = 32KB — fits in 48KB limit)
/// - Small-M (M ≤ 64, N large): mirror the small-N tile transposed
/// - Large square (default): 128×128, block_k=16, thread_m=8, thread_n=8
///   (128×16 + 16×128 = 4096 floats × 2 buffers = 32KB — fits in 48KB limit)
///
/// Note: the matmul_f32 and matmul_batched_f32 kernels are double-buffered, so
/// shared memory is allocated as 2 × (block_m*block_k + block_k*block_n) floats.
#[inline]
pub fn f32_batched_tile_config(m: usize, n: usize, _k: usize) -> TileConfig {
    if n <= 64 || m <= 64 {
        // Attention shapes: Scores Q@Kᵀ (M=512, N=512 but K=64 so inner loop short)
        // and Context attn@V (M=512, N=64, K=512).
        // For N≤64: block_n=64 so no wasted columns; block_k=32 halves sync count.
        // thread_m=8, thread_n=4: 8×(64/4)=128 threads/block (2 warps × 4).
        // Smem per buffer: (64×32 + 32×64) × 4 = 16 384 bytes. Two buffers = 32KB.
        TileConfig {
            block_m: 64,
            block_n: 64,
            block_k: 32,
            thread_m: 8,
            thread_n: 4,
        }
    } else {
        // Large square shapes (e.g. 512×512×512, 1024×1024×1024).
        // block_k=8 matches the compile-time-tiled `matmul_f32_tiled_128x128x8_8x8`
        // kernel (register-blocked, unrolled micro-kernel — ~100x the old runtime-
        // param kernel). Smem per buffer: (128×8 + 8×128)×4 = 8 192 B; ×2 = 16 KB.
        TileConfig {
            block_m: 128,
            block_n: 128,
            block_k: 8,
            thread_m: 8,
            thread_n: 8,
        }
    }
}

/// Launch native tiled matmul kernel: C[M,N] = A[M,K] @ B[K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - C: M * N elements
pub unsafe fn launch_matmul_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // FP8 first: `gemv.cu` has no FP8 kernels, so the small-M fast path below
    // would look up a name that does not exist.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // Use GEMV kernel for small M (single-token decode in LLM inference)
    // The tiled GEMM wastes 99%+ compute when M < block_m (typically 128)
    if m <= 16 {
        unsafe {
            return launch_gemv_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // Tensor-core WMMA path: F16/BF16 with 16-aligned dims → up to ~100 TFLOPS on Ampere.
    if use_wmma(dtype, m, n, k) {
        unsafe {
            return launch_matmul_wmma_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k,
            );
        }
    }
    // Integers: same reason as F32 below. Their accumulator is a 16-byte
    // `Numr128`, so a runtime-sized `reg_c` spills four registers per slot.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // F32: dispatch to compile-time-tiled kernels so NVCC can unroll micro-kernel
    // loops and keep accumulators in registers (avoids local-memory spill).
    if dtype == DType::F32 {
        let tile_cfg = f32_batched_tile_config(m, n, k);
        unsafe {
            return launch_matmul_f32_tiled(
                context,
                stream,
                device_index,
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k,
                &tile_cfg,
            );
        }
    }
    let tile_cfg = default_tile_config(dtype);
    unsafe {
        launch_matmul_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            &tile_cfg,
        )
    }
}

/// Launch GEMV kernel: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] for small M
///
/// B is [K,N] row-major (non-transposed). One thread per output column, iterates K.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_gemv_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N/256), M, batch), block: (256, 1, 1)
    // One thread per output column, each thread iterates over K.
    let block_size: u32 = 256;
    let grid_x = ((n as u32) + block_size - 1) / block_size;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA GEMV kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch GEMV kernel with transposed B: C[batch,M,N] = A[batch,M,K] @ B^T
///
/// B is stored [N,K] row-major (transposed weight matrix, common for nn.Linear).
/// Warp-cooperative: each warp reduces one output column along K using shuffle.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
/// `b_ptr` points to the raw [N,K] data (NOT the transposed [K,N] view).
pub unsafe fn launch_gemv_kernel_bt(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv_bt", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N/WARPS_PER_BLOCK), M, batch), block: (256, 1, 1)
    // 8 warps per block, each warp handles one output column.
    let warps_per_block: u32 = 8;
    let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA GEMV-BT kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch multi-row GEMV kernel with transposed B: C[batch,M,N] = A[batch,M,K] @ B^T
///
/// Each warp computes 2 output columns, sharing the activation vector load across rows.
/// This halves activation memory bandwidth compared to `launch_gemv_kernel_bt`.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
/// `b_ptr` points to the raw [N,K] data (NOT the transposed [K,N] view).
pub unsafe fn launch_gemv_kernel_bt_mr(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv_bt_mr", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N / (WARPS_PER_BLOCK * ROWS_PER_WARP)), M, batch), block: (256, 1, 1)
    // 8 warps per block, each warp handles 2 output columns.
    let warps_per_block: u32 = 8;
    let rows_per_warp: u32 = 2;
    let cols_per_block = warps_per_block * rows_per_warp; // 16
    let grid_x = ((n as u32) + cols_per_block - 1) / cols_per_block;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA GEMV-BT-MR kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch native tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation.
    // For F32, the kernel is double-buffered (2 ping-pong smem slots).
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };
    let smem_factor: u32 = if dtype == DType::F32 { 2 } else { 1 };

    let base_cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    let cfg = LaunchConfig {
        shared_mem_bytes: base_cfg.shared_mem_bytes * smem_factor,
        ..base_cfg
    };
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA matmul kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch compile-time-tiled FP32 GEMM: C[M,N] = A[M,K] @ B[K,N].
///
/// Selects the extern "C" kernel instantiation that matches `tile_cfg` so that
/// NVCC can fully unroll the micro-kernel loops and keep all accumulators in
/// registers (no local-memory spill).
///
/// Supported configs (must match the extern "C" instantiations in matmul.cu):
///   128×128×8  TM=8 TN=8  → kernel `matmul_f32_tiled_128x128x8_8x8`  (256 threads)
///   64×64×32   TM=8 TN=4  → kernel `matmul_f32_tiled_64x64x32_8x4`   (128 threads)
///
/// Any other tile_cfg falls back to the generic `matmul_f32` kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
unsafe fn launch_matmul_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    // Map tile config to a specialised extern "C" kernel name.
    let specialized: Option<&'static str> = match (
        tile_cfg.block_m,
        tile_cfg.block_n,
        tile_cfg.block_k,
        tile_cfg.thread_m,
        tile_cfg.thread_n,
    ) {
        (128, 128, 8, 8, 8) => Some("matmul_f32_tiled_128x128x8_8x8"),
        (64, 64, 32, 8, 4) => Some("matmul_f32_tiled_64x64x32_8x4"),
        _ => None,
    };

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;

    if let Some(kernel_fn_name) = specialized {
        let func = get_kernel_function(&module, kernel_fn_name)?;

        // Grid: (ceil(N/BN), ceil(M/BM), 1)   Block: (BN/TN, BM/TM, 1)
        let bm = tile_cfg.block_m as u32;
        let bn = tile_cfg.block_n as u32;
        let tn = tile_cfg.thread_n as u32;
        let tm = tile_cfg.thread_m as u32;
        let grid_x = ((n as u32) + bn - 1) / bn;
        let grid_y = ((m as u32) + bm - 1) / bm;
        // The specialized tiled kernels (matmul_f32_tiled_*) use ONLY static
        // __shared__ arrays (no extern __shared__).  Dynamic shared memory must
        // be 0; setting it to the static-tile formula would add unused dynamic
        // smem on top of the existing static pool, pushing the per-block total
        // past the 48 KB default hardware limit and causing a silent launch
        // failure on sm_86 (Ampere) for the 64×64×32 config (32 KB static +
        // 32 KB dynamic = 64 KB > 48 KB).
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (bn / tn, bm / tm, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&c_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!(
                    "CUDA matmul F32 tiled kernel '{}' launch failed: {:?}",
                    kernel_fn_name, e
                ))
            })?;
        }
        Ok(())
    } else {
        // Fallback to existing generic kernel for any config we didn't specialise.
        let func = get_kernel_function(&module, "matmul_f32")?;

        let elem_size = 4usize; // f32
        let smem_factor: u32 = 2; // double-buffered
        let base_cfg = matmul_launch_config(m, n, tile_cfg, elem_size);
        let cfg = LaunchConfig {
            shared_mem_bytes: base_cfg.shared_mem_bytes * smem_factor,
            ..base_cfg
        };
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let block_m = tile_cfg.block_m as u32;
        let block_n = tile_cfg.block_n as u32;
        let block_k = tile_cfg.block_k as u32;
        let thread_m = tile_cfg.thread_m as u32;
        let thread_n = tile_cfg.thread_n as u32;

        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&c_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&block_m);
            builder.arg(&block_n);
            builder.arg(&block_k);
            builder.arg(&thread_m);
            builder.arg(&thread_n);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!(
                    "CUDA matmul F32 generic fallback kernel launch failed: {:?}",
                    e
                ))
            })?;
        }
        Ok(())
    }
}

/// Block and thread tile the compile-time-tiled integer kernels are built at.
///
/// `matmul_int.cu` instantiates exactly this shape and encodes it in the kernel
/// names, so the two must stay in step. A `Numr128` accumulator is 16 bytes, so
/// every `reg_c` slot costs four registers: a 4x4 thread tile spends 64, while
/// an 8x8 tile would need 256 and blow past the 255-register per-thread limit.
const INT_TILE: TileConfig = TileConfig {
    block_m: 64,
    block_n: 64,
    block_k: 8,
    thread_m: 4,
    thread_n: 4,
};

/// Launch the compile-time-tiled integer GEMM, batched or not, with or without
/// a fused bias.
///
/// `bias_ptr` selects the fused entry point, which seeds the 128-bit
/// accumulator with the bias instead of adding it after the narrow-back store.
///
/// The tile dimensions are template parameters in `matmul_int.cu` rather than
/// kernel arguments, so `reg_c` is sized exactly and stays in registers. That
/// fixes the shape of the launch here: four kernel names per dtype, and a grid
/// derived from `INT_TILE`.
///
/// Shared memory in those kernels is static, so the dynamic shared-memory
/// request must be zero - adding the tile formula on top would push the block
/// past the 48 KB limit and fail the launch silently.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
unsafe fn launch_matmul_int_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: Option<u64>,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // A single batch with no broadcasting takes the 2-D entry point, which skips
    // the per-block batch offset arithmetic.
    let plain = batch == 1 && a_batch == 1 && b_batch == 1;
    if !int_matmul_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "matmul_int_tiled",
        });
    }
    // `matmul_int.cu` builds all four names from the same dtype suffix and tile
    // shape, so they are composed here rather than listed per dtype.
    let base = match (bias_ptr.is_some(), plain) {
        (false, true) => "matmul",
        (false, false) => "matmul_batched",
        (true, true) => "matmul_bias",
        (true, false) => "matmul_bias_batched",
    };
    let kernel_fn_name = format!("{}_{}_tiled_64x64x8_4x4", base, dtype_suffix(dtype));

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_INT_MODULE)?;
    let func = get_kernel_function(&module, &kernel_fn_name)?;

    let bm = INT_TILE.block_m as u32;
    let bn = INT_TILE.block_n as u32;
    let tm = INT_TILE.thread_m as u32;
    let tn = INT_TILE.thread_n as u32;
    let cfg = LaunchConfig {
        grid_dim: (
            ((n as u32) + bn - 1) / bn,
            ((m as u32) + bm - 1) / bm,
            batch as u32,
        ),
        block_dim: (bn / tn, bm / tm, 1),
        shared_mem_bytes: 0,
    };

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        // The bias parameter sits between B and C in the fused kernels, so it is
        // bound before C rather than appended.
        if let Some(bias) = bias_ptr.as_ref() {
            builder.arg(bias);
        }
        builder.arg(&c_ptr);
        if !plain {
            builder.arg(&batch_u32);
        }
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        if !plain {
            builder.arg(&a_batch_u32);
            builder.arg(&b_batch_u32);
        }
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA integer matmul kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }

    Ok(())
}

/// Block and thread tile the compile-time-tiled FP8 kernels are built at.
///
/// `matmul_fp8.cu` instantiates exactly this shape and encodes it in the kernel
/// names, so the two must stay in step.
const FP8_TILE: TileConfig = TileConfig {
    block_m: 64,
    block_n: 64,
    block_k: 8,
    thread_m: 4,
    thread_n: 4,
};

/// Launch the compile-time-tiled FP8 GEMM, batched or not, with or without bias.
///
/// Same dtype in and out, accumulating in F32 to match CPU. The mixed-precision
/// kernels behind `Fp8MatmulOps` are a different operation and are not used here.
///
/// `bias_ptr` selects the fused-bias entry point, which seeds the accumulator
/// with the bias instead of adding it to a narrowed result.
///
/// Shared memory in these kernels is static, so the dynamic shared-memory
/// request must be zero.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
unsafe fn launch_matmul_fp8_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: Option<u64>,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // A single batch with no broadcasting takes the 2-D entry point, which skips
    // the per-block batch offset arithmetic.
    let plain = batch == 1 && a_batch == 1 && b_batch == 1;
    let kernel_fn_name = match (dtype, plain, bias_ptr.is_some()) {
        (DType::FP8E4M3, true, false) => "matmul_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, true, false) => "matmul_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, false, false) => "matmul_batched_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, false, false) => "matmul_batched_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, true, true) => "matmul_bias_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, true, true) => "matmul_bias_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, false, true) => "matmul_bias_batched_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, false, true) => "matmul_bias_batched_fp8_e5m2_tiled_64x64x8_4x4",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "matmul_fp8_tiled",
            });
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_FP8_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;

    let bm = FP8_TILE.block_m as u32;
    let bn = FP8_TILE.block_n as u32;
    let tm = FP8_TILE.thread_m as u32;
    let tn = FP8_TILE.thread_n as u32;
    let cfg = LaunchConfig {
        grid_dim: (
            ((n as u32) + bn - 1) / bn,
            ((m as u32) + bm - 1) / bm,
            batch as u32,
        ),
        block_dim: (bn / tn, bm / tm, 1),
        shared_mem_bytes: 0,
    };

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        // The bias entry points take it as the third argument, before C.
        if let Some(bias) = bias_ptr.as_ref() {
            builder.arg(bias);
        }
        builder.arg(&c_ptr);
        if !plain {
            builder.arg(&batch_u32);
        }
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        if !plain {
            builder.arg(&a_batch_u32);
            builder.arg(&b_batch_u32);
        }
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA FP8 matmul kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch native batched tiled matmul kernel: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - C: batch * M * N elements
pub unsafe fn launch_matmul_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // FP8 first: `gemv.cu` has no FP8 kernels (see launch_matmul_kernel).
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Use GEMV kernel for small M (batched case)
    if m <= 16 {
        unsafe {
            return launch_gemv_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Tensor-core WMMA path for F16/BF16 with 16-aligned dims.
    if use_wmma(dtype, m, n, k) {
        unsafe {
            return launch_matmul_wmma_batched_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Integers use the compile-time-tiled kernels at every batch size.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // F32 uses shape-aware tiles to avoid wasted columns and reduce sync count.
    let tile_cfg = match dtype {
        DType::F32 => f32_batched_tile_config(m, n, k),
        _ => default_tile_config(dtype),
    };
    unsafe {
        launch_matmul_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            batch,
            m,
            n,
            k,
            &tile_cfg,
            a_batch,
            b_batch,
        )
    }
}

/// Launch native batched tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_batched_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation.
    // For F32, the kernel is double-buffered (2 ping-pong smem slots).
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };
    let smem_factor: u32 = if dtype == DType::F32 { 2 } else { 1 };

    let base_cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    let cfg = LaunchConfig {
        shared_mem_bytes: base_cfg.shared_mem_bytes * smem_factor,
        ..base_cfg
    };
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA batched matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

// ============================================================================
// Tensor-Core WMMA GEMM Launcher
// ============================================================================
//
// Block: WARP_ROWS*WARP_COLS warps × 32 threads = 8 warps × 32 = 256 threads.
//   Warp grid: 4 rows × 2 cols. Each warp: WARP_M=2 × WARP_N=4 frags (32×64).
//   8 warps × 32×64 = 128×128 block tile. ✓
// Grid:  ceil(N/128) × ceil(M/128) [× batch]
// Static shared memory per block (single-buffered, no cp.async):
//   smem_A:   128 × 24 × 2 bytes = 6 144
//   smem_B:    16 × 136 × 2 bytes = 4 352
//   scratch:    8 × 256 × 4 bytes = 8 192
//   Total:   18 688 bytes ≈ 18.25 KB  (well within 48 KB)

/// Returns true when the WMMA path should be taken for this dtype and shape.
///
/// Conditions:
/// - dtype is F16 or BF16
/// - M, N, K are all multiples of 16 (WMMA requirement)
/// - M > 16 (keep existing m<=16 GEMV fast path)
#[inline]
fn use_wmma(dtype: DType, m: usize, n: usize, k: usize) -> bool {
    // The WMMA kernel is only correct for 16-aligned M/N/K (its sub-16 fragment
    // boundary handling is buggy). The matmul op (src/ops/cuda/matmul.rs) PADS
    // unaligned F16/BF16 operands up to the next multiple of 16 before dispatch,
    // so by the time we get here the dims are aligned — critical for the varlen
    // embedding path where M = total_tokens is rarely a multiple of 16 (without
    // the pad+WMMA, F16 fell to the ~100x-slower generic kernel: 57 vs 8500
    // GFLOP/s). `m > 16` keeps tiny-M matmuls on the GEMV path.
    matches!(dtype, DType::F16 | DType::BF16)
        && m > 16
        && m.is_multiple_of(16)
        && n.is_multiple_of(16)
        && k.is_multiple_of(16)
}

// WMMA block: 16 warps (4×4 warp grid), each warp = 32 threads → 512 threads.
// Each warp computes WARP_M=2 × WARP_N=2 fragments (32×32 outputs).
// 16 warps × 32×32 = 128×128. ✓
const WMMA_BLOCK_THREADS: u32 = 512;
const WMMA_BLOCK_TILE_M: u32 = 128;
const WMMA_BLOCK_TILE_N: u32 = 128;

/// Shared-memory per WMMA block in bytes.
///
/// Single-buffered A+B staging + per-warp F32 epilogue scratch:
///   smem_A:   128 × 24 × 2 bytes =  6 144 bytes
///   smem_B:    16 × 136 × 2 bytes =  4 352 bytes
///   scratch:   16 warps × 256 × 4 bytes = 16 384 bytes = 16 KB
///   Total:    26 880 bytes ≈ 26.25 KB  (well within 48 KB)
// WMMA kernels use only statically-declared __shared__ arrays; there is no
// extern __shared__ (dynamic) allocation.  Pass 0 so CUDA does not add
// extra dynamic smem on top of the static pool (which would push total over
// the 48 KB default per-block limit on sm_86).
const WMMA_SMEM_BYTES: u32 = 0;

/// Launch 2-D (non-batched) WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16.
pub unsafe fn launch_matmul_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!("matmul_wmma_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &func_name)?;

    let grid_x = ((n as u32) + WMMA_BLOCK_TILE_N - 1) / WMMA_BLOCK_TILE_N;
    let grid_y = ((m as u32) + WMMA_BLOCK_TILE_M - 1) / WMMA_BLOCK_TILE_M;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA WMMA matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16.
pub unsafe fn launch_matmul_wmma_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!(
        "matmul_wmma_batched_{}",
        crate::runtime::cuda::kernels::loader::dtype_suffix(dtype)
    );
    let func = get_kernel_function(&module, &func_name)?;

    let grid_x = ((n as u32) + WMMA_BLOCK_TILE_N - 1) / WMMA_BLOCK_TILE_N;
    let grid_y = ((m as u32) + WMMA_BLOCK_TILE_M - 1) / WMMA_BLOCK_TILE_M;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    };

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Fused Matmul+Bias Kernel Launch
// ============================================================================

/// Launch native tiled fused matmul+bias kernel: C[M,N] = A[M,K] @ B[K,N] + bias[N]
///
/// Uses the same tiled GEMM algorithm as matmul, but fuses bias addition into the
/// epilogue to avoid an extra memory round-trip.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - bias: N elements (1D, broadcast across rows)
/// - C: M * N elements (output)
pub unsafe fn launch_matmul_bias_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // FP8 has its own fused-bias kernels; `matmul.cu` instantiates none.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // Integers likewise: `matmul.cu` has no integer kernels at all, and the
    // fused form is required rather than merely faster - the bias has to join
    // the 128-bit accumulator before it saturates, which a separate add cannot
    // do. There is no small-M shortcut here because `gemv_int.cu` takes no bias.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    unsafe {
        launch_matmul_bias_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            m,
            n,
            k,
            &default_tile_config(dtype),
        )
    }
}

/// Launch native tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA matmul_bias kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch native batched tiled fused matmul+bias kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - bias: N elements (1D, broadcast across all batches and rows)
/// - C: batch * M * N elements (output)
pub unsafe fn launch_matmul_bias_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // FP8 has its own fused-bias kernels; `matmul.cu` instantiates none.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Integers: same reason as the non-batched entry point above.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    unsafe {
        launch_matmul_bias_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            batch,
            m,
            n,
            k,
            &default_tile_config(dtype),
            a_batch,
            b_batch,
        )
    }
}

/// Launch native batched tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_batched_kernel_with_config(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA batched matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Semiring Matrix Multiplication Kernel Launchers
// ============================================================================

/// Launch semiring matrix multiplication kernel.
///
/// Uses a simple one-thread-per-element kernel parameterized by semiring op.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_semiring_matmul_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    semiring_op: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SEMIRING_MATMUL_MODULE)?;
    let func_name = kernel_name("semiring_matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // Simple 16x16 thread blocks, one thread per output element
    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (n as u32 + block_x - 1) / block_x;
    let grid_y = (m as u32 + block_y - 1) / block_y;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&semiring_op);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA semiring matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched semiring matrix multiplication kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_semiring_matmul_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    semiring_op: u32,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SEMIRING_MATMUL_MODULE)?;
    let func_name = kernel_name("semiring_matmul_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (n as u32 + block_x - 1) / block_x;
    let grid_y = (m as u32 + block_y - 1) / block_y;
    let grid_z = batch as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let batch_u32 = batch as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&semiring_op);
        builder.arg(&batch_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA batched semiring matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
