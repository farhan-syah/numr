//! Matrix multiplication algorithm contracts
//!
//! This module defines the universal tiled GEMM algorithm that all backends
//! must implement. The algorithm uses a 3-level memory hierarchy optimization:
//!
//! 1. **Thread Block Tile** - Load tiles into shared memory (GPU) or L1 cache (CPU)
//! 2. **Register Tile** - Each thread/lane computes a micro-tile (e.g., 4x4 or 8x8)
//! 3. **Micro-Kernel** - FMA instructions on data in registers
//!
//! # Algorithm Overview
//!
//! ```text
//! C`[M,N]` = A`[M,K]` @ B`[K,N]`
//!
//! For each block tile (BLOCK_M × BLOCK_N) of C:
//!   For k = 0..K step BLOCK_K:
//!     1. Cooperative load: threads load A tile `[BLOCK_M, BLOCK_K]` to shared mem
//!     2. Cooperative load: threads load B tile `[BLOCK_K, BLOCK_N]` to shared mem
//!     3. Barrier sync
//!     4. Each thread:
//!        - Load THREAD_M elements from A tile into registers
//!        - Load THREAD_N elements from B tile into registers
//!        - Compute THREAD_M × THREAD_N FMAs (outer product)
//!     5. Barrier sync
//!   Write THREAD_M × THREAD_N results to C
//! ```
//!
//! # Tile Configuration
//!
//! The [`TileConfig`] struct defines the tiling parameters. Different backends
//! should use configurations tuned for their hardware:
//!
//! | Backend | BLOCK_M | BLOCK_N | BLOCK_K | THREAD_M | THREAD_N |
//! |---------|---------|---------|---------|----------|----------|
//! | CUDA    | 128     | 128     | 8       | 8        | 8        |
//! | WebGPU  | 64      | 64      | 8       | 4        | 4        |
//! | CPU AVX | 48      | 8       | 8       | 6        | 8        |
//!
//! # Batched MatMul
//!
//! For batched operations, the batch dimension maps to:
//! - **GPU**: `blockIdx.z` / `GlobalID.z`
//! - **CPU**: Parallel outer loop
//!
//! Each batch is processed independently using the same tiled algorithm.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Tile configuration for the tiled GEMM algorithm
///
/// These parameters control the blocking strategy at different memory levels.
/// Optimal values depend on the target hardware's cache hierarchy.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Block tile size for M dimension (shared memory / L1 cache level)
    pub block_m: usize,
    /// Block tile size for N dimension
    pub block_n: usize,
    /// Block tile size for K dimension (loop tiling)
    pub block_k: usize,
    /// Register tile size for M dimension (per thread/lane)
    pub thread_m: usize,
    /// Register tile size for N dimension (per thread/lane)
    pub thread_n: usize,
}

impl TileConfig {
    /// Single default configuration applied to every CUDA architecture
    /// (Turing through Blackwell). Tuned on Ampere consumer hardware; no
    /// per-architecture branch selects a different tile.
    ///
    /// - 128×128 block tiles fit well in shared memory
    /// - 8×8 register tiles maximize occupancy
    /// - BLOCK_K=8 balances memory bandwidth vs compute
    pub const CUDA: Self = Self {
        block_m: 128,
        block_n: 128,
        block_k: 8,
        thread_m: 8,
        thread_n: 8,
    };

    /// WebGPU-optimized configuration
    ///
    /// - 64×64 block tiles for broader device compatibility
    /// - 4×4 register tiles reduce register pressure
    pub const WGPU: Self = Self {
        block_m: 64,
        block_n: 64,
        block_k: 8,
        thread_m: 4,
        thread_n: 4,
    };

    /// CPU AVX2/AVX-512 optimized configuration
    ///
    /// - 48×8 tiles fit in L1 cache (32KB)
    /// - 6×8 register tiles fill 12-16 YMM/ZMM registers
    pub const CPU_AVX: Self = Self {
        block_m: 48,
        block_n: 8,
        block_k: 8,
        thread_m: 6,
        thread_n: 8,
    };

    /// CPU NEON (ARM) optimized configuration
    pub const CPU_NEON: Self = Self {
        block_m: 32,
        block_n: 8,
        block_k: 8,
        thread_m: 4,
        thread_n: 8,
    };

    /// Simple configuration for testing/fallback
    pub const SIMPLE: Self = Self {
        block_m: 32,
        block_n: 32,
        block_k: 8,
        thread_m: 4,
        thread_n: 4,
    };

    /// Number of threads per block (for GPU backends)
    #[inline]
    pub const fn threads_per_block(&self) -> usize {
        (self.block_m / self.thread_m) * (self.block_n / self.thread_n)
    }

    /// Validate tile configuration consistency
    pub fn validate(&self) -> Result<()> {
        use crate::error::Error;

        if !self.block_m.is_multiple_of(self.thread_m) {
            return Err(Error::Internal(format!(
                "BLOCK_M ({}) must be divisible by THREAD_M ({})",
                self.block_m, self.thread_m
            )));
        }
        if !self.block_n.is_multiple_of(self.thread_n) {
            return Err(Error::Internal(format!(
                "BLOCK_N ({}) must be divisible by THREAD_N ({})",
                self.block_n, self.thread_n
            )));
        }
        Ok(())
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self::SIMPLE
    }
}

/// Algorithmic contract for matrix multiplication
///
/// All backends implementing matmul MUST implement this trait using the
/// tiled GEMM algorithm with register blocking to ensure:
///
/// 1. **Numerical parity** - Same results across CPU/GPU (within FP tolerance)
/// 2. **Performance** - Memory-hierarchy-aware computation
/// 3. **Portability** - Works on all backends with tunable parameters
///
/// # Algorithm: Register-Tiled GEMM
///
/// ```text
/// // Pseudocode for the universal algorithm
/// //
/// // Level 1: Block Tiling (Shared Memory / L1 Cache)
/// // Level 2: Register Tiling (Per-Thread Micro-Tile)
/// // Level 3: FMA Micro-Kernel
///
/// fn tiled_gemm(A`[M,K]`, B`[K,N]`, C`[M,N]`, config: TileConfig):
///     for bm = 0..M step BLOCK_M:
///         for bn = 0..N step BLOCK_N:
///             // Each thread block handles one (BLOCK_M × BLOCK_N) tile of C
///             reg_c`[THREAD_M][THREAD_N]` = 0  // Register tile accumulator
///
///             for bk = 0..K step BLOCK_K:
///                 // Cooperative load into shared memory
///                 shared_a`[BLOCK_M][BLOCK_K]` = A`[bm:bm+BLOCK_M, bk:bk+BLOCK_K]`
///                 shared_b`[BLOCK_K][BLOCK_N]` = B`[bk:bk+BLOCK_K, bn:bn+BLOCK_N]`
///                 barrier()
///
///                 // Register blocking: each thread computes THREAD_M × THREAD_N outputs
///                 for k = 0..BLOCK_K:
///                     reg_a`[THREAD_M]` = shared_a`[thread_row:thread_row+THREAD_M, k]`
///                     reg_b`[THREAD_N]` = shared_b`[k, thread_col:thread_col+THREAD_N]`
///
///                     // Outer product: THREAD_M × THREAD_N FMAs
///                     for i = 0..THREAD_M:
///                         for j = 0..THREAD_N:
///                             reg_c`[i][j]` += reg_a`[i]` * reg_b`[j]`
///
///                 barrier()
///
///             // Write register tile to global memory
///             C`[bm+thread_row:..., bn+thread_col:...]` = reg_c
/// ```
///
/// # Accumulation Precision
///
/// | Input Type | Accumulator | Reason                     |
/// |------------|-------------|----------------------------|
/// | F64        | F64         | Full precision             |
/// | F32        | F32         | Standard                   |
/// | F16/BF16   | F32         | Prevent overflow/underflow |
/// | FP8        | F32         | Very limited range         |
///
/// # Implementation Requirements
///
/// Backends MAY differ in:
/// - Tile size tuning (via TileConfig)
/// - Memory coalescing patterns
/// - Vectorization width (float4, half2, etc.)
/// - Warp-level primitives (shuffle, WMMA)
///
/// Backends MUST match in:
/// - Mathematical formula (same accumulation order)
/// - Accumulator precision per dtype
/// - Handling of edge tiles (partial tiles at boundaries)
pub trait MatmulAlgorithm<R: Runtime> {
    /// Get the tile configuration for this backend
    fn tile_config(&self) -> TileConfig;

    /// Dense matrix multiplication: C = A @ B
    ///
    /// Uses the register-tiled GEMM algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A with shape `[..., M, K]`
    /// * `b` - Matrix B with shape `[..., K, N]`
    ///
    /// # Returns
    ///
    /// Matrix C with shape `[..., M, N]`
    fn tiled_matmul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Batched matrix multiplication: `C[i]` = `A[i]` @ `B[i]`
    ///
    /// Uses Grid-Z mapping for GPU, parallel loop for CPU.
    ///
    /// # Arguments
    ///
    /// * `a` - Batched matrix A with shape `[batch, M, K]`
    /// * `b` - Batched matrix B with shape `[batch, K, N]`
    ///
    /// # Returns
    ///
    /// Batched matrix C with shape `[batch, M, N]`
    fn tiled_batched_matmul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate matmul input shapes and compute output shape
pub fn validate_matmul_shapes(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    use crate::error::Error;

    if a_shape.is_empty() || b_shape.is_empty() {
        return Err(Error::Internal(
            "Matmul requires at least 1D tensors".to_string(),
        ));
    }

    // Get the last two dimensions
    let a_k = a_shape[a_shape.len() - 1];
    let a_m = if a_shape.len() >= 2 {
        a_shape[a_shape.len() - 2]
    } else {
        1
    };

    let b_n = b_shape[b_shape.len() - 1];
    let b_k = if b_shape.len() >= 2 {
        b_shape[b_shape.len() - 2]
    } else {
        b_shape[b_shape.len() - 1]
    };

    // Check inner dimensions match
    if a_k != b_k {
        return Err(Error::ShapeMismatch {
            expected: vec![a_k],
            got: vec![b_k],
        });
    }

    // Compute output shape
    let mut out_shape = Vec::new();

    // Handle batch dimensions
    let a_batch = &a_shape[..a_shape.len().saturating_sub(2)];
    let b_batch = &b_shape[..b_shape.len().saturating_sub(2)];

    // Broadcast batch dimensions
    let max_batch = a_batch.len().max(b_batch.len());
    for i in 0..max_batch {
        let a_dim = if i < a_batch.len() {
            a_batch[a_batch.len() - 1 - i]
        } else {
            1
        };
        let b_dim = if i < b_batch.len() {
            b_batch[b_batch.len() - 1 - i]
        } else {
            1
        };

        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(Error::ShapeMismatch {
                expected: a_batch.to_vec(),
                got: b_batch.to_vec(),
            });
        }
        out_shape.push(a_dim.max(b_dim));
    }
    out_shape.reverse();

    // Add M and N
    out_shape.push(a_m);
    out_shape.push(b_n);

    Ok(out_shape)
}

/// Get accumulator dtype for mixed-precision matmul
pub fn accumulator_dtype(input_dtype: DType) -> DType {
    match input_dtype {
        DType::F64 => DType::F64,
        DType::F32 => DType::F32,
        DType::F16 | DType::BF16 => DType::F32, // F32 accumulation for half precision
        DType::FP8E4M3 | DType::FP8E5M2 => DType::F32, // FP8 always uses F32 accumulator
        // Complex types accumulate in same precision
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        // Integer types
        DType::I8 => DType::I32,
        DType::I16 => DType::I32,
        DType::I32 => DType::I64,
        DType::I64 => DType::I64,
        DType::U8 => DType::U32,
        DType::U16 => DType::U32,
        DType::U32 => DType::U64,
        DType::U64 => DType::U64,
        DType::Bool => DType::I32,
    }
}
