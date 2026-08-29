//! Matrix multiplication operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Matrix multiplication operations
pub trait MatmulOps<R: Runtime> {
    /// Matrix multiplication: a @ b
    ///
    /// Supports batched matmul for tensors with more than 2 dimensions.
    fn matmul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Fused matrix multiplication with bias addition: C = A @ B + bias
    ///
    /// This is a fused operation that combines matrix multiplication and bias addition
    /// into a single kernel, avoiding an extra memory round-trip compared to separate
    /// `matmul` followed by `add`. This is the core operation for neural network linear
    /// layers: `output = input @ weight.T + bias`.
    ///
    /// # Algorithm (Epilogue Fusion)
    ///
    /// The bias addition is fused into the GEMM epilogue:
    /// ```text
    /// 1. Load tiles of A and B into shared memory
    /// 2. Compute partial products, accumulate in registers
    /// 3. Repeat for all K tiles
    /// 4. EPILOGUE: For each output element C[i][j]:
    ///    C[i][j] = accumulated_value[i][j] + bias[j]
    /// 5. Write final result to global memory
    /// ```
    ///
    /// This saves one global memory read/write cycle vs the naive:
    /// ```text
    /// temp = A @ B       // Write temp to global memory
    /// C = temp + bias    // Read temp, write C
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `` `[..., M, K]` ``
    /// * `b` - Weight tensor of shape `` `[..., K, N]` ``
    /// * `bias` - Bias tensor of shape `` `[N]` `` (1D, broadcast across all M rows)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `` `[..., M, N]` `` where `` `C[..., i, j] = sum_k(A[..., i, k] * B[..., k, j]) + bias[j]` ``
    ///
    /// The output dtype is the operand dtype for every width except I8, which
    /// widens to I32 (quantized accumulation, the same rule `matmul` follows).
    /// An I8 operand therefore takes an I32 bias: the bias seeds the I32
    /// accumulator instead of being added to a narrowed result.
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if:
    /// - Inner dimensions don't match (A's last dim != B's second-to-last dim)
    /// - Bias shape doesn't match output columns (bias.len() != N)
    /// - Bias is not 1D
    ///
    /// Returns `Error::DTypeMismatch` if A and B differ, or if the bias differs
    /// from the output dtype at a width that does not widen. An I8 operand with
    /// a non-I32 bias returns `Error::InvalidArgument` naming the expected dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::MatmulOps;
    ///
    /// // Linear layer: output = input @ weight.T + bias
    /// let batch = 2; let seq_len = 5; let hidden = 3; let out_features = 4;
    /// let input = client.randn(&[batch, seq_len, hidden], DType::F32)?;
    /// let weight = client.randn(&[out_features, hidden], DType::F32)?;
    /// let bias = client.randn(&[out_features], DType::F32)?;
    ///
    /// // Using fused operation (faster):
    /// let output = client.matmul_bias(&input, &weight.transpose(-1, -2)?, &bias)?;
    ///
    /// // Equivalent to (slower - extra memory round-trip):
    /// let temp = client.matmul(&input, &weight.transpose(-1, -2)?)?;
    /// let output = client.add(&temp, &bias.unsqueeze(0)?.unsqueeze(0)?)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Performance
    ///
    /// Fusing bias into the GEMM epilogue provides:
    /// - ~2x memory bandwidth reduction for the bias addition
    /// - Better cache utilization (output stays in registers)
    /// - Reduced kernel launch overhead (one kernel instead of two)
    ///
    /// For large matrices where GEMM is compute-bound, the speedup is modest.
    /// For smaller matrices (typical in LLM inference), the speedup is more significant.
    ///
    /// # Backend Support
    ///
    /// | Backend | Supported DTypes | Tensor Dims | Notes |
    /// |---------|------------------|-------------|-------|
    /// | CPU     | All dtypes       | 2D, 3D+     | Full support via generic kernels |
    /// | CUDA    | All dtypes       | 2D, 3D+     | Integers fuse the bias into the wide accumulator |
    /// | WebGPU  | F32, I32, U32, F16 | 2D, 3D only | Returns error for >3D tensors |
    ///
    /// WebGPU is limited to 3D workgroup dispatches and returns an error for >3D tensors.
    fn matmul_bias(&self, a: &Tensor<R>, b: &Tensor<R>, bias: &Tensor<R>) -> Result<Tensor<R>>;
}
