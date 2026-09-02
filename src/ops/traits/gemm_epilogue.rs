//! GEMM epilogue operations trait.
//!
//! Fused matrix multiplication with bias and activation/residual in a single kernel.
//! Eliminates extra kernel launches and memory round-trips for `Linear + Activation` patterns.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Activation function to fuse into the GEMM epilogue.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GemmActivation {
    /// No activation (identity)
    None,
    /// ReLU: max(0, x)
    ReLU,
    /// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    GELU,
    /// SiLU/Swish: x * sigmoid(x)
    SiLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: hyperbolic tangent
    Tanh,
}

impl GemmActivation {
    /// Numeric code passed to the kernels.
    ///
    /// Every backend's epilogue switches on this value, so the mapping lives
    /// here once rather than being restated per backend — a kernel and its
    /// launcher disagreeing about a code would silently apply the wrong
    /// activation.
    pub fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::ReLU => 1,
            Self::GELU => 2,
            Self::SiLU => 3,
            Self::Sigmoid => 4,
            Self::Tanh => 5,
        }
    }
}

/// Fused GEMM + bias + activation/residual operations.
///
/// These operations fuse post-processing into the GEMM epilogue, avoiding extra
/// kernel launches and memory round-trips compared to separate matmul_bias + activation.
///
/// # Performance
///
/// For a typical `Linear + ReLU` pattern:
/// - **Unfused**: `temp = A @ B + bias` (write temp), `out = relu(temp)` (read temp, write out)
/// - **Fused**: `out = relu(A @ B + bias)` (single write)
///
/// This saves one full read+write of the output matrix.
///
/// # Backend Support
///
/// | Backend | Supported DTypes | Notes |
/// |---------|------------------|-------|
/// | CPU     | All dtypes       | SIMD-accelerated activations |
/// | CUDA    | F32, F64, F16, BF16 | Fused in GEMM epilogue |
/// | WebGPU  | F32              | Per-activation entry points |
pub trait GemmEpilogueOps<R: Runtime> {
    /// Fused GEMM + bias + activation: `activation(A @ B + bias)`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `[..., M, K]`
    /// * `b` - Weight tensor of shape `[..., K, N]`
    /// * `bias` - Bias tensor of shape `[N]` (1D, broadcast across rows)
    /// * `activation` - Activation function to apply element-wise after bias addition
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]`
    fn matmul_bias_activation(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        bias: &Tensor<R>,
        activation: GemmActivation,
    ) -> Result<Tensor<R>>;

    /// Fused GEMM + bias + residual: `A @ B + bias + residual`
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of shape `[..., M, K]`
    /// * `b` - Weight tensor of shape `[..., K, N]`
    /// * `bias` - Bias tensor of shape `[N]` (1D, broadcast across rows)
    /// * `residual` - Residual tensor of shape `[..., M, N]` (same shape as output)
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[..., M, N]`
    fn matmul_bias_residual(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        bias: &Tensor<R>,
        residual: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Backward pass for fused GEMM + bias + activation.
    ///
    /// Computes gradients for `activation(A @ B + bias)`.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient of the loss w.r.t. the output, shape `[..., M, N]`
    /// * `a` - Input tensor from forward pass, shape `[..., M, K]`
    /// * `b` - Weight tensor from forward pass, shape `[..., K, N]`
    /// * `bias` - Bias tensor from forward pass, shape `[N]`
    /// * `activation` - Activation function used in forward pass
    ///
    /// # Returns
    ///
    /// Tuple of `(d_a, d_b, d_bias)`:
    /// * `d_a` - Gradient w.r.t. input A, shape `[..., M, K]`
    /// * `d_b` - Gradient w.r.t. weight B, shape `[..., K, N]`
    /// * `d_bias` - Gradient w.r.t. bias, shape `[N]`
    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
        bias: &Tensor<R>,
        activation: GemmActivation,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
