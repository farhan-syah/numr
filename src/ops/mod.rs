//! Tensor operations
//!
//! This module defines operation traits and implementations for
//! arithmetic, matrix operations, reductions, and activations.
//!
//! # Design
//!
//! Operations are defined as traits that are implemented by `RuntimeClient`.
//! This gives operations access to device and allocator for creating output tensors.
//!
//! ```text
//! RuntimeClient<R>
//!   └── implements TensorOps<R>
//!         ├── add, sub, mul, div (binary arithmetic)
//!         ├── neg, sqrt, exp, ... (unary operations)
//!         ├── matmul             (matrix multiplication)
//!         ├── sum, mean, max, min (reductions)
//!         └── relu, sigmoid, softmax (activations)
//! ```
//!
//! # Implementing Operations for a New Backend
//!
//! To add operations for a new backend (e.g., CUDA, WebGPU):
//!
//! 1. **Implement `TensorOps<YourRuntime>` for your `Client` type:**
//!    ```ignore
//!    impl TensorOps<CudaRuntime> for CudaClient {
//!        fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
//!            // 1. Validate shapes are broadcastable
//!            let out_shape = broadcast_shape(a.shape(), b.shape())
//!                .ok_or(Error::BroadcastError { ... })?;
//!
//!            // 2. Allocate output tensor
//!            let out = Tensor::try_empty(&out_shape, a.dtype(), self.device())?;
//!
//!            // 3. Dispatch kernel
//!            cuda_add_kernel(a.ptr(), b.ptr(), out.ptr(), ...);
//!
//!            Ok(out)
//!        }
//!        // ... other operations
//!    }
//!    ```
//!
//! 2. **Use helper types for operation parameters:**
//!    - [`BinaryOp`], [`UnaryOp`] - Operation kind enums for dispatch
//!    - [`MatmulParams`] - Matrix multiplication configuration
//!    - [`ReduceOp`] - Reduction operation kinds
//!    - [`ActivationKind`] - Activation function kinds
//!
//! 3. **Use validation helpers:**
//!    - `broadcast_shape` - Compute broadcast shape for binary ops
//!    - `validate_matmul_shapes` - Validate matmul dimensions
//!    - `reduce_output_shape` - Compute reduction output shape
//!
//! # Operation Categories
//!
//! ## Element-wise Operations
//! Binary (add, sub, mul, div) and unary (neg, abs, sqrt, exp, log, sin, cos, tanh).
//!
//! **Note:** Broadcasting is implemented for binary arithmetic and comparison ops
//! via strided kernels on CPU.
//!
//! ## Matrix Operations
//! Matrix multiplication with batching support. Inner dimensions must match.
//!
//! ## Reductions
//! Sum, mean, max, min over specified dimensions with optional keepdim.
//!
//! ## Activations
//! ReLU, sigmoid, softmax for neural network layers.

pub(crate) mod activation;
mod arithmetic;
pub(crate) mod common;
pub(crate) mod conv_common;
pub(crate) mod conv_transpose_common;
mod dispatch;
pub(crate) mod distance_common;
pub(crate) mod impl_generic;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod semiring;
mod special;
pub mod traits;

// Re-export user-facing types
pub use activation::ActivationKind;
pub use arithmetic::{BinaryOp, CompareOp, UnaryOp};
pub use matmul::MatmulParams;
pub use reduce::ReduceOp;
pub use semiring::SemiringOp;
pub use special::SpecialFunctions;

// Internal re-exports (accessible within the crate only)
pub(crate) use arithmetic::broadcast_shape;
pub(crate) use matmul::{
    matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes,
};
pub(crate) use reduce::{
    AccumulationPrecision, compute_reduce_strides, reduce_dim_output_shape, reduce_output_shape,
};
pub use traits::Fp8MatmulOps;
pub use traits::{
    ActivationOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps, ConvOps, CumulativeOps,
    DistanceMetric, DistanceOps, EinsumOps, GemmActivation, GemmEpilogueOps, IndexingOps, Kernel,
    LinalgOps, LogicalOps, MatmulOps, MeshgridIndexing, NormalizationOps, PaddingMode, ReduceOps,
    ScalarOps, ScatterReduceOp, SemiringMatmulOps, ShapeOps, SortingOps, StatisticalOps, TensorOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
pub use traits::{AdvancedRandomOps, MultivariateRandomOps, QuasiRandomOps, RandomOps};
