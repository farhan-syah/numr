//! # numr
//!
//! **High-performance numerical computing for Rust with multi-backend GPU acceleration.**
//!
//! numr provides n-dimensional arrays (tensors), linear algebra, FFT, and automatic
//! differentiation - with the same API across CPU, CUDA, and WebGPU backends.
//!
//! ## Why numr?
//!
//! - **Multi-backend**: Same code runs on CPU, CUDA, and WebGPU
//! - **No vendor lock-in**: Native kernels, not cuBLAS/MKL wrappers
//! - **Pure Rust**: No Python runtime, no FFI overhead, single binary deployment
//! - **Autograd included**: Reverse-mode automatic differentiation built-in
//! - **Sparse tensors**: CSR, CSC, COO formats with GPU support
//!
//! ## Features
//!
//! - **Tensors**: N-dimensional arrays with broadcasting, slicing, views
//! - **Linear algebra**: Matmul, LU, QR, SVD, Cholesky, eigendecomposition
//! - **FFT**: Fast Fourier transforms (1D, 2D, ND)
//! - **Element-wise ops**: Full set of math functions
//! - **Reductions**: Sum, mean, max, min, argmax, argmin along axes
//! - **Multiple dtypes**: f64, f32, f16, bf16, fp8, integers, bool
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use numr::prelude::*;
//!
//! let device = CpuDevice;
//! let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
//! let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();
//!
//! let c = &a + &b;
//! let d = a.matmul(&b)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `cpu` (default): CPU backend
//! - `cuda`: NVIDIA CUDA backend
//! - `wgpu`: Cross-platform GPU via WebGPU
//! - `rayon` (default): Multi-threaded CPU operations
//! - `f16`: Half-precision floats (F16, BF16)
//! - `sparse`: Sparse tensor formats (CSR, CSC, COO)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::items_after_test_module)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_return)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::identity_op)]
#![allow(clippy::useless_vec)]
#![allow(clippy::type_complexity)]
#![allow(clippy::let_and_return)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::erasing_op)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::len_zero)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::repeat_once)]
#![allow(clippy::unused_unit)]
#![allow(clippy::extra_unused_type_parameters)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::manual_repeat_n)]

pub mod algorithm;
pub mod autograd;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod runtime;
#[cfg(feature = "sparse")]
pub mod sparse;
pub mod tensor;

/// Prelude module for convenient imports
///
/// Import everything needed for tensor operations with `use numr::prelude::*`:
/// - Core types: `Tensor`, `DType`, `Layout`, `Error`, `Result`
/// - Runtime traits: `Runtime`, `Device`, `RuntimeClient`
/// - Operation traits: `TensorOps`, `ScalarOps`, `CompareOps`
/// - Algorithm traits: `LinearAlgebraAlgorithms`, `FftAlgorithms`, `SpecialFunctions`
/// - Backend runtimes: `CpuRuntime`, `CudaRuntime`, `WgpuRuntime` (feature-gated)
pub mod prelude {
    // Core types
    pub use crate::dtype::{DType, DataType};
    pub use crate::error::{Error, Result};
    pub use crate::tensor::{Layout, Shape, Strides, Tensor};

    // Runtime traits and graph types
    pub use crate::runtime::{CapturedGraph, Device, Graph, NoOpGraph, Runtime, RuntimeClient};

    // Operation traits (same API across all backends)
    pub use crate::ops::{
        ActivationOps, BinaryOps, CompareOps, ComplexOps, ConditionalOps, ConvOps, CumulativeOps,
        DistanceMetric, DistanceOps, IndexingOps, LinalgOps, LogicalOps, MatmulOps,
        MeshgridIndexing, NormalizationOps, PaddingMode, ReduceOps, ScalarOps, ShapeOps,
        SortingOps, StatisticalOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
    };
    pub use crate::ops::{AdvancedRandomOps, MultivariateRandomOps, QuasiRandomOps, RandomOps};

    // Algorithm traits
    pub use crate::algorithm::SpecialFunctions;
    pub use crate::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};

    // Backend runtimes
    pub use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[cfg(feature = "cuda")]
    pub use crate::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    #[cfg(feature = "wgpu")]
    pub use crate::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

    // Sparse tensors (feature-gated)
    #[cfg(feature = "sparse")]
    pub use crate::sparse::Sparse24Ops;
    #[cfg(feature = "sparse")]
    pub use crate::sparse::{Sparse24Tensor, SparseFormat, SparseOps, SparseTensor};
}

/// Default runtime based on enabled features
///
/// - With `cuda` feature: `CudaRuntime`
/// - With `wgpu` feature (no cuda): `WgpuRuntime`
/// - Otherwise: `CpuRuntime`
#[cfg(feature = "cuda")]
pub type DefaultRuntime = runtime::cuda::CudaRuntime;

/// Default runtime based on enabled features
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type DefaultRuntime = runtime::wgpu::WgpuRuntime;

/// Default runtime based on enabled features
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type DefaultRuntime = runtime::cpu::CpuRuntime;
