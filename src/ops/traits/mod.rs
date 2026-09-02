//! Operation traits for tensor operations.
//!
//! This module contains trait definitions for various tensor operations.
//! Implementations are in the backend-specific modules (cpu/, cuda/, wgpu/).

mod activation;
mod advanced_random;
mod binary;
mod compare;
mod complex;
mod conditional;
mod conv;
mod cumulative;
mod distance;
mod einsum;
mod fp8_matmul;
mod gemm_epilogue;
mod grouped_matmul;
mod indexing;
mod kernel;
mod linalg;
mod logical;
mod matmul;
pub mod multivariate;
mod normalization;
mod quasirandom;
mod random;
mod reduce;
mod scalar;
mod semiring_matmul;
mod shape;
mod sorting;
#[cfg(feature = "sparse")]
mod sparse_24;
mod statistics;
mod tensor_ops;
mod type_conversion;
mod unary;
mod utility;

pub use activation::ActivationOps;
pub use advanced_random::AdvancedRandomOps;
pub use binary::BinaryOps;
pub use compare::CompareOps;
pub use complex::ComplexOps;
pub use conditional::ConditionalOps;
pub use conv::{ConvOps, PaddingMode};
pub use cumulative::CumulativeOps;
pub use distance::{DistanceMetric, DistanceOps};
pub use einsum::EinsumOps;
pub use fp8_matmul::Fp8MatmulOps;
pub use gemm_epilogue::{GemmActivation, GemmEpilogueOps};
pub use grouped_matmul::GroupedMatmulOps;
pub use indexing::{IndexingOps, ScatterReduceOp};
pub use kernel::Kernel;
pub use linalg::LinalgOps;
pub use logical::LogicalOps;
pub use matmul::MatmulOps;
pub use multivariate::MultivariateRandomOps;
pub use normalization::NormalizationOps;
pub use quasirandom::QuasiRandomOps;
pub use random::RandomOps;
pub use reduce::ReduceOps;
pub use scalar::ScalarOps;
pub use semiring_matmul::SemiringMatmulOps;
pub use shape::ShapeOps;
pub use sorting::SortingOps;
#[cfg(feature = "sparse")]
pub use sparse_24::Sparse24Ops;
pub use statistics::StatisticalOps;
pub use tensor_ops::TensorOps;
pub use type_conversion::TypeConversionOps;
pub use unary::UnaryOps;
pub use utility::{MeshgridIndexing, UtilityOps};
