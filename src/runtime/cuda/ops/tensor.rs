//! CUDA runtime tensor operation implementations
//!
//! This module loads all operation implementations from the src/ops/cuda/ directory.
//! Each operation type (type conversion, complex, normalization, etc.) is implemented in its own module.

use super::super::{CudaClient, CudaRuntime};
use crate::ops::TensorOps;

// TensorOps is a supertrait that composes all individual operation traits.
// Since we implement all the component traits in separate files, we just need
// this empty impl to satisfy the supertrait requirement.
impl TensorOps<CudaRuntime> for CudaClient {}

// Load all CUDA operation implementations from src/ops/cuda/
#[path = "../../../ops/cuda/type_conversion.rs"]
mod type_conversion;

#[path = "../../../ops/cuda/complex.rs"]
mod complex;

#[path = "../../../ops/cuda/normalization.rs"]
mod normalization;

#[path = "../../../ops/cuda/matmul.rs"]
mod matmul;

#[path = "../../../ops/cuda/grouped_matmul.rs"]
mod grouped_matmul;

#[path = "../../../ops/cuda/conv.rs"]
mod conv;

#[path = "../../../ops/cuda/conv1d_im2col.rs"]
mod conv1d_im2col;

#[path = "../../../ops/cuda/conv_transpose1d_gemm.rs"]
mod conv_transpose1d_gemm;

#[path = "../../../ops/cuda/conv2d_im2col.rs"]
mod conv2d_im2col;

#[path = "../../../ops/cuda/cumulative.rs"]
mod cumulative;

#[path = "../../../ops/cuda/activation.rs"]
mod activation;

#[path = "../../../ops/cuda/binary.rs"]
mod binary;

#[path = "../../../ops/cuda/unary.rs"]
mod unary;

#[path = "../../../ops/cuda/indexing/mod.rs"]
mod indexing;

#[path = "../../../ops/cuda/statistics.rs"]
mod statistics;

#[path = "../../../ops/cuda/random.rs"]
mod random;

#[path = "../../../ops/cuda/advanced_random.rs"]
mod advanced_random;

#[path = "../../../ops/cuda/quasirandom.rs"]
mod quasirandom;

#[path = "../../../ops/cuda/linalg.rs"]
mod linalg;

#[path = "../../../ops/cuda/reduce.rs"]
mod reduce;

#[path = "../../../ops/cuda/shape.rs"]
mod shape;

#[path = "../../../ops/cuda/sorting.rs"]
mod sorting;

#[path = "../../../ops/cuda/conditional.rs"]
mod conditional;

#[path = "../../../ops/cuda/utility.rs"]
mod utility;

#[path = "../../../ops/cuda/distance.rs"]
mod distance;

#[path = "../../../ops/cuda/multivariate.rs"]
mod multivariate;

#[path = "../../../ops/cuda/gemm_epilogue.rs"]
mod gemm_epilogue;

#[path = "../../../ops/cuda/semiring_matmul.rs"]
mod semiring_matmul;

#[path = "../../../ops/cuda/compare.rs"]
mod compare;

#[path = "../../../ops/cuda/scalar.rs"]
mod scalar;

#[path = "../../../ops/cuda/logical.rs"]
mod logical;

#[path = "../../../ops/cuda/einsum.rs"]
mod einsum;

#[cfg(feature = "fp8")]
#[path = "../../../ops/cuda/fp8_matmul.rs"]
mod fp8_matmul;

#[cfg(feature = "sparse")]
#[path = "../../../ops/cuda/sparse_24.rs"]
mod sparse_24;
