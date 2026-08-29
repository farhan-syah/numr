//! CPU runtime tensor operation implementations
//!
//! This module loads all operation implementations from the src/ops/cpu/ directory.
//! Each operation type (unary, binary, reduce, etc.) is implemented in its own module.

use super::{CpuClient, CpuRuntime};
use crate::ops::TensorOps;

// TensorOps is a supertrait that composes all individual operation traits.
// Since we implement all the component traits in separate files, we just need
// this empty impl to satisfy the supertrait requirement.
impl TensorOps<CpuRuntime> for CpuClient {}

// Load all CPU operation implementations from src/ops/cpu/
#[path = "../../ops/cpu/type_conversion.rs"]
mod type_conversion;

#[path = "../../ops/cpu/complex.rs"]
mod complex;

#[path = "../../ops/cpu/normalization.rs"]
mod normalization;

#[path = "../../ops/cpu/matmul.rs"]
mod matmul;

// I8 matmul widens to I32, so it cannot share `matmul`'s same-dtype
// generic path and lives in its own module.
#[path = "../../ops/cpu/matmul_i8.rs"]
pub mod matmul_i8;

// Column-split parallelism for the tiled matmul paths. Rayon-only: without it
// there is no pool to split across and `matmul` takes its serial branch.
#[cfg(feature = "rayon")]
#[path = "../../ops/cpu/matmul_columns.rs"]
mod matmul_columns;

#[path = "../../ops/cpu/conv.rs"]
mod conv;

#[path = "../../ops/cpu/cumulative.rs"]
mod cumulative;

#[path = "../../ops/cpu/activation.rs"]
mod activation;

#[path = "../../ops/cpu/binary.rs"]
mod binary;

#[path = "../../ops/cpu/unary.rs"]
mod unary;

#[path = "../../ops/cpu/linalg.rs"]
mod linalg;

#[path = "../../ops/cpu/statistics.rs"]
mod statistics;

#[path = "../../ops/cpu/random.rs"]
mod random;

#[path = "../../ops/cpu/advanced_random.rs"]
mod advanced_random;

#[path = "../../ops/cpu/quasirandom.rs"]
mod quasirandom;

#[path = "../../ops/cpu/reduce.rs"]
mod reduce;

#[path = "../../ops/cpu/sorting.rs"]
mod sorting;

#[path = "../../ops/cpu/conditional.rs"]
mod conditional;

#[path = "../../ops/cpu/utility.rs"]
mod utility;

#[path = "../../ops/cpu/scalar.rs"]
mod scalar;

#[path = "../../ops/cpu/compare.rs"]
mod compare;

#[path = "../../ops/cpu/logical.rs"]
mod logical;

#[path = "../../ops/cpu/indexing.rs"]
mod indexing;

#[path = "../../ops/cpu/shape.rs"]
mod shape;

#[path = "../../ops/cpu/distance.rs"]
mod distance;

#[path = "../../ops/cpu/multivariate.rs"]
mod multivariate;

#[path = "../../ops/cpu/semiring_matmul.rs"]
mod semiring_matmul;

#[path = "../../ops/cpu/einsum.rs"]
mod einsum;

#[path = "../../ops/cpu/gemm_epilogue.rs"]
mod gemm_epilogue;

#[cfg(feature = "fp8")]
#[path = "../../ops/cpu/fp8_matmul.rs"]
mod fp8_matmul;

#[cfg(feature = "sparse")]
#[path = "../../ops/cpu/sparse_24.rs"]
mod sparse_24;
