//! WebGPU runtime tensor operation implementations
//!
//! This module loads all operation implementations from the src/ops/wgpu/ directory.
//! Each operation type (type conversion, complex, normalization, etc.) is implemented in its own module.

use super::super::{WgpuClient, WgpuRuntime};
use crate::ops::TensorOps;

// TensorOps is a supertrait that composes all individual operation traits.
// Since we implement all the component traits in separate files, we just need
// this empty impl to satisfy the supertrait requirement.
impl TensorOps<WgpuRuntime> for WgpuClient {}

// Load all WebGPU operation implementations from src/ops/wgpu/.
//
// THIS LIST IS THE MODULE TREE for src/ops/wgpu/. That directory has no mod.rs
// and `crate::ops` declares no `mod wgpu`, so a file added under src/ops/wgpu/
// is compiled ONLY if it gets a `#[path]` + `mod` entry here. Subdirectories
// (e.g. `indexing/`) keep their own mod.rs, reached through the entry below.
#[path = "../../../ops/wgpu/type_conversion.rs"]
mod type_conversion;

#[path = "../../../ops/wgpu/complex.rs"]
mod complex;

#[path = "../../../ops/wgpu/normalization.rs"]
mod normalization;

#[path = "../../../ops/wgpu/matmul.rs"]
mod matmul;

#[path = "../../../ops/wgpu/conv.rs"]
mod conv;

#[path = "../../../ops/wgpu/cumulative.rs"]
mod cumulative;

#[path = "../../../ops/wgpu/activation.rs"]
mod activation;

#[path = "../../../ops/wgpu/binary.rs"]
mod binary;

#[path = "../../../ops/wgpu/unary.rs"]
mod unary;

#[path = "../../../ops/wgpu/random.rs"]
mod random;

#[path = "../../../ops/wgpu/quasirandom.rs"]
mod quasirandom;

#[path = "../../../ops/wgpu/advanced_random.rs"]
mod advanced_random;

#[path = "../../../ops/wgpu/linalg.rs"]
mod linalg;

#[path = "../../../ops/wgpu/shape.rs"]
mod shape;

#[path = "../../../ops/wgpu/statistics.rs"]
mod statistics;

#[path = "../../../ops/wgpu/sorting.rs"]
mod sorting;

#[path = "../../../ops/wgpu/indexing/mod.rs"]
mod indexing;

#[path = "../../../ops/wgpu/reduce.rs"]
mod reduce;

#[path = "../../../ops/wgpu/conditional.rs"]
mod conditional;

#[path = "../../../ops/wgpu/utility.rs"]
mod utility;

#[path = "../../../ops/wgpu/distance.rs"]
mod distance;

#[path = "../../../ops/wgpu/multivariate.rs"]
mod multivariate;

#[path = "../../../ops/wgpu/gemm_epilogue.rs"]
mod gemm_epilogue;

#[path = "../../../ops/wgpu/semiring_matmul.rs"]
mod semiring_matmul;

#[path = "../../../ops/wgpu/logical.rs"]
mod logical;

#[path = "../../../ops/wgpu/compare.rs"]
mod compare;

#[path = "../../../ops/wgpu/scalar.rs"]
mod scalar;

#[path = "../../../ops/wgpu/einsum.rs"]
mod einsum;

#[path = "../../../ops/wgpu/fp8_matmul.rs"]
mod fp8_matmul;

#[cfg(feature = "sparse")]
#[path = "../../../ops/wgpu/sparse_24.rs"]
mod sparse_24;
