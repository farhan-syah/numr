//! CPU implementation of tensor operations.
//!
//! This module contains the operation trait implementations for the CPU runtime.
//! Each operation type has its own module.

pub mod activation;
pub mod advanced_random;
pub mod binary;
pub mod compare;
pub mod complex;
pub mod conditional;
pub mod conv;
pub mod cumulative;
pub mod distance;
pub mod einsum;
#[cfg(feature = "fp8")]
pub mod fp8_matmul;
pub mod gemm_epilogue;
pub mod indexing;
pub mod linalg;
pub mod logical;
pub mod matmul;
pub mod matmul_i8;
pub mod multivariate;
pub mod normalization;
pub mod quasirandom;
pub mod random;
pub mod reduce;
pub mod scalar;
pub mod semiring_matmul;
pub mod shape;
pub mod sorting;
#[cfg(feature = "sparse")]
pub mod sparse_24;
pub mod statistics;
pub mod type_conversion;
pub mod unary;
pub mod utility;
