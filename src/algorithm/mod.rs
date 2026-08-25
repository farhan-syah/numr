//! Algorithm contracts for runtime backends
//!
//! This module defines trait-based contracts that ensure all backends implement
//! the same mathematical algorithms, guaranteeing numerical parity across
//! CPU, CUDA, WebGPU, ROCm, and other compute devices.
//!
//! # Trait-First Architecture
//!
//! Every algorithm is defined as a trait FIRST, then implemented per backend:
//!
//! 1. **Define trait here** - Specifies exact algorithm with pseudocode/description
//! 2. **Implement for each backend** - CPU, CUDA, WebGPU follow same algorithm
//! 3. **Compile-time enforcement** - Missing implementations cause errors
//!
//! # Available Algorithm Contracts
//!
//! - [`MatmulAlgorithm`] - Tiled GEMM with register blocking for dense matmul
//! - [`LinearAlgebraAlgorithms`] - LU, Cholesky, QR, solve, inverse, det, trace
//! - `FftAlgorithms` - FFT, IFFT, RFFT using Stockham autosort algorithm
//! - `SparseAlgorithms` - SpGEMM, SpMV, DSMM (requires `sparse` feature)
//!
//! # Signal Processing
//!
//! Signal processing operations (convolution, STFT, window functions) are provided
//! by the **solvr** library, which builds on numr's FFT primitives.
//!
//! # Universal Algorithm Design
//!
//! Algorithms are designed to be portable across all backends:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Algorithm Trait                         │
//! │  (defines: tile sizes, register blocking, pseudocode)   │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │
//!     ┌──────────────────┼──────────────────┐
//!     ▼                  ▼                  ▼
//! ┌─────────┐      ┌──────────┐      ┌───────────┐
//! │   CPU   │      │   CUDA   │      │   WebGPU  │
//! │ (SIMD)  │      │ (shared  │      │ (workgrp  │
//! │         │      │  memory) │      │  memory)  │
//! └─────────┘      └──────────┘      └───────────┘
//! ```

pub mod fft;
pub mod fft_bluestein;
pub mod linalg;
pub mod matmul;
pub mod polynomial;
pub mod special;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "sparse")]
pub mod sparse_linalg;

#[cfg(feature = "sparse")]
pub mod iterative;

pub use linalg::{
    CholeskyDecomposition, EigenDecomposition, GeneralEigenDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, QrDecomposition,
    SchurDecomposition, SvdDecomposition, linalg_working_dtype, machine_epsilon,
    validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};

pub use matmul::{MatmulAlgorithm, TileConfig};

pub use polynomial::{
    PolynomialAlgorithms, PolynomialRoots, validate_polynomial_coeffs, validate_polynomial_dtype,
    validate_polynomial_roots,
};

pub use special::{
    EULER_MASCHERONI, LANCZOS_COEFFICIENTS, LANCZOS_G, LN_SQRT_2PI, SQRT_PI, SpecialFunctions,
    TWO_OVER_SQRT_PI, validate_special_dtype,
};

#[cfg(feature = "sparse")]
pub use sparse::{
    SparseAlgorithms, validate_dsmm_shapes, validate_dtype_match, validate_spgemm_shapes,
    zero_tolerance,
};

#[cfg(feature = "sparse")]
pub use sparse_linalg::{
    // Types
    IcDecomposition,
    IcOptions,
    IluDecomposition,
    IluFillLevel,
    IluMetrics,
    IluOptions,
    IlukDecomposition,
    IlukOptions,
    IlukSymbolic,
    // Level scheduling
    LevelSchedule,
    // Trait and validation
    SparseLinAlgAlgorithms,
    SymbolicIlu0,
    compute_levels_csc_lower,
    compute_levels_csc_upper,
    compute_levels_ilu,
    compute_levels_lower,
    compute_levels_upper,
    flatten_levels,
    // CPU implementations
    ic0_cpu,
    ilu0_cpu,
    ilu0_numeric_cpu,
    ilu0_symbolic_cpu,
    iluk_cpu,
    iluk_numeric_cpu,
    iluk_symbolic_cpu,
    sparse_solve_triangular_cpu,
    validate_square_sparse,
    validate_triangular_solve_dims,
};

#[cfg(feature = "sparse")]
pub use iterative::{
    // Types
    AdaptiveGmresResult,
    AdaptivePreconditionerOptions,
    BiCgStabOptions,
    BiCgStabResult,
    CgOptions,
    CgResult,
    CgsOptions,
    CgsResult,
    ConvergenceReason,
    GmresDiagnostics,
    GmresOptions,
    GmresResult,
    // Trait and validation
    IterativeSolvers,
    MinresOptions,
    MinresResult,
    PreconditionerType,
    SparseEigComplexResult,
    SparseEigOptions,
    SparseEigResult,
    StagnationParams,
    WhichEigenvalues,
    // Implementations
    adaptive_gmres_impl,
    arnoldi_eig_impl,
    bicgstab_impl,
    cg_impl,
    cgs_impl,
    gmres_impl,
    lanczos_eig_impl,
    minres_impl,
    validate_iterative_inputs,
};
