//! Special mathematical functions for scientific computing
//!
//! This module re-exports the `SpecialFunctions` trait from the algorithm module.
//! The trait provides error functions, gamma functions, beta functions, and
//! incomplete gamma/beta functions needed for probability distributions.
//!
//! # Usage
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::ops::SpecialFunctions;
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! let x = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3], &device)?;
//! let erf_x = client.erf(&x)?;
//! # Ok::<(), numr::error::Error>(())
//! ```
//!
//! # Available Functions
//!
//! - **Error functions**: `erf`, `erfc`, `erfinv`
//! - **Gamma functions**: `gamma`, `lgamma`, `digamma`
//! - **Beta functions**: `beta`, `betainc`
//! - **Incomplete gamma**: `gammainc`, `gammaincc`

// Re-export from algorithm module - single source of truth
pub use crate::algorithm::special::SpecialFunctions;
