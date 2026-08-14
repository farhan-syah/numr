//! SIMD 1D convolution, vectorised over output positions.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod dispatch;
mod driver;
#[cfg(target_arch = "aarch64")]
mod neon;

pub use dispatch::{conv1d_f32, conv1d_f64};
