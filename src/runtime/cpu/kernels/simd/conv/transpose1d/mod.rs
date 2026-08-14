//! SIMD transposed 1D convolution, vectorised over output positions.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod dispatch;
mod driver;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;

pub use dispatch::{conv_transpose1d_f32, conv_transpose1d_f64};
