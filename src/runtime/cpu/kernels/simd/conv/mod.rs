//! SIMD-accelerated convolution kernels: conv1d, conv2d, depthwise_conv2d and
//! conv_transpose1d. See `conv2d` for the dispatch strategy.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod conv1d;
mod conv2d;
mod depthwise_conv2d;
#[cfg(feature = "f16")]
mod half;
mod scalar;
#[cfg(test)]
mod tests;
mod threshold;
mod transpose1d;

pub use conv1d::{conv1d_f32, conv1d_f64};
pub use conv2d::{conv2d_f32, conv2d_f64};
pub use depthwise_conv2d::{depthwise_conv2d_f32, depthwise_conv2d_f64};
#[cfg(feature = "f16")]
pub use half::*;
pub use transpose1d::{conv_transpose1d_f32, conv_transpose1d_f64};
