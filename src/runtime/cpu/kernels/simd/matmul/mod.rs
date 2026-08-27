//! SIMD-optimized matrix multiplication.
//!
//! See [`dispatch`] for the public API and microkernel dispatch functions.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub(crate) mod dispatch;
pub(crate) mod gemv_bt;
// i32 matmul: AVX2 with an i64 accumulator when a magnitude prescan proves
// every partial sum fits, otherwise the exact i128 scalar path. An earlier AVX2
// kernel accumulated with `_mm256_add_epi32` and wrapped mid-dot-product; the
// guard is what makes a wider-but-still-finite accumulator safe.
#[cfg(target_arch = "x86_64")]
pub(crate) mod int32;
pub(crate) mod int8;
pub(crate) mod macros;
pub(crate) mod packing;
pub(crate) mod scalar;
pub(crate) mod small;
pub(crate) mod small_kernels;
pub(crate) mod tiling;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
pub(crate) mod half_convert;

pub use dispatch::{
    KC, MC, MR, NC, matmul_bias_f32, matmul_bias_f64, matmul_bt_f32, matmul_bt_f64,
    matmul_bt_is_tiled, matmul_f32, matmul_f64,
};

pub use dispatch::{
    call_microkernel_2x_f32, call_microkernel_2x_f64, call_microkernel_f32, call_microkernel_f64,
};
