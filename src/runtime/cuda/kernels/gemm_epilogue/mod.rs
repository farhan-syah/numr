//! CUDA GEMM epilogue kernels and launchers.

mod bwd_launcher;
mod launcher;
mod tiled_f32;

pub use bwd_launcher::{launch_gemm_bias_act_bwd_batched_kernel, launch_gemm_bias_act_bwd_kernel};
pub use launcher::{
    launch_gemm_bias_act_batched_kernel, launch_gemm_bias_act_kernel,
    launch_gemm_bias_residual_batched_kernel, launch_gemm_bias_residual_kernel,
};
