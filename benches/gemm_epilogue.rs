#![allow(dead_code)]
//! Fused GEMM-epilogue CUDA benchmarks: matmul_bias_activation, matmul_bias_residual.
//!
//! CUDA-only: `src/runtime/cuda/kernels/gemm_epilogue/` has no other benchmark coverage,
//! so a tile-config change on this path was previously unmeasurable.

use fluxbench::{Bencher, flux};
use std::hint::black_box;

#[cfg(feature = "cuda")]
use numr::ops::{GemmActivation, GemmEpilogueOps};
#[cfg(feature = "cuda")]
use numr::prelude::*;

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

#[cfg(feature = "cuda")]
fn rand_cuda_f64(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F64).unwrap()
}

// ---------------------------------------------------------------------------
// matmul_bias_activation: non-batched, F32
// ---------------------------------------------------------------------------

/// Large square: M=N=K=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_act_f32")]
fn cuda_gemm_bias_act_1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[1024, 1024], &device);
    let bm = rand_cuda(&[1024, 1024], &device);
    let bias = rand_cuda(&[1024], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_activation(&a, &bm, &bias, GemmActivation::ReLU)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Skewed: M=512, K=4096, N=1024 — tile choice matters most when M != N.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_act_f32")]
fn cuda_gemm_bias_act_512m_4096k_1024n(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[512, 4096], &device);
    let bm = rand_cuda(&[4096, 1024], &device);
    let bias = rand_cuda(&[1024], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_activation(&a, &bm, &bias, GemmActivation::GELU)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// matmul_bias_activation: batched, F32
// ---------------------------------------------------------------------------

/// Batched: batch=4, M=512, K=1024, N=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_act_batched_f32")]
fn cuda_gemm_bias_act_batched_4x512x1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[4, 512, 1024], &device);
    let bm = rand_cuda(&[4, 1024, 1024], &device);
    let bias = rand_cuda(&[1024], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_activation(&a, &bm, &bias, GemmActivation::SiLU)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// matmul_bias_residual: non-batched, F32
// ---------------------------------------------------------------------------

/// Large square: M=N=K=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_residual_f32")]
fn cuda_gemm_bias_residual_1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[1024, 1024], &device);
    let bm = rand_cuda(&[1024, 1024], &device);
    let bias = rand_cuda(&[1024], &device);
    let residual = rand_cuda(&[1024, 1024], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_residual(&a, &bm, &bias, &residual)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Skewed: M=512, K=4096, N=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_residual_f32")]
fn cuda_gemm_bias_residual_512m_4096k_1024n(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[512, 4096], &device);
    let bm = rand_cuda(&[4096, 1024], &device);
    let bias = rand_cuda(&[1024], &device);
    let residual = rand_cuda(&[512, 1024], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_residual(&a, &bm, &bias, &residual)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// matmul_bias_activation: F64 (different tile branch than F32)
// ---------------------------------------------------------------------------

/// F64: M=N=K=512.
#[cfg(feature = "cuda")]
#[flux::bench(group = "gemm_epilogue_bias_act_f64")]
fn cuda_gemm_bias_act_f64_512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda_f64(&[512, 512], &device);
    let bm = rand_cuda_f64(&[512, 512], &device);
    let bias = rand_cuda_f64(&[512], &device);
    b.iter(|| {
        let r = black_box(
            client
                .matmul_bias_activation(&a, &bm, &bias, GemmActivation::ReLU)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

fn main() {
    fluxbench::run().unwrap();
}
