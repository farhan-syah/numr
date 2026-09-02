#![allow(dead_code)]
//! CUDA normalization benchmarks: rms_norm, layer_norm, fused_add_rms_norm.
//!
//! All three are one-block-per-row reductions over the last dimension
//! (`src/ops/cuda/normalization.rs`), the same kernel shape as softmax.
//! `rms_norm` and `layer_norm` are benched on identical shapes deliberately:
//! `rms_norm` reduces mean(x^2) alone, `layer_norm` needs both mean and
//! variance, so the two use different reduction strategies and are only
//! comparable when the shape is held fixed. Shapes vary row width and row
//! count independently so each can be isolated.

// Every benchmark here targets a CUDA kernel, so without that feature the file
// has no bodies and even the harness imports are unused.
#[cfg(feature = "cuda")]
use fluxbench::{Bencher, flux};
#[cfg(feature = "cuda")]
use std::hint::black_box;

#[cfg(feature = "cuda")]
use numr::ops::NormalizationOps;
#[cfg(feature = "cuda")]
use numr::prelude::*;

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------

/// LLM hidden-size row, many rows: [4, 512, 4096].
#[cfg(feature = "cuda")]
#[flux::bench(group = "rms_norm_f32")]
fn cuda_rms_norm_4x512x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 4096], &device);
    let weight = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(client.rms_norm(&input, &weight, 1e-6).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Few rows, same width as above: [1, 8, 4096] — isolates row count.
#[cfg(feature = "cuda")]
#[flux::bench(group = "rms_norm_f32")]
fn cuda_rms_norm_1x8x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 8, 4096], &device);
    let weight = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(client.rms_norm(&input, &weight, 1e-6).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Wider row: [4, 512, 8192].
#[cfg(feature = "cuda")]
#[flux::bench(group = "rms_norm_f32")]
fn cuda_rms_norm_4x512x8192(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 8192], &device);
    let weight = rand_cuda(&[8192], &device);
    b.iter(|| {
        let r = black_box(client.rms_norm(&input, &weight, 1e-6).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Many more rows, narrower: [32, 512, 1024].
#[cfg(feature = "cuda")]
#[flux::bench(group = "rms_norm_f32")]
fn cuda_rms_norm_32x512x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[32, 512, 1024], &device);
    let weight = rand_cuda(&[1024], &device);
    b.iter(|| {
        let r = black_box(client.rms_norm(&input, &weight, 1e-6).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// layer_norm — same four shapes as rms_norm, for direct comparison
// ---------------------------------------------------------------------------

/// LLM hidden-size row, many rows: [4, 512, 4096].
#[cfg(feature = "cuda")]
#[flux::bench(group = "layer_norm_f32")]
fn cuda_layer_norm_4x512x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 4096], &device);
    let weight = rand_cuda(&[4096], &device);
    let bias = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(client.layer_norm(&input, &weight, &bias, 1e-5).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Few rows, same width as above: [1, 8, 4096] — isolates row count.
#[cfg(feature = "cuda")]
#[flux::bench(group = "layer_norm_f32")]
fn cuda_layer_norm_1x8x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 8, 4096], &device);
    let weight = rand_cuda(&[4096], &device);
    let bias = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(client.layer_norm(&input, &weight, &bias, 1e-5).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Wider row: [4, 512, 8192].
#[cfg(feature = "cuda")]
#[flux::bench(group = "layer_norm_f32")]
fn cuda_layer_norm_4x512x8192(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 8192], &device);
    let weight = rand_cuda(&[8192], &device);
    let bias = rand_cuda(&[8192], &device);
    b.iter(|| {
        let r = black_box(client.layer_norm(&input, &weight, &bias, 1e-5).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Many more rows, narrower: [32, 512, 1024].
#[cfg(feature = "cuda")]
#[flux::bench(group = "layer_norm_f32")]
fn cuda_layer_norm_32x512x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[32, 512, 1024], &device);
    let weight = rand_cuda(&[1024], &device);
    let bias = rand_cuda(&[1024], &device);
    b.iter(|| {
        let r = black_box(client.layer_norm(&input, &weight, &bias, 1e-5).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// fused_add_rms_norm — the residual-connection hot path
// ---------------------------------------------------------------------------

/// LLM hidden-size row: [4, 512, 4096].
#[cfg(feature = "cuda")]
#[flux::bench(group = "fused_add_norm_f32")]
fn cuda_fused_add_rms_norm_4x512x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let x = rand_cuda(&[4, 512, 4096], &device);
    let residual = rand_cuda(&[4, 512, 4096], &device);
    let weight = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(
            client
                .fused_add_rms_norm(&x, &residual, &weight, 1e-6)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Wider row: [4, 512, 8192].
#[cfg(feature = "cuda")]
#[flux::bench(group = "fused_add_norm_f32")]
fn cuda_fused_add_rms_norm_4x512x8192(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let x = rand_cuda(&[4, 512, 8192], &device);
    let residual = rand_cuda(&[4, 512, 8192], &device);
    let weight = rand_cuda(&[8192], &device);
    b.iter(|| {
        let r = black_box(
            client
                .fused_add_rms_norm(&x, &residual, &weight, 1e-6)
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
