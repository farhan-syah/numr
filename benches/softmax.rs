#![allow(dead_code)]
//! CUDA softmax benchmarks: last-dim vs non-last-dim kernel.
//!
//! `softmax` dispatches on WHICH axis is reduced (`src/ops/cuda/activation.rs`):
//! reducing the last dimension takes the `launch_softmax` kernel (one block per
//! row, cooperative threads reducing over `dim_size`); reducing any other
//! dimension takes the separate `launch_softmax_dim` kernel
//! (`src/runtime/cuda/kernels/activation/softmax.rs`). The two groups below
//! benchmark each kernel independently. For the non-last-dim kernel,
//! `inner_size` (the product of dims after the reduced one) is the axis the
//! kernel parallelizes over, so shapes vary `inner_size` deliberately.

use fluxbench::{Bencher, flux};
use std::hint::black_box;

#[cfg(feature = "cuda")]
use numr::ops::ActivationOps;
#[cfg(feature = "cuda")]
use numr::prelude::*;

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

// ---------------------------------------------------------------------------
// softmax — last-dim path (launch_softmax)
// ---------------------------------------------------------------------------

/// LLM hidden-size row: [4, 512, 4096], reduce the last (4096) axis.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_last_dim_f32")]
fn cuda_softmax_last_dim_4x512x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 4096], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, -1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Vocab-size logits row: [1, 512, 32000], reduce the last (32000) axis.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_last_dim_f32")]
fn cuda_softmax_last_dim_1x512x32000(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 32000], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, -1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Attention-score row, many rows: [32, 512, 512], reduce the last (512) axis.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_last_dim_f32")]
fn cuda_softmax_last_dim_32x512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[32, 512, 512], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, -1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Few long rows: [1, 8, 4096], reduce the last (4096) axis.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_last_dim_f32")]
fn cuda_softmax_last_dim_1x8x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 8, 4096], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, -1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// softmax — non-last-dim path (launch_softmax_dim)
// ---------------------------------------------------------------------------

/// [4, 512, 4096], dim=1: reduces 512, inner_size = 4096.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_non_last_dim_f32")]
fn cuda_softmax_non_last_dim_4x512x4096_dim1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 4096], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, 1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// [4, 4096, 512], dim=1: reduces 4096, inner_size = 512.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_non_last_dim_f32")]
fn cuda_softmax_non_last_dim_4x4096x512_dim1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 4096, 512], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, 1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// [32, 512, 512], dim=1: reduces 512, inner_size = 512.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_non_last_dim_f32")]
fn cuda_softmax_non_last_dim_32x512x512_dim1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[32, 512, 512], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, 1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// [4, 512, 64], dim=1: reduces 512, small inner_size = 64.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_non_last_dim_f32")]
fn cuda_softmax_non_last_dim_4x512x64_dim1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 512, 64], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, 1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// [4, 64, 4096], dim=1: short reduction (64), large inner_size = 4096.
#[cfg(feature = "cuda")]
#[flux::bench(group = "softmax_non_last_dim_f32")]
fn cuda_softmax_non_last_dim_4x64x4096_dim1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 64, 4096], &device);
    b.iter(|| {
        let r = black_box(client.softmax(&input, 1).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

fn main() {
    fluxbench::run().unwrap();
}
