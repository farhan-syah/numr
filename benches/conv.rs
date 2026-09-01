#![allow(dead_code)]
//! CUDA convolution benchmarks: conv1d, conv_transpose1d, conv2d, depthwise_conv2d.
//!
//! `conv1d_oc4_f32` and `conv_transpose1d_f32` accounted for 34.9% of GPU time in a
//! real TTS workload profile, and had no numr-level benchmark coverage. `conv1d`
//! dispatches to the `conv1d_oc4` kernel when `c_out_per_group >= 4`
//! (`src/runtime/cuda/kernels/conv.rs`, `CONV1D_OC_BLOCK = 4`), else the scalar
//! `conv1d` kernel. `conv_transpose1d` has no such split — one kernel for all shapes.

use fluxbench::{Bencher, flux};
use std::hint::black_box;

#[cfg(feature = "cuda")]
use numr::ops::{ConvOps, PaddingMode};
#[cfg(feature = "cuda")]
use numr::prelude::*;

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

// ---------------------------------------------------------------------------
// conv1d — oc4 path (c_out_per_group >= 4)
// ---------------------------------------------------------------------------

/// TTS/audio hot shape: batch=1, c_in=c_out=1536, k=7, L=32 -> L_out=26.
/// Cited directly in the conv1d_oc4 kernel's own comment as the realistic case.
/// c_out_per_group = 1536 >= 4 -> conv1d_oc4 path.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_1536ch_k7_lout26(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 32], &device);
    let weight = rand_cuda(&[1536, 1536, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Same channels/kernel as above, larger output length: L=306 -> L_out=300.
/// Separates "small L_out / low occupancy" from "large work" on the oc4 path.
/// c_out_per_group = 1536 >= 4 -> conv1d_oc4 path.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_1536ch_k7_lout300(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 306], &device);
    let weight = rand_cuda(&[1536, 1536, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// conv1d — scalar path (c_out_per_group < 4)
// ---------------------------------------------------------------------------

/// c_out=3 -> c_out_per_group = 3 < CONV1D_OC_BLOCK (4) -> scalar conv1d kernel.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_scalar_cout3_k7_l64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 64, 64], &device);
    let weight = rand_cuda(&[3, 64, 7], &device);
    let bias = rand_cuda(&[3], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// conv1d — grouped
// ---------------------------------------------------------------------------

/// groups=4, c_out_per_group = 1536/4 = 384 >= 4 -> conv1d_oc4 path, grouped.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_1536ch_k7_groups4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 32], &device);
    let weight = rand_cuda(&[1536, 1536 / 4, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 4)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// conv_transpose1d — optimization target
// ---------------------------------------------------------------------------

/// Mirrors cuda_conv1d_oc4_1536ch_k7_lout26: same channels/kernel, L_in=20 -> L_out=26.
/// Weight layout is (C_in, C_out/groups, K) per ConvOps::conv_transpose1d.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv_transpose1d_f32")]
fn cuda_conv_transpose1d_1536ch_k7_lout26(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 20], &device);
    let weight = rand_cuda(&[1536, 1536, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv_transpose1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 0, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Mirrors cuda_conv1d_oc4_1536ch_k7_lout300: same channels/kernel, L_in=294 -> L_out=300.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv_transpose1d_f32")]
fn cuda_conv_transpose1d_1536ch_k7_lout300(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 294], &device);
    let weight = rand_cuda(&[1536, 1536, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv_transpose1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 0, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Stride-2 upsampling: c_in=c_out=512, k=4, L_in=64 -> L_out=130. Typical
/// vocoder/decoder usage (upsample a latent sequence toward waveform length).
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv_transpose1d_f32")]
fn cuda_conv_transpose1d_512ch_k4_stride2_upsample(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 64], &device);
    let weight = rand_cuda(&[512, 512, 4], &device);
    let bias = rand_cuda(&[512], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv_transpose1d(&input, &weight, Some(&bias), 2, PaddingMode::Valid, 0, 1, 1)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Multi-channel grouped case: groups=4, same channels/kernel as the lout26 shape.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv_transpose1d_f32")]
fn cuda_conv_transpose1d_1536ch_k7_groups4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 20], &device);
    let weight = rand_cuda(&[1536, 1536 / 4, 7], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv_transpose1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 0, 1, 4)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// conv2d / depthwise_conv2d — baseline coverage
// ---------------------------------------------------------------------------

/// Representative image-conv shape: batch=1, c_in=64, c_out=128, 3x3 kernel, 56x56.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv2d_f32")]
fn cuda_conv2d_64_128ch_k3_56x56(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 64, 56, 56], &device);
    let weight = rand_cuda(&[128, 64, 3, 3], &device);
    let bias = rand_cuda(&[128], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv2d(
                    &input,
                    &weight,
                    Some(&bias),
                    (1, 1),
                    PaddingMode::Valid,
                    (1, 1),
                    1,
                )
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Representative depthwise shape: 32 channels, 3x3 kernel, 112x112 (mobile-net-style
/// early layer).
#[cfg(feature = "cuda")]
#[flux::bench(group = "depthwise_conv2d_f32")]
fn cuda_depthwise_conv2d_32ch_k3_112x112(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 32, 112, 112], &device);
    let weight = rand_cuda(&[32, 1, 3, 3], &device);
    let bias = rand_cuda(&[32], &device);
    b.iter(|| {
        let r = black_box(
            client
                .depthwise_conv2d(
                    &input,
                    &weight,
                    Some(&bias),
                    (1, 1),
                    PaddingMode::Valid,
                    (1, 1),
                )
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
