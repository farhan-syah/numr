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

/// Brackets MIN_OUTPUT_LENGTH, the im2col dispatch threshold in
/// `src/ops/cuda/conv1d_im2col.rs`. Same family as the oc4 cases above, so only
/// output_length varies: length 18 - (kernel 7 - 1) = 12.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_1536ch_k7_lout12(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 18], &device);
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

/// Brackets MIN_OUTPUT_LENGTH, the im2col dispatch threshold in
/// `src/ops/cuda/conv1d_im2col.rs`. Same family as the oc4 cases above, so only
/// output_length varies: length 26 - (kernel 7 - 1) = 20.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_1536ch_k7_lout20(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 26], &device);
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
// conv1d — depthwise (groups == c_in == c_out)
// ---------------------------------------------------------------------------
//
// c_out_per_group = c_out/groups = 1 < CONV1D_OC_BLOCK (4) -> scalar conv1d
// kernel on every case below, regardless of channel count. `groups=4` in
// `cuda_conv1d_oc4_1536ch_k7_groups4` above still has c_out_per_group=384, so
// it never touches this path.

/// Mamba `d_inner`-scale decode step: single new token, K=4 causal window
/// already provided by the cache so no padding is needed. L_out=1.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_depthwise_decode_1536ch_k4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 4], &device);
    let weight = rand_cuda(&[1536, 1, 4], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1536)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Same decode shape, wider channel count and batched: batch=4, c=4096.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_depthwise_decode_wide_4096ch_k4_batch4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[4, 4096, 4], &device);
    let weight = rand_cuda(&[4096, 1, 4], &device);
    let bias = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 4096)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Full-sequence causal depthwise conv: K=4, left padding K-1=3, so
/// L_out = L = 32 (output length matches input, the standard Mamba causal
/// conv1d shape for a short prefill/chunk).
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_depthwise_short_1536ch_k4_l32(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 32], &device);
    let weight = rand_cuda(&[1536, 1, 4], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(
                    &input,
                    &weight,
                    Some(&bias),
                    1,
                    PaddingMode::conv1d(3, 0),
                    1,
                    1536,
                )
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Same causal shape, long sequence: L=1024 -> L_out=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_depthwise_long_1536ch_k4_l1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 1536, 1024], &device);
    let weight = rand_cuda(&[1536, 1, 4], &device);
    let bias = rand_cuda(&[1536], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(
                    &input,
                    &weight,
                    Some(&bias),
                    1,
                    PaddingMode::conv1d(3, 0),
                    1,
                    1536,
                )
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

/// Long causal sequence, wider channels and batched: batch=2, c=4096,
/// L=2048 -> L_out=2048.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_depthwise_long_wide_4096ch_k4_l2048_batch2(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[2, 4096, 2048], &device);
    let weight = rand_cuda(&[4096, 1, 4], &device);
    let bias = rand_cuda(&[4096], &device);
    b.iter(|| {
        let r = black_box(
            client
                .conv1d(
                    &input,
                    &weight,
                    Some(&bias),
                    1,
                    PaddingMode::conv1d(3, 0),
                    1,
                    4096,
                )
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// conv1d — narrow output, still oc4 path (c_out/groups >= 4)
// ---------------------------------------------------------------------------

/// c_out_per_group = 512 >= CONV1D_OC_BLOCK (4) -> conv1d_oc4, but L_out=1
/// (like the depthwise decode cases above): isolates launch-geometry effects
/// on the register-blocked kernel from the depthwise scalar kernel above.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_oc4_narrow_output_256_512ch_k3_lout1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 256, 3], &device);
    let weight = rand_cuda(&[512, 256, 3], &device);
    let bias = rand_cuda(&[512], &device);
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
