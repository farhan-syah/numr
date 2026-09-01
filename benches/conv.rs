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
// conv1d — im2col dispatch corner (MIN_CONTRACTION / MIN_C_OUT_PER_GROUP floor)
// ---------------------------------------------------------------------------
//
// `src/ops/cuda/conv1d_im2col.rs` gates the im2col path on MIN_CONTRACTION=64,
// MIN_C_OUT_PER_GROUP=4, MIN_OUTPUT_LENGTH=4, MAX_COL_ELEMENTS=1<<26. A sweep
// on c_in=c_out=1536, K=7 found im2col faster at every output_length from 1
// to 300, so MIN_OUTPUT_LENGTH dropped from 16 to 4. That sweep never
// covered c_out at its MIN_C_OUT_PER_GROUP floor with contraction at its
// MIN_CONTRACTION floor, where the GEMM is degenerate. These cases close
// that gap. Every case below has groups=1, contraction >= MIN_CONTRACTION,
// c_out/groups >= MIN_C_OUT_PER_GROUP, and col_elements << MAX_COL_ELEMENTS,
// so only output_length varies within each family.

/// Both floors at once: c_in=16, c_out=4, K=4 -> contraction=64=MIN_CONTRACTION,
/// c_out/groups=4=MIN_C_OUT_PER_GROUP. Shortest output_length routed through
/// im2col: output_length=1 -> input length=1+3=4. col_elements=16*4*1=64.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_floor_c16_c4_k4_lout1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 16, 4], &device);
    let weight = rand_cuda(&[4, 16, 4], &device);
    let bias = rand_cuda(&[4], &device);
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

/// Same floor family, output_length=4=MIN_OUTPUT_LENGTH -> input length=4+3=7.
/// col_elements=16*4*4=256.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_floor_c16_c4_k4_lout4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 16, 7], &device);
    let weight = rand_cuda(&[4, 16, 4], &device);
    let bias = rand_cuda(&[4], &device);
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

/// Same floor family, output_length=16 -> input length=16+3=19.
/// col_elements=16*4*16=1024.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_floor_c16_c4_k4_lout16(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 16, 19], &device);
    let weight = rand_cuda(&[4, 16, 4], &device);
    let bias = rand_cuda(&[4], &device);
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

/// Same floor family, output_length=64 -> input length=64+3=67.
/// col_elements=16*4*64=4096.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_floor_c16_c4_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 16, 67], &device);
    let weight = rand_cuda(&[4, 16, 4], &device);
    let bias = rand_cuda(&[4], &device);
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

/// Midpoint between the floor family above and the 1536-channel family:
/// c_in=c_out=32, K=4 -> contraction=128, c_out/groups=32. output_length=4 ->
/// input length=4+3=7. col_elements=32*4*4=512.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_mid_c32_c32_k4_lout4(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 32, 7], &device);
    let weight = rand_cuda(&[32, 32, 4], &device);
    let bias = rand_cuda(&[32], &device);
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

/// Same midpoint family, output_length=64 -> input length=64+3=67.
/// col_elements=32*4*64=8192.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_mid_c32_c32_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 32, 67], &device);
    let weight = rand_cuda(&[32, 32, 4], &device);
    let bias = rand_cuda(&[32], &device);
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
// conv1d — im2col work-axis sweep (c_in=c_out, K=4, output_length=64)
// ---------------------------------------------------------------------------
//
// Brackets the gap between two measured points on the c_in*c_out*K*output_length
// work axis: cuda_conv1d_im2col_mid_c32_c32_k4_lout64 (work ~262K, im2col loses)
// and the 1536-channel, K=7, lout1 family (work ~16.5M, im2col wins). These four
// cases hold K=4 and output_length=64 fixed and step c_in=c_out from 64 to 512
// to locate the crossover. Each satisfies groups=1, contraction (c_in*K) >=
// MIN_CONTRACTION (64), c_out/groups >= MIN_C_OUT_PER_GROUP (4), and
// col_elements (c_in*K*output_length) << MAX_COL_ELEMENTS (1<<26).

/// Sweep point 1 of 4: c_in=c_out=64, K=4, output_length=64 -> input length=67.
/// contraction=256, col_elements=16384, work=64*256*64=1.05M.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_sweep_c64_c64_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 64, 67], &device);
    let weight = rand_cuda(&[64, 64, 4], &device);
    let bias = rand_cuda(&[64], &device);
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

/// Sweep point 2 of 4: c_in=c_out=128, K=4, output_length=64 -> input length=131.
/// contraction=512, col_elements=32768, work=128*512*64=4.2M.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_sweep_c128_c128_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 128, 131], &device);
    let weight = rand_cuda(&[128, 128, 4], &device);
    let bias = rand_cuda(&[128], &device);
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

/// Sweep point 3 of 4: c_in=c_out=256, K=4, output_length=64 -> input length=259.
/// contraction=1024, col_elements=65536, work=256*1024*64=16.8M.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_sweep_c256_c256_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 256, 259], &device);
    let weight = rand_cuda(&[256, 256, 4], &device);
    let bias = rand_cuda(&[256], &device);
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

/// Sweep point 4 of 4: c_in=c_out=512, K=4, output_length=64 -> input length=515.
/// contraction=2048, col_elements=131072, work=512*2048*64=67M.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_sweep_c512_c512_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 515], &device);
    let weight = rand_cuda(&[512, 512, 4], &device);
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

/// Deep contraction, few output channels. Separates contraction depth from c_out.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_deep_c512_c32_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 67], &device);
    let weight = rand_cuda(&[32, 512, 4], &device);
    let bias = rand_cuda(&[32], &device);
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

/// Shallow contraction, many output channels. The mirror of the deep case.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_wide_c32_c512_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 32, 67], &device);
    let weight = rand_cuda(&[512, 32, 4], &device);
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

/// Deep contraction at the c_out floor. Probes whether MIN_C_OUT_PER_GROUP still matters.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_deep_narrow_c512_c4_k4_lout64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 67], &device);
    let weight = rand_cuda(&[4, 512, 4], &device);
    let bias = rand_cuda(&[4], &device);
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

/// Deep contraction at output length 1. Probes whether MIN_OUTPUT_LENGTH still matters.
#[cfg(feature = "cuda")]
#[flux::bench(group = "conv1d_f32")]
fn cuda_conv1d_im2col_deep_c512_c512_k4_lout1(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, 512, 4], &device);
    let weight = rand_cuda(&[512, 512, 4], &device);
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
