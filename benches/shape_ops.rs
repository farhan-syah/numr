#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (device, client)
}

fn rand_t(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

// ---------------------------------------------------------------------------
// repeat
// ---------------------------------------------------------------------------

#[flux::bench(group = "repeat_f32")]
fn numr_repeat_256x256_2x2(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[256, 256], &device);
    b.iter(|| black_box(client.repeat(&t, &[2, 2]).unwrap()));
}

#[flux::bench(group = "repeat_f32")]
fn numr_repeat_1024x64_4x1(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1024, 64], &device);
    b.iter(|| black_box(client.repeat(&t, &[4, 1]).unwrap()));
}

// ---------------------------------------------------------------------------
// repeat_interleave
// ---------------------------------------------------------------------------

#[flux::bench(group = "repeat_interleave_f32")]
fn numr_repeat_interleave_1k_x4(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000], &device);
    b.iter(|| black_box(client.repeat_interleave(&t, 4, Some(0)).unwrap()));
}

#[flux::bench(group = "repeat_interleave_f32")]
fn numr_repeat_interleave_256x64_x4(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[256, 64], &device);
    b.iter(|| black_box(client.repeat_interleave(&t, 4, Some(0)).unwrap()));
}

// ---------------------------------------------------------------------------
// unfold (sliding window)
// ---------------------------------------------------------------------------

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_10k_win64_step1(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 64, 1).unwrap()));
}

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_10k_win64_step32(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 64, 32).unwrap()));
}

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_100k_win256_step128(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 256, 128).unwrap()));
}

// ---------------------------------------------------------------------------
// cat (concatenation)
// ---------------------------------------------------------------------------

#[flux::bench(group = "cat_f32")]
fn numr_cat_10x_1000(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..10).map(|_| rand_t(&[1000], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.cat(&refs, 0).unwrap()));
}

#[flux::bench(group = "cat_f32")]
fn numr_cat_10x_256x64(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..10).map(|_| rand_t(&[256, 64], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.cat(&refs, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// stack
// ---------------------------------------------------------------------------

#[flux::bench(group = "stack_f32")]
fn numr_stack_8x_1000(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..8).map(|_| rand_t(&[1000], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.stack(&refs, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// split / chunk
// ---------------------------------------------------------------------------

#[flux::bench(group = "split_f32")]
fn numr_split_10k_into_100(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.split(&t, 100, 0).unwrap()));
}

#[flux::bench(group = "split_f32")]
fn numr_chunk_10k_into_10(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.chunk(&t, 10, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// CUDA benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "cat_f32")]
fn cuda_cat_10x_256x64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let tensors: Vec<_> = (0..10).map(|_| rand_cuda(&[256, 64], &device)).collect();
    let refs: Vec<&Tensor<CudaRuntime>> = tensors.iter().collect();
    b.iter(|| {
        let r = black_box(client.cat(&refs, 0).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "repeat_f32")]
fn cuda_repeat_256x256_2x2(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[256, 256], &device);
    b.iter(|| {
        let r = black_box(client.repeat(&t, &[2, 2]).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "stack_f32")]
fn cuda_stack_8x_1000(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let tensors: Vec<_> = (0..8).map(|_| rand_cuda(&[1000], &device)).collect();
    let refs: Vec<&Tensor<CudaRuntime>> = tensors.iter().collect();
    b.iter(|| {
        let r = black_box(client.stack(&refs, 0).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// contiguous: transposing views take the tiled strided_transpose.cu path
// when TransposePlan::detect recognizes them; everything else (including a
// full axis reversal, which detect declines) falls back to strided_copy.cu's
// one div+mod per dimension.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[flux::bench(group = "cuda_contiguous_f32")]
fn cuda_contiguous_2d_transpose_4096x4096(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[4096, 4096], &device);
    let view = t.transpose(0, 1).unwrap();
    b.iter(|| {
        let r = black_box(view.contiguous().unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "cuda_contiguous_f32")]
fn cuda_contiguous_3d_permute_256x256x256(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[256, 256, 256], &device);
    let view = t.permute(&[2, 0, 1]).unwrap();
    b.iter(|| {
        let r = black_box(view.contiguous().unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "cuda_contiguous_f32")]
fn cuda_contiguous_4d_permute_64x64x64x64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[64, 64, 64, 64], &device);
    // Same element count as cuda_contiguous_3d_permute_256x256x256 (256^3 == 64^4)
    // but one more dimension, isolating the per-dimension div+mod decode cost.
    let view = t.permute(&[3, 2, 1, 0]).unwrap();
    b.iter(|| {
        let r = black_box(view.contiguous().unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "cuda_contiguous_f32")]
fn cuda_contiguous_3d_swap_last_two_512x128x128(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[512, 128, 128], &device);
    // Outer dimension stays contiguous; only the last two axes swap strides.
    let view = t.transpose(1, 2).unwrap();
    b.iter(|| {
        let r = black_box(view.contiguous().unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "cuda_contiguous_f32")]
fn cuda_contiguous_2d_slice_8192x8192_half(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[8192, 8192], &device);
    // Contiguous-outer, strided-inner view: first half of the last axis.
    let view = t.narrow(1, 0, 4096).unwrap();
    b.iter(|| {
        let r = black_box(view.contiguous().unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// ndarray comparison: repeat via broadcast + to_owned
// ---------------------------------------------------------------------------

#[flux::bench(group = "cat_f32")]
fn ndarray_cat_10x_1000(b: &mut Bencher) {
    let vecs: Vec<ndarray::Array1<f32>> = (0..10)
        .map(|_| ndarray::Array1::from_vec((0..1000).map(|i| (i as f32) / 1000.0).collect()))
        .collect();
    let views: Vec<ndarray::ArrayView1<f32>> = vecs.iter().map(|a| a.view()).collect();
    b.iter(|| black_box(ndarray::concatenate(ndarray::Axis(0), &views).unwrap()));
}

#[flux::bench(group = "cat_f32")]
fn ndarray_cat_10x_256x64(b: &mut Bencher) {
    let vecs: Vec<ndarray::Array2<f32>> = (0..10)
        .map(|_| {
            ndarray::Array2::from_shape_vec(
                (256, 64),
                (0..256 * 64).map(|i| (i as f32) / 16384.0).collect(),
            )
            .unwrap()
        })
        .collect();
    let views: Vec<ndarray::ArrayView2<f32>> = vecs.iter().map(|a| a.view()).collect();
    b.iter(|| black_box(ndarray::concatenate(ndarray::Axis(0), &views).unwrap()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "cat_1d",
    title = "Concatenate 10x 1000-elem (numr vs ndarray)",
    benchmarks = ["numr_cat_10x_1000", "ndarray_cat_10x_1000"],
    baseline = "numr_cat_10x_1000",
    metric = "mean"
)]
struct Cat1D;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "cat_2d",
    title = "Concatenate 10× 256×64 (numr vs ndarray)",
    benchmarks = ["numr_cat_10x_256x64", "ndarray_cat_10x_256x64"],
    baseline = "numr_cat_10x_256x64",
    metric = "mean"
)]
struct Cat2D;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "cat_2d",
    title = "Concatenate 10× 256×64 (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_cat_10x_256x64", "ndarray_cat_10x_256x64", "cuda_cat_10x_256x64"],
    baseline = "numr_cat_10x_256x64",
    metric = "mean"
)]
struct Cat2D;

// ---------------------------------------------------------------------------
// Verifications: numr must be competitive with ndarray
// ---------------------------------------------------------------------------
// 1D cat (~800ns) has high run-to-run variance (~20-40% between runs),
// so the 1.4x threshold accommodates noise while still catching regressions.
// 2D cat is the meaningful performance test with stable measurements.

#[flux::verify(
    expr = "numr_cat_10x_1000 / ndarray_cat_10x_1000 < 1.4",
    severity = "warning"
)]
struct VerifyCat1D;

#[flux::verify(
    expr = "numr_cat_10x_256x64 / ndarray_cat_10x_256x64 < 1.1",
    severity = "warning"
)]
struct VerifyCat2D;

#[flux::synthetic(
    id = "cat_1d_ratio",
    formula = "numr_cat_10x_1000 / ndarray_cat_10x_1000",
    unit = "x"
)]
struct Cat1DRatio;

#[flux::synthetic(
    id = "cat_2d_ratio",
    formula = "numr_cat_10x_256x64 / ndarray_cat_10x_256x64",
    unit = "x"
)]
struct Cat2DRatio;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_cat_speedup",
    formula = "numr_cat_10x_256x64 / cuda_cat_10x_256x64",
    unit = "x"
)]
struct CudaCatSpeedup;

fn main() {
    fluxbench::run().unwrap();
}
