#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_numr(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

fn rand_numr_f64(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F64).unwrap()
}

fn rand_vec_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + 3) % 1000) as f32 / 1000.0)
        .collect()
}

// ---------------------------------------------------------------------------
// numr: single-dim sum (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_single_dim_f32", args = [1_000, 100_000, 1_000_000, 10_000_000])]
fn numr_sum(b: &mut Bencher, n: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[n], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: multi-dim reduce (2D matrix, reduce rows)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_2d_rows_f32", args = [256, 1024])]
fn numr_sum_rows(b: &mut Bencher, size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[size, size], &device);
    b.iter(|| black_box(client.sum(&t, &[1], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: multi-dim reduce (reduce ALL dims)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_all_dims_f32", args = [256, 1024])]
fn numr_sum_all(b: &mut Bencher, size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[size, size], &device);
    b.iter(|| black_box(client.sum(&t, &[0, 1], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: mean and max
// ---------------------------------------------------------------------------

#[flux::bench(group = "mean_f32")]
fn numr_mean_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.mean(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "max_f32")]
fn numr_max_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.max(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: f64 reductions
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_f64")]
fn numr_sum_f64_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr_f64(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
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
#[flux::bench(group = "sum_single_dim_f32", args = [1_000_000, 10_000_000])]
fn cuda_sum(b: &mut Bencher, n: usize) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[n], &device);
    b.iter(|| {
        let r = black_box(client.sum(&t, &[0], false).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "sum_2d_rows_f32")]
fn cuda_sum_rows_1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1024, 1024], &device);
    b.iter(|| {
        let r = black_box(client.sum(&t, &[1], false).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "mean_f32")]
fn cuda_mean_1m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1_000_000], &device);
    b.iter(|| {
        let r = black_box(client.mean(&t, &[0], false).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "max_f32")]
fn cuda_max_1m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1_000_000], &device);
    b.iter(|| {
        let r = black_box(client.max(&t, &[0], false).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// ndarray comparison (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_single_dim_f32", args = [1_000, 100_000, 1_000_000, 10_000_000])]
fn ndarray_sum(b: &mut Bencher, n: usize) {
    let data = rand_vec_f32(n);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.sum()));
}

#[flux::bench(group = "sum_2d_rows_f32", args = [256, 1024])]
fn ndarray_sum_rows(b: &mut Bencher, size: usize) {
    let data = rand_vec_f32(size * size);
    let a = ndarray::Array2::from_shape_vec((size, size), data).unwrap();
    b.iter(|| black_box(a.sum_axis(ndarray::Axis(1))));
}

#[flux::bench(group = "mean_f32")]
fn ndarray_mean_1m(b: &mut Bencher) {
    let data = rand_vec_f32(1_000_000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.mean()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_1m",
    title = "Sum 1M elements (numr vs ndarray)",
    benchmarks = ["numr_sum@1_000_000", "ndarray_sum@1_000_000"],
    baseline = "numr_sum@1_000_000",
    metric = "mean"
)]
struct Sum1M;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_1m",
    title = "Sum 1M elements (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum@1_000_000", "ndarray_sum@1_000_000", "cuda_sum@1_000_000"],
    baseline = "numr_sum@1_000_000",
    metric = "mean"
)]
struct Sum1M;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_10m",
    title = "Sum 10M elements (numr vs ndarray)",
    benchmarks = ["numr_sum@10_000_000", "ndarray_sum@10_000_000"],
    baseline = "numr_sum@10_000_000",
    metric = "mean"
)]
struct Sum10M;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_10m",
    title = "Sum 10M elements (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum@10_000_000", "ndarray_sum@10_000_000", "cuda_sum@10_000_000"],
    baseline = "numr_sum@10_000_000",
    metric = "mean"
)]
struct Sum10M;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_rows_1024",
    title = "Row-sum 1024×1024 (numr vs ndarray)",
    benchmarks = ["numr_sum_rows@1024", "ndarray_sum_rows@1024"],
    baseline = "numr_sum_rows@1024",
    metric = "mean"
)]
struct SumRows1024;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_rows_1024",
    title = "Row-sum 1024×1024 (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum_rows@1024", "ndarray_sum_rows@1024", "cuda_sum_rows_1024x1024"],
    baseline = "numr_sum_rows@1024",
    metric = "mean"
)]
struct SumRows1024;

// ---------------------------------------------------------------------------
// Scaling series
// ---------------------------------------------------------------------------

#[flux::compare(id = "rscale_1k", title = "Reduce Scaling", benchmarks = ["numr_sum@1_000"], group = "reduce_scaling", x = "1000")]
struct RScale1K;

#[flux::compare(id = "rscale_100k", title = "Reduce Scaling", benchmarks = ["numr_sum@100_000"], group = "reduce_scaling", x = "100000")]
struct RScale100K;

#[flux::compare(id = "rscale_1m", title = "Reduce Scaling", benchmarks = ["numr_sum@1_000_000"], group = "reduce_scaling", x = "1000000")]
struct RScale1M;

#[flux::compare(id = "rscale_10m", title = "Reduce Scaling", benchmarks = ["numr_sum@10_000_000"], group = "reduce_scaling", x = "10000000")]
struct RScale10M;

// ---------------------------------------------------------------------------
// Verifications: numr must be >= 90% of ndarray speed (ratio < 1.1)
// ---------------------------------------------------------------------------

#[flux::verify(
    expr = "numr_sum@1_000_000 / ndarray_sum@1_000_000 < 1.1",
    severity = "critical"
)]
struct VerifySum1M;

#[flux::verify(
    expr = "numr_sum@10_000_000 / ndarray_sum@10_000_000 < 1.1",
    severity = "critical"
)]
struct VerifySum10M;

#[flux::verify(
    expr = "numr_sum_rows@1024 / ndarray_sum_rows@1024 < 1.1",
    severity = "warning"
)]
struct VerifyRows1024;

#[flux::synthetic(
    id = "sum_1m_ratio",
    formula = "numr_sum@1_000_000 / ndarray_sum@1_000_000",
    unit = "x"
)]
struct Sum1MRatio;

#[flux::synthetic(
    id = "sum_10m_ratio",
    formula = "numr_sum@10_000_000 / ndarray_sum@10_000_000",
    unit = "x"
)]
struct Sum10MRatio;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_sum_speedup_1m",
    formula = "numr_sum@1_000_000 / cuda_sum@1_000_000",
    unit = "x"
)]
struct CudaSumSpeedup1M;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_sum_speedup_10m",
    formula = "numr_sum@10_000_000 / cuda_sum@10_000_000",
    unit = "x"
)]
struct CudaSumSpeedup10M;

fn main() {
    fluxbench::run().unwrap();
}
