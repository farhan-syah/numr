//! CI Regression Benchmarks
//!
//! Focused benchmark suite for regression detection on PRs. Cherry-picks the
//! most critical operations from the full benchmark suite to keep CI fast
//! while covering the hot paths.
//!
//! Usage:
//!   # Run benchmarks:
//!   cargo bench --bench ci_regression
//!
//!   # Save baseline (on main):
//!   cargo bench --bench ci_regression -- --save-baseline
//!
//!   # Compare against baseline (on PR):
//!   cargo bench --bench ci_regression -- --baseline target/fluxbench/baseline.json
//!
//!   # GitHub Actions summary output:
//!   cargo bench --bench ci_regression -- --format github-summary --baseline target/fluxbench/baseline.json

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_f32(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

fn rand_complex(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    let real = client.rand(&[n], DType::F64).unwrap();
    client.cast(&real, DType::Complex128).unwrap()
}

fn rand_indices(n: usize, max_val: i32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let data: Vec<i32> = (0..n).map(|i| (i as i32) % max_val).collect();
    Tensor::<CpuRuntime>::try_from_slice(&data, &[n], device).unwrap()
}

// ---------------------------------------------------------------------------
// Matmul — core of all ML workloads
// ---------------------------------------------------------------------------

#[flux::bench(
    id = "matmul_512",
    group = "matmul",
    severity = "critical",
    threshold = 5.0
)]
fn matmul_512(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_f32(&[512, 512], &device);
    let bm = rand_f32(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(
    id = "matmul_1024",
    group = "matmul",
    severity = "critical",
    threshold = 5.0
)]
fn matmul_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_f32(&[1024, 1024], &device);
    let bm = rand_f32(&[1024, 1024], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// Reduce — used in every loss/norm computation
// ---------------------------------------------------------------------------

#[flux::bench(
    id = "reduce_sum_1m",
    group = "reduce",
    severity = "critical",
    threshold = 5.0
)]
fn reduce_sum_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_f32(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(
    id = "reduce_sum_10m",
    group = "reduce",
    severity = "warning",
    threshold = 20.0,
    samples = 15
)]
fn reduce_sum_10m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_f32(&[10_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// FFT — complex algorithm, easy to regress
// ---------------------------------------------------------------------------

#[flux::bench(id = "fft_1024", group = "fft", severity = "critical", threshold = 5.0)]
fn fft_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(1024, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(
    id = "fft_16384",
    group = "fft",
    severity = "warning",
    threshold = 10.0
)]
fn fft_16384(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(16384, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// Embedding lookup — every forward pass in LLMs
// ---------------------------------------------------------------------------

#[flux::bench(
    id = "embedding_32k",
    group = "embedding",
    severity = "warning",
    threshold = 20.0,
    samples = 20
)]
fn embedding_32k(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let embeddings = rand_f32(&[32_000, 128], &device);
    let idx = rand_indices(512, 32_000, &device);
    b.iter(|| black_box(client.embedding_lookup(&embeddings, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// Concatenation — shape ops used everywhere
// ---------------------------------------------------------------------------

#[flux::bench(
    id = "cat_10x_256x64",
    group = "shape",
    severity = "warning",
    threshold = 25.0,
    samples = 20
)]
fn cat_10x_256x64(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let tensors: Vec<_> = (0..10).map(|_| rand_f32(&[256, 64], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.cat(&refs, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// Regression gates
// ---------------------------------------------------------------------------

#[flux::verify(expr = "matmul_512 < 50000000", severity = "critical")]
#[allow(dead_code)]
struct Matmul512Budget; // 50ms absolute ceiling

#[flux::verify(expr = "matmul_1024 < 500000000", severity = "critical")]
#[allow(dead_code)]
struct Matmul1024Budget; // 500ms absolute ceiling

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
