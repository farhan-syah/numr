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

fn rand_indices(n: usize, max_val: i32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let data: Vec<i32> = (0..n).map(|i| (i as i32) % max_val).collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[n], device).unwrap()
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

#[flux::bench(group = "gather_f32")]
fn numr_gather_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 64], &device);
    let idx = rand_indices(500, 1000, &device);
    let idx = idx.reshape(&[500, 1]).unwrap();
    let idx = {
        let client = CpuRuntime::default_client(&device);
        client.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.gather(&t, 0, &idx).unwrap()));
}

#[flux::bench(group = "gather_f32")]
fn numr_gather_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000, 64], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    let idx = idx.reshape(&[10_000, 1]).unwrap();
    let idx = {
        let client = CpuRuntime::default_client(&device);
        client.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.gather(&t, 0, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// index_select
// ---------------------------------------------------------------------------

#[flux::bench(group = "index_select_f32")]
fn numr_index_select_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 128], &device);
    let idx = rand_indices(256, 1000, &device);
    b.iter(|| black_box(client.index_select(&t, 0, &idx).unwrap()));
}

#[flux::bench(group = "index_select_f32")]
fn numr_index_select_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000, 128], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    b.iter(|| black_box(client.index_select(&t, 0, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// take (flat indexing)
// ---------------------------------------------------------------------------

#[flux::bench(group = "take_f32")]
fn numr_take_10k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    b.iter(|| black_box(client.take(&t, &idx).unwrap()));
}

#[flux::bench(group = "take_f32")]
fn numr_take_100k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1_000_000], &device);
    let idx = rand_indices(100_000, 1_000_000, &device);
    b.iter(|| black_box(client.take(&t, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

#[flux::bench(group = "scatter_f32")]
fn numr_scatter_1k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000, 64], &device);
    let src = rand_t(&[500, 64], &device);
    let idx = rand_indices(500, 1000, &device);
    let idx = idx.reshape(&[500, 1]).unwrap();
    let idx = {
        let c = CpuRuntime::default_client(&device);
        c.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| black_box(client.scatter(&t, 0, &idx, &src).unwrap()));
}

// ---------------------------------------------------------------------------
// put (flat scatter)
// ---------------------------------------------------------------------------

#[flux::bench(group = "put_f32")]
fn numr_put_10k(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    let idx = rand_indices(10_000, 100_000, &device);
    let vals = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.put(&t, &idx, &vals).unwrap()));
}

// ---------------------------------------------------------------------------
// embedding_lookup (common ML pattern)
// ---------------------------------------------------------------------------

#[flux::bench(group = "embedding_f32")]
fn numr_embedding_32k_vocab(b: &mut Bencher) {
    let (device, client) = setup();
    let embeddings = rand_t(&[32_000, 128], &device);
    let idx = rand_indices(512, 32_000, &device);
    b.iter(|| black_box(client.embedding_lookup(&embeddings, &idx).unwrap()));
}

#[flux::bench(group = "embedding_f32")]
fn numr_embedding_128k_vocab(b: &mut Bencher) {
    let (device, client) = setup();
    let embeddings = rand_t(&[128_000, 128], &device);
    let idx = rand_indices(512, 128_000, &device);
    b.iter(|| black_box(client.embedding_lookup(&embeddings, &idx).unwrap()));
}

// ---------------------------------------------------------------------------
// CUDA benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_setup() -> (CudaDevice, CudaClient) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (device, client)
}

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

#[cfg(feature = "cuda")]
fn rand_cuda_indices(n: usize, max_val: i32, device: &CudaDevice) -> Tensor<CudaRuntime> {
    let data: Vec<i32> = (0..n).map(|i| (i as i32) % max_val).collect();
    Tensor::<CudaRuntime>::from_slice(&data, &[n], device).unwrap()
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "index_select_f32")]
fn cuda_index_select_100k(b: &mut Bencher) {
    let (device, client) = cuda_setup();
    let t = rand_cuda(&[100_000, 128], &device);
    let idx = rand_cuda_indices(10_000, 100_000, &device);
    b.iter(|| {
        let r = black_box(client.index_select(&t, 0, &idx).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "embedding_f32")]
fn cuda_embedding_32k_vocab(b: &mut Bencher) {
    let (device, client) = cuda_setup();
    let embeddings = rand_cuda(&[32_000, 128], &device);
    let idx = rand_cuda_indices(512, 32_000, &device);
    b.iter(|| {
        let r = black_box(client.embedding_lookup(&embeddings, &idx).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "gather_f32")]
fn cuda_gather_100k(b: &mut Bencher) {
    let (device, client) = cuda_setup();
    let t = rand_cuda(&[100_000, 64], &device);
    let idx = rand_cuda_indices(10_000, 100_000, &device);
    let idx = idx.reshape(&[10_000, 1]).unwrap();
    let idx = {
        let c = CudaRuntime::default_client(&device);
        c.repeat(&idx, &[1, 64]).unwrap()
    };
    b.iter(|| {
        let r = black_box(client.gather(&t, 0, &idx).unwrap());
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "index_select_cmp",
    title = "index_select: 1K vs 100K source rows",
    benchmarks = ["numr_index_select_1k", "numr_index_select_100k"],
    baseline = "numr_index_select_1k",
    metric = "mean"
)]
struct IndexSelectCmp;

#[flux::compare(
    id = "take_cmp",
    title = "take: 10K vs 100K indices",
    benchmarks = ["numr_take_10k", "numr_take_100k"],
    baseline = "numr_take_10k",
    metric = "mean"
)]
struct TakeCmp;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "embedding_cmp",
    title = "Embedding: 32K vs 128K vocab",
    benchmarks = ["numr_embedding_32k_vocab", "numr_embedding_128k_vocab"],
    baseline = "numr_embedding_32k_vocab",
    metric = "mean"
)]
struct EmbeddingCmp;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "embedding_cmp",
    title = "Embedding: CPU vs CUDA (32K vocab)",
    benchmarks = ["numr_embedding_32k_vocab", "numr_embedding_128k_vocab", "cuda_embedding_32k_vocab"],
    baseline = "numr_embedding_32k_vocab",
    metric = "mean"
)]
struct EmbeddingCmp;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_embedding_speedup",
    formula = "numr_embedding_32k_vocab / cuda_embedding_32k_vocab",
    unit = "x"
)]
struct CudaEmbeddingSpeedup;

fn main() {
    fluxbench::run().unwrap();
}
