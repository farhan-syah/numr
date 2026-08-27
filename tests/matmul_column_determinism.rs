//! CPU matmul output must not depend on the size of the thread pool.
//!
//! The tiled CPU matmul splits its output columns across rayon. A column split
//! moves the N-block boundaries the kernel sees, so the chunk boundaries decide
//! which microkernel variant produces an element near a boundary and therefore
//! how its float accumulation rounds. When those boundaries were derived from
//! the thread count, the same VoxCPM2 sentence decoded to different audio on a
//! 1-thread run and a 24-thread run — a model shipped to unknown hardware
//! cannot behave that way.
//!
//! These tests pin the fix: the chunk boundaries are a pure function of the
//! shape, so every pool size produces bit-identical bytes. A client built with
//! `ParallelismConfig` owns a rayon pool of exactly that size, which is how the
//! shipping API reaches the pool sizes a user's machine would give it.

use numr::ops::MatmulOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

/// Pool sizes a user's machine plausibly hands the client.
const POOL_SIZES: [usize; 3] = [1, 2, 8];

/// The shape that motivated the column split: the local DiT decode step.
const DIT_M: usize = 22;
const DIT_K: usize = 1024;

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.004 - 0.5)
        .collect()
}

fn client_with_threads(device: &CpuDevice, threads: usize) -> CpuClient {
    CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(threads), None))
}

/// Raw bits, so the comparison is exact rather than tolerant.
fn bits(out: &[f32]) -> Vec<u32> {
    out.iter().map(|v| v.to_bits()).collect()
}

/// `A[m, k] @ B[k, n]` with a contiguous B, on a pool of `threads` threads.
fn contiguous_on_pool(m: usize, n: usize, k: usize, threads: usize) -> Vec<u32> {
    let device = CpuDevice::new();
    let client = client_with_threads(&device, threads);

    let a = Tensor::<CpuRuntime>::from_slice(&values(m * k, 1), &[m, k], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&values(k * n, 2), &[k, n], &device).unwrap();

    bits(&client.matmul(&a, &b).unwrap().to_vec::<f32>())
}

/// `A[m, k] @ W[n, k]^T`, the transposed-B path, on a pool of `threads` threads.
fn transposed_on_pool(m: usize, n: usize, k: usize, threads: usize) -> Vec<u32> {
    let device = CpuDevice::new();
    let client = client_with_threads(&device, threads);

    let a = Tensor::<CpuRuntime>::from_slice(&values(m * k, 1), &[m, k], &device).unwrap();
    let w = Tensor::<CpuRuntime>::from_slice(&values(n * k, 2), &[n, k], &device).unwrap();
    let b_view = w.transpose(0, 1).unwrap();

    bits(&client.matmul(&a, &b_view).unwrap().to_vec::<f32>())
}

/// Every pool size must reproduce the one-thread bytes exactly.
fn assert_pool_invariant(run: impl Fn(usize) -> Vec<u32>, label: &str) {
    let reference = run(POOL_SIZES[0]);
    for threads in &POOL_SIZES[1..] {
        let got = run(*threads);
        assert_eq!(got.len(), reference.len(), "{label}: length at {threads}");
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(
                g,
                r,
                "{label}: element {i} differs at {threads} threads ({} vs {})",
                f32::from_bits(*g),
                f32::from_bits(*r)
            );
        }
    }
}

/// The hot decode shape, whose `n` is an exact multiple of the chunk width.
#[test]
fn contiguous_dit_shape_is_pool_invariant() {
    assert_pool_invariant(
        |threads| contiguous_on_pool(DIT_M, 4096, DIT_K, threads),
        "contiguous 22x1024x4096",
    );
}

/// Same shape through the transposed-B path, which packs B out of an `[N, K]`
/// buffer instead of materializing the view.
#[test]
fn transposed_dit_shape_is_pool_invariant() {
    assert_pool_invariant(
        |threads| transposed_on_pool(DIT_M, 4096, DIT_K, threads),
        "transposed 22x1024x4096",
    );
}

/// A prime `n` is a multiple of neither the chunk width nor the register block,
/// so the balanced chunks are uneven and both remainder paths run.
#[test]
fn prime_n_is_pool_invariant() {
    assert_pool_invariant(
        |threads| contiguous_on_pool(DIT_M, 1021, DIT_K, threads),
        "contiguous n=1021",
    );
    assert_pool_invariant(
        |threads| transposed_on_pool(DIT_M, 1021, DIT_K, threads),
        "transposed n=1021",
    );
}
