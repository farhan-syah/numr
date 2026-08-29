#![allow(dead_code)]
// Imports are only used by the CUDA-gated cases below.
#![allow(unused_imports)]

//! Integer CUDA GEMM benchmarks (I32, I64).
//!
//! `matmul.rs` covers float matmul; this file covers the integer path, which
//! previously had no benchmark coverage at all. It started as a comparison
//! between single- and double-buffered shared-memory staging - single won (see
//! `matmul_int.cu`) and the double-buffered kernel was deleted - so only the
//! plain cases remain.

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

#[cfg(feature = "cuda")]
fn randint_cuda(shape: &[usize], dtype: DType, device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    // Small magnitudes keep every product well inside the accumulator; the
    // kernel does the same work regardless of value.
    client.randint(-64, 64, shape, dtype).unwrap()
}

#[cfg(feature = "cuda")]
fn bench_int_matmul(b: &mut Bencher, shape_a: &[usize], shape_b: &[usize], dtype: DType) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = randint_cuda(shape_a, dtype, &device);
    let bm = randint_cuda(shape_b, dtype, &device);
    b.iter(|| {
        let r = black_box(client.matmul(&a, &bm).unwrap());
        // Sync so the measured time is the kernel, not the launch.
        client.synchronize();
        r
    });
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i32_1024(b: &mut Bencher) {
    bench_int_matmul(b, &[1024, 1024], &[1024, 1024], DType::I32);
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i64_1024(b: &mut Bencher) {
    bench_int_matmul(b, &[1024, 1024], &[1024, 1024], DType::I64);
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i32_256(b: &mut Bencher) {
    bench_int_matmul(b, &[256, 256], &[256, 256], DType::I32);
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i64_256(b: &mut Bencher) {
    bench_int_matmul(b, &[256, 256], &[256, 256], DType::I64);
}

/// Batched: [8, 512, 512] @ [8, 512, 512]
#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i32_batch8_512(b: &mut Bencher) {
    bench_int_matmul(b, &[8, 512, 512], &[8, 512, 512], DType::I32);
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_int")]
fn cuda_i64_batch8_512(b: &mut Bencher) {
    bench_int_matmul(b, &[8, 512, 512], &[8, 512, 512], DType::I64);
}

fn main() {
    fluxbench::run().unwrap();
}
