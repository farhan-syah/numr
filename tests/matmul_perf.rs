//! Matmul efficiency microbenchmark (CUDA). Measures achieved GFLOP/s for F32
//! and F16 across square and embed-shaped GEMMs to quantify the F32-vs-WMMA gap.
//!
//! Run: cargo test --features cuda,f16 --test matmul_perf -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::time::Instant;

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::ops::TypeConversionOps;
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn bench(
    client: &<CudaRuntime as Runtime>::Client,
    device: &CudaDevice,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    label: &str,
) {
    let a32 = Tensor::<CudaRuntime>::try_from_slice(
        &(0..m * k)
            .map(|i| ((i % 97) as f32) * 0.01)
            .collect::<Vec<_>>(),
        &[m, k],
        device,
    )
    .unwrap();
    let b32 = Tensor::<CudaRuntime>::try_from_slice(
        &(0..k * n)
            .map(|i| ((i % 89) as f32) * 0.01)
            .collect::<Vec<_>>(),
        &[k, n],
        device,
    )
    .unwrap();
    let (a, b) = if dtype == DType::F32 {
        (a32, b32)
    } else {
        (
            client.cast(&a32, dtype).unwrap(),
            client.cast(&b32, dtype).unwrap(),
        )
    };

    // Warm up (kernel JIT + caches).
    for _ in 0..3 {
        let _ = client.matmul(&a, &b).unwrap();
    }
    client.synchronize();

    const ITERS: usize = 20;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let c = client.matmul(&a, &b).unwrap();
        std::hint::black_box(&c);
    }
    client.synchronize();
    let secs = t0.elapsed().as_secs_f64() / ITERS as f64;
    let gflop = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;
    println!(
        "  {label:<22} {m}x{k}x{n} {:?}: {:.2} ms => {:.0} GFLOP/s",
        dtype,
        secs * 1e3,
        gflop / secs
    );
}

#[test]
#[ignore = "perf microbenchmark — run with: cargo test --features cuda,f16 --test matmul_perf -- --ignored --nocapture"]
fn matmul_efficiency() {
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!("no CUDA");
        return;
    }
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    println!("RTX 3060 FP32 peak ~13 TFLOP/s, F16 tensor-core peak ~50-100 TFLOP/s");
    for &(m, k, n, tag) in &[
        (2048, 2048, 2048, "square-2048"),
        (4096, 4096, 4096, "square-4096"),
        (8192, 768, 3072, "embed-ffn-up"),
        (8190, 768, 768, "embed-proj-M-UNALIGNED"),
        (8192, 768, 768, "embed-proj"),
        (8192, 3072, 768, "embed-ffn-down"),
    ] {
        bench(&client, &device, m, k, n, DType::F32, tag);
        #[cfg(feature = "f16")]
        bench(&client, &device, m, k, n, DType::F16, tag);
    }
}
