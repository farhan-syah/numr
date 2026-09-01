//! Profiling target for the CUDA strided-copy kernel, reached via
//! `Tensor::contiguous` on a permuted view.
//!
//! The benchmark for this path times `contiguous()`, which allocates its
//! destination on every iteration, so the wall-clock figure mixes allocation
//! with kernel time. A profiler attributes time to the kernel alone. The bench
//! harness forks per case and cannot be attached to; this is a plain
//! single-process binary.
//!
//! ```text
//! cargo build --release --features cuda,f16 --example cuda_strided_copy_profile
//! ncu --kernel-name regex:strided --launch-count 4 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis \
//!     ./target/release/examples/cuda_strided_copy_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::prelude::*;

    /// Enough launches for a profiler to sample, few enough that a
    /// replay-based collection stays quick.
    const ITERS: usize = 8;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    // Same element count, different dimension counts. `cube_view`'s trailing
    // axes merge into one run (see `collapse_axes`), so it takes the tiled
    // strided_transpose path; `hyper_view` below is a full axis reversal,
    // which does not merge, so it falls back to strided_copy's per-dimension
    // index decode. The pair isolates that difference.
    let cube = client.rand(&[256, 256, 256], DType::F32).unwrap();
    let cube_view = cube.permute(&[2, 0, 1]).unwrap();

    let hyper = client.rand(&[64, 64, 64, 64], DType::F32).unwrap();
    let hyper_view = hyper.permute(&[3, 2, 1, 0]).unwrap();

    // Contiguous inner runs, so this one should already coalesce well.
    let wide = client.rand(&[8192, 8192], DType::F32).unwrap();
    let wide_view = wide.narrow(1, 0, 4096).unwrap();

    for _ in 0..ITERS {
        std::hint::black_box(cube_view.contiguous().unwrap());
        std::hint::black_box(hyper_view.contiguous().unwrap());
        std::hint::black_box(wide_view.contiguous().unwrap());
    }
    client.synchronize();
}
