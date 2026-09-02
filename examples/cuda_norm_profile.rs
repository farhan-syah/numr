//! Profiling target for the CUDA RMSNorm and LayerNorm kernels.
//!
//! Both are one block per row. `rms_norm` reduces with a shared-memory tree and
//! `layer_norm` with a warp-shuffle Welford pass, so profiling the two on the
//! same shape separates the cost of the reduction primitive from the cost of
//! the two passes over the row.
//!
//! The bench harness forks per case and cannot be attached to; this is a plain
//! single-process binary.
//!
//! ```text
//! cargo build --release --features cuda,f16 --example cuda_norm_profile
//! ncu --kernel-name regex:norm --launch-count 4 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
//!     ./target/release/examples/cuda_norm_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::ops::NormalizationOps;
    use numr::prelude::*;

    /// Enough launches for a profiler to sample, few enough that a
    /// replay-based collection stays quick.
    const ITERS: usize = 8;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let hidden = 4096usize;
    let input = client.rand(&[4, 512, hidden], DType::F32).unwrap();
    let weight = client.rand(&[hidden], DType::F32).unwrap();
    let bias = client.rand(&[hidden], DType::F32).unwrap();

    for _ in 0..ITERS {
        std::hint::black_box(client.rms_norm(&input, &weight, 1e-5).unwrap());
        std::hint::black_box(client.layer_norm(&input, &weight, &bias, 1e-5).unwrap());
    }
    client.synchronize();
}
