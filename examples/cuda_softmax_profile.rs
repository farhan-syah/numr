//! Profiling target for the last-dimension CUDA softmax kernel.
//!
//! That kernel makes three passes over the row: find the max, write `exp`
//! values into the output buffer while summing them, then read those back to
//! normalize. Whether removing the output round-trip pays depends on the kernel
//! actually being bandwidth-bound, which only a profiler settles.
//!
//! The bench harness forks per case and cannot be attached to; this is a plain
//! single-process binary.
//!
//! ```text
//! cargo build --release --features cuda,f16 --example cuda_softmax_profile
//! ncu --kernel-name regex:softmax --launch-count 4 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
//!     ./target/release/examples/cuda_softmax_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::ops::ActivationOps;
    use numr::prelude::*;

    /// Enough launches for a profiler to sample, few enough that a
    /// replay-based collection stays quick.
    const ITERS: usize = 8;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    // Hidden-size row: 4096 floats is 16 KB, which would fit in shared memory.
    let hidden = client.rand(&[4, 512, 4096], DType::F32).unwrap();

    // Vocab-size row: 32000 floats is 125 KB, far past any shared-memory
    // budget, so it must stay a multi-pass kernel whatever the smaller case does.
    let vocab = client.rand(&[1, 512, 32000], DType::F32).unwrap();

    for _ in 0..ITERS {
        std::hint::black_box(client.softmax(&hidden, -1).unwrap());
        std::hint::black_box(client.softmax(&vocab, -1).unwrap());
    }
    client.synchronize();
}
