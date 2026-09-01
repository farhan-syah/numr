//! Profiling target for the two CUDA convolution shapes that sit furthest above
//! their memory-bandwidth floor: depthwise conv2d and long depthwise conv1d.
//!
//! Benchmarks report wall-clock only, which cannot separate a bandwidth limit
//! from a latency or occupancy limit. A profiler can, but it needs a plain
//! single-process binary to attach to — the bench harness isolates each case in
//! a child process. This example is that binary.
//!
//! Run under a profiler, filtering to the kernel of interest:
//!
//! ```text
//! cargo build --release --features cuda,f16 --example cuda_conv_profile
//! ncu --kernel-name regex:depthwise --launch-count 4 \
//!     --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
//!     ./target/release/examples/cuda_conv_profile
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("this example needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use numr::ops::{ConvOps, PaddingMode};
    use numr::prelude::*;

    /// Enough iterations for a profiler to collect several launches, few enough
    /// that a replay-based collection still finishes quickly.
    const ITERS: usize = 8;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    // depthwise conv2d, 96 channels, 3x3, 112x112, batch 2. Column-blocked
    // kernel; the largest shape in the bench sweep and the one furthest above
    // its bandwidth floor.
    let dw2d_input = client.rand(&[2, 96, 112, 112], DType::F32).unwrap();
    let dw2d_weight = client.rand(&[96, 1, 3, 3], DType::F32).unwrap();
    let dw2d_bias = client.rand(&[96], DType::F32).unwrap();

    // depthwise conv1d, 1536 channels, k=4, L=1024, causal padding. Runs the
    // position-blocked conv1d_ox kernel.
    let dw1d_input = client.rand(&[1, 1536, 1024], DType::F32).unwrap();
    let dw1d_weight = client.rand(&[1536, 1, 4], DType::F32).unwrap();
    let dw1d_bias = client.rand(&[1536], DType::F32).unwrap();

    for _ in 0..ITERS {
        let out = client
            .depthwise_conv2d(
                &dw2d_input,
                &dw2d_weight,
                Some(&dw2d_bias),
                (1, 1),
                PaddingMode::Valid,
                (1, 1),
            )
            .unwrap();
        std::hint::black_box(&out);

        let out = client
            .conv1d(
                &dw1d_input,
                &dw1d_weight,
                Some(&dw1d_bias),
                1,
                PaddingMode::Custom(3, 0, 0, 0),
                1,
                1536,
            )
            .unwrap();
        std::hint::black_box(&out);
    }
    client.synchronize();
}
