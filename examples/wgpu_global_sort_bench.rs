//! WGPU global-sort wall-time diagnostic.
//!
//! Run with:
//! `cargo run --release --example wgpu_global_sort_bench --features wgpu`

#[cfg(feature = "wgpu")]
fn main() -> numr::error::Result<()> {
    use numr::ops::SortingOps;
    use numr::runtime::Runtime;
    use numr::runtime::RuntimeClient;
    use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
    use numr::tensor::Tensor;
    use std::time::Instant;

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let sizes = [513usize, 4_097, 65_537, 1_000_003];

    println!("boundary=public sort call + queue completion; input upload/output readback excluded");
    for size in sizes {
        let data: Vec<u32> = (0..size as u32)
            .map(|index| index.wrapping_mul(747_796_405).wrapping_add(2_891_336_453))
            .collect();
        let input = Tensor::from_slice(&data, &[size], &device);

        let validation: Vec<u32> = client.sort(&input, 0, false)?.to_vec();
        assert!(validation.windows(2).all(|pair| pair[0] <= pair[1]));

        for _ in 0..3 {
            let output = client.sort(&input, 0, false)?;
            client.synchronize();
            std::hint::black_box(output);
        }

        let sample_count = if size >= 1_000_000 {
            7
        } else if size >= 65_000 {
            11
        } else {
            21
        };
        let mut samples_ms = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let start = Instant::now();
            let output = client.sort(&input, 0, false)?;
            client.synchronize();
            samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            std::hint::black_box(output);
        }
        samples_ms.sort_by(f64::total_cmp);
        println!(
            "size={size} samples={sample_count} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
            samples_ms[sample_count / 2],
            samples_ms[0],
            samples_ms[sample_count - 1]
        );
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("enable the `wgpu` feature");
}
