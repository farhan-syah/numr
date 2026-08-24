//! FFT Round-Trip
//!
//! Demonstrates the Fast Fourier Transform APIs in numr:
//! - Complex FFT → inverse FFT (round-trip identity)
//! - Real FFT (rfft) → inverse real FFT (irfft)
//! - Inspecting frequency-domain magnitudes
//!
//! All FFT operations use the Stockham autosort algorithm, giving identical
//! results on CPU, CUDA, and WebGPU backends.
//!
//! Run with:
//! ```sh
//! cargo run --example fft_roundtrip
//! ```

use numr::dtype::complex::Complex64;
use numr::prelude::*;

fn main() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let n = 64; // must be a power of 2

    // -----------------------------------------------------------------------
    // 1. Complex FFT round-trip
    // -----------------------------------------------------------------------
    // Build a complex signal: two pure tones at bin 3 and bin 10.
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let val = (2.0 * std::f32::consts::PI * 3.0 * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 10.0 * t).cos();
            Complex64::new(val, 0.0)
        })
        .collect();
    let input = Tensor::<CpuRuntime>::try_from_slice(&signal, &[n], &device)?;

    // Forward FFT (no normalization on forward).
    let freq = client.fft(&input, FftDirection::Forward, FftNormalization::Backward)?;

    // Print the five largest frequency magnitudes.
    let freq_data: Vec<Complex64> = freq.to_vec();
    let mut magnitudes: Vec<(usize, f32)> = freq_data
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.magnitude()))
        .collect();
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 frequency bins by magnitude:");
    for &(bin, mag) in magnitudes.iter().take(5) {
        println!("  bin {bin:>3}: {mag:.4}");
    }

    // Inverse FFT (Backward normalization divides by N on inverse).
    let recovered = client.fft(&freq, FftDirection::Inverse, FftNormalization::Backward)?;
    let recovered_data: Vec<Complex64> = recovered.to_vec();

    // Verify round-trip: original ≈ recovered.
    let max_err: f32 = signal
        .iter()
        .zip(recovered_data.iter())
        .map(|(a, b)| {
            let dr = a.re - b.re;
            let di = a.im - b.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0f32, f32::max);
    println!("\nComplex FFT round-trip max error: {max_err:.2e}");
    assert!(max_err < 1e-4, "Round-trip error should be small");

    // -----------------------------------------------------------------------
    // 2. Real FFT round-trip (rfft / irfft)
    // -----------------------------------------------------------------------
    // rfft exploits Hermitian symmetry: for N real inputs it outputs N/2+1
    // complex values, saving half the computation and storage.

    let real_signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            (2.0 * std::f32::consts::PI * 5.0 * t).sin()
        })
        .collect();
    let real_input = Tensor::<CpuRuntime>::try_from_slice(&real_signal, &[n], &device)?;

    let real_freq = client.rfft(&real_input, FftNormalization::Backward)?;
    println!(
        "\nrfft: input length = {n}, output length = {} (N/2+1 complex)",
        real_freq.shape()[0],
    );

    // irfft recovers the original real signal.
    let real_recovered = client.irfft(&real_freq, Some(n), FftNormalization::Backward)?;
    let real_recovered_data: Vec<f32> = real_recovered.to_vec();

    let real_max_err: f32 = real_signal
        .iter()
        .zip(real_recovered_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Real FFT round-trip max error: {real_max_err:.2e}");
    assert!(real_max_err < 1e-4, "Real round-trip error should be small");

    println!("\nFFT round-trip example completed successfully!");
    Ok(())
}
