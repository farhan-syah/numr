use super::bluestein::BluesteinPlan;
use super::stockham::{stockham_fft_c64, stockham_fft_c128};
use super::*;
use std::f64::consts::PI;

#[test]
fn test_fft_impulse() {
    // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
    let input = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let mut output = [Complex64::default(); 4];

    unsafe {
        stockham_fft_c64(&input, &mut output, false, 1.0);
    }

    for c in &output {
        assert!((c.re - 1.0).abs() < 1e-5, "Expected 1.0, got {}", c.re);
        assert!(c.im.abs() < 1e-5, "Expected 0.0i, got {}i", c.im);
    }
}

#[test]
fn test_fft_ifft_roundtrip() {
    // FFT followed by IFFT should recover original signal
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let mut fft_output = [Complex64::default(); 4];
    let mut ifft_output = [Complex64::default(); 4];

    unsafe {
        // Forward FFT (no normalization)
        stockham_fft_c64(&input, &mut fft_output, false, 1.0);
        // Inverse FFT (normalize by 1/N = 0.25)
        stockham_fft_c64(&fft_output, &mut ifft_output, true, 0.25);
    }

    for i in 0..4 {
        assert!(
            (ifft_output[i].re - input[i].re).abs() < 1e-5,
            "Real mismatch at {}: {} vs {}",
            i,
            ifft_output[i].re,
            input[i].re
        );
        assert!(
            (ifft_output[i].im - input[i].im).abs() < 1e-5,
            "Imag mismatch at {}: {} vs {}",
            i,
            ifft_output[i].im,
            input[i].im
        );
    }
}

#[test]
fn test_fft_parseval() {
    // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
    let input = [
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, 1.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(1.5, 0.0),
    ];
    let mut output = [Complex64::default(); 4];

    unsafe {
        stockham_fft_c64(&input, &mut output, false, 1.0);
    }

    let energy_time: f32 = input.iter().map(|c| c.re * c.re + c.im * c.im).sum();
    let energy_freq: f32 = output.iter().map(|c| c.re * c.re + c.im * c.im).sum();

    // energy_time = (1/N) * energy_freq
    let expected_freq_energy = energy_time * 4.0;
    assert!(
        (energy_freq - expected_freq_energy).abs() < 1e-4,
        "Parseval failed: {} vs {}",
        energy_freq,
        expected_freq_energy
    );
}

#[test]
fn test_fft_size_2() {
    // Simple N=2 case
    let input = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let mut output = [Complex64::default(); 2];

    unsafe {
        stockham_fft_c64(&input, &mut output, false, 1.0);
    }

    // X[0] = x[0] + x[1] = 3
    // X[1] = x[0] - x[1] = -1
    assert!((output[0].re - 3.0).abs() < 1e-5);
    assert!(output[0].im.abs() < 1e-5);
    assert!((output[1].re - (-1.0)).abs() < 1e-5);
    assert!(output[1].im.abs() < 1e-5);
}

#[test]
fn test_fft_c128() {
    // Test f64 precision FFT
    let input = [
        Complex128::new(1.0, 0.0),
        Complex128::new(0.0, 0.0),
        Complex128::new(0.0, 0.0),
        Complex128::new(0.0, 0.0),
    ];
    let mut output = [Complex128::default(); 4];

    unsafe {
        stockham_fft_c128(&input, &mut output, false, 1.0);
    }

    for c in &output {
        assert!((c.re - 1.0).abs() < 1e-10);
        assert!(c.im.abs() < 1e-10);
    }
}

#[test]
fn test_rfft() {
    // Real FFT of [1, 2, 3, 4]
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [Complex64::default(); 3]; // N/2 + 1

    unsafe {
        rfft_c64(&input, &mut output, 1.0);
    }

    // Expected (from numpy.fft.rfft):
    // [10+0j, -2+2j, -2+0j]
    assert!((output[0].re - 10.0).abs() < 1e-4);
    assert!(output[0].im.abs() < 1e-4);
    assert!((output[1].re - (-2.0)).abs() < 1e-4);
    assert!((output[1].im - 2.0).abs() < 1e-4);
    assert!((output[2].re - (-2.0)).abs() < 1e-4);
    assert!(output[2].im.abs() < 1e-4);
}

#[test]
fn test_irfft_roundtrip() {
    let original = [1.0f32, 2.0, 3.0, 4.0];
    let mut rfft_out = [Complex64::default(); 3];
    let mut recovered = [0.0f32; 4];

    unsafe {
        rfft_c64(&original, &mut rfft_out, 1.0);
        irfft_c64(&rfft_out, &mut recovered, 0.25); // normalize by 1/N
    }

    for i in 0..4 {
        assert!(
            (recovered[i] - original[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            recovered[i],
            original[i]
        );
    }
}

#[test]
fn test_fftshift() {
    let input = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];
    let mut output = [Complex64::default(); 4];

    unsafe {
        fftshift_c64(&input, &mut output);
    }

    // [0, 1, 2, 3] -> [2, 3, 0, 1]
    assert!((output[0].re - 2.0).abs() < 1e-5);
    assert!((output[1].re - 3.0).abs() < 1e-5);
    assert!((output[2].re - 0.0).abs() < 1e-5);
    assert!((output[3].re - 1.0).abs() < 1e-5);
}

#[test]
fn test_fftshift_ifftshift_roundtrip() {
    let original = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let mut shifted = [Complex64::default(); 4];
    let mut unshifted = [Complex64::default(); 4];

    unsafe {
        fftshift_c64(&original, &mut shifted);
        ifftshift_c64(&shifted, &mut unshifted);
    }

    for i in 0..4 {
        assert!((unshifted[i].re - original[i].re).abs() < 1e-5);
        assert!((unshifted[i].im - original[i].im).abs() < 1e-5);
    }
}

// ========================================================================
// Arbitrary-size (Bluestein) tests
// ========================================================================

/// Sizes exercised by every arbitrary-size test. 400 is Whisper's `n_fft`.
const ARBITRARY_SIZES: [usize; 7] = [3, 5, 7, 12, 100, 400, 1000];

/// Deterministic xorshift64 sample generator in [-1, 1)
fn deterministic_samples(n: usize, seed: u64) -> Vec<Complex128> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..n).map(|_| Complex128::new(next(), next())).collect()
}

/// Naive O(n^2) DFT reference, evaluated in f64.
///
/// The angle uses `(j * k) % n` so the phase stays exact for large indices.
fn naive_dft(input: &[Complex128], inverse: bool) -> Vec<Complex128> {
    let n = input.len();
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    (0..n)
        .map(|k| {
            let mut acc = Complex128::new(0.0, 0.0);
            for (j, x) in input.iter().enumerate() {
                let theta = sign * 2.0 * PI * ((j * k) % n) as f64 / (n as f64);
                acc += *x * Complex128::new(theta.cos(), theta.sin());
            }
            acc
        })
        .collect()
}

/// Magnitude scale of a signal, used to size tolerances.
fn signal_scale(input: &[Complex128]) -> f64 {
    input.iter().map(|c| c.re.abs() + c.im.abs()).sum()
}

fn assert_close_c128(got: &[Complex128], want: &[Complex128], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", label);
    for i in 0..got.len() {
        let dr = (got[i].re - want[i].re).abs();
        let di = (got[i].im - want[i].im).abs();
        assert!(
            dr <= tol && di <= tol,
            "{}: bin {} got ({}, {}), want ({}, {}), tol {}",
            label,
            i,
            got[i].re,
            got[i].im,
            want[i].re,
            want[i].im,
            tol
        );
    }
}

fn assert_close_c64(got: &[Complex64], want: &[Complex128], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", label);
    for i in 0..got.len() {
        let dr = (got[i].re as f64 - want[i].re).abs();
        let di = (got[i].im as f64 - want[i].im).abs();
        assert!(
            dr <= tol && di <= tol,
            "{}: bin {} got ({}, {}), want ({}, {}), tol {}",
            label,
            i,
            got[i].re,
            got[i].im,
            want[i].re,
            want[i].im,
            tol
        );
    }
}

#[test]
fn test_bluestein_forward_matches_naive_dft_c128() {
    for &n in &ARBITRARY_SIZES {
        let input = deterministic_samples(n, 0x5eed_1234 ^ n as u64);
        let expected = naive_dft(&input, false);
        let tol = 1e-11 * signal_scale(&input) + 1e-11;

        let mut got = vec![Complex128::default(); n];
        unsafe {
            fft_c128(&input, &mut got, false, 1.0);
        }

        assert_close_c128(&got, &expected, tol, &format!("forward c128 n={}", n));
    }
}

#[test]
fn test_bluestein_inverse_matches_naive_dft_c128() {
    for &n in &ARBITRARY_SIZES {
        let input = deterministic_samples(n, 0xbeef_0000 ^ n as u64);
        let expected = naive_dft(&input, true);
        let tol = 1e-11 * signal_scale(&input) + 1e-11;

        // Unnormalized inverse, so it matches the reference sum directly.
        let mut got = vec![Complex128::default(); n];
        unsafe {
            fft_c128(&input, &mut got, true, 1.0);
        }

        assert_close_c128(&got, &expected, tol, &format!("inverse c128 n={}", n));
    }
}

#[test]
fn test_bluestein_forward_matches_naive_dft_c64() {
    for &n in &ARBITRARY_SIZES {
        let wide = deterministic_samples(n, 0x1357_9bdf ^ n as u64);
        // Narrow to f32 first, then use the exact same samples for the reference
        // so the comparison isolates the transform, not the input rounding.
        let narrow: Vec<Complex64> = wide
            .iter()
            .map(|c| Complex64::new(c.re as f32, c.im as f32))
            .collect();
        let reference_input: Vec<Complex128> = narrow
            .iter()
            .map(|c| Complex128::new(c.re as f64, c.im as f64))
            .collect();
        let expected = naive_dft(&reference_input, false);
        let tol = 1e-6 * signal_scale(&reference_input) + 1e-5;

        let mut got = vec![Complex64::default(); n];
        unsafe {
            fft_c64(&narrow, &mut got, false, 1.0);
        }

        assert_close_c64(&got, &expected, tol, &format!("forward c64 n={}", n));
    }
}

#[test]
fn test_bluestein_inverse_matches_naive_dft_c64() {
    for &n in &ARBITRARY_SIZES {
        let wide = deterministic_samples(n, 0x2468_ace0 ^ n as u64);
        let narrow: Vec<Complex64> = wide
            .iter()
            .map(|c| Complex64::new(c.re as f32, c.im as f32))
            .collect();
        let reference_input: Vec<Complex128> = narrow
            .iter()
            .map(|c| Complex128::new(c.re as f64, c.im as f64))
            .collect();
        let expected = naive_dft(&reference_input, true);
        let tol = 1e-6 * signal_scale(&reference_input) + 1e-5;

        let mut got = vec![Complex64::default(); n];
        unsafe {
            fft_c64(&narrow, &mut got, true, 1.0);
        }

        assert_close_c64(&got, &expected, tol, &format!("inverse c64 n={}", n));
    }
}

#[test]
fn test_power_of_two_dispatch_is_bit_identical_to_stockham() {
    // The size dispatcher must not perturb the existing power-of-two path.
    for &n in &[1usize, 2, 4, 8, 16, 64, 512] {
        let wide = deterministic_samples(n, 0x0f0f_0f0f ^ n as u64);
        let narrow: Vec<Complex64> = wide
            .iter()
            .map(|c| Complex64::new(c.re as f32, c.im as f32))
            .collect();

        for &inverse in &[false, true] {
            let mut via_dispatch = vec![Complex64::default(); n];
            let mut via_stockham = vec![Complex64::default(); n];
            unsafe {
                fft_c64(&narrow, &mut via_dispatch, inverse, 0.5);
                stockham_fft_c64(&narrow, &mut via_stockham, inverse, 0.5);
            }
            for i in 0..n {
                assert_eq!(
                    via_dispatch[i].re.to_bits(),
                    via_stockham[i].re.to_bits(),
                    "c64 re bits differ at n={} inverse={} bin {}",
                    n,
                    inverse,
                    i
                );
                assert_eq!(
                    via_dispatch[i].im.to_bits(),
                    via_stockham[i].im.to_bits(),
                    "c64 im bits differ at n={} inverse={} bin {}",
                    n,
                    inverse,
                    i
                );
            }

            let mut wide_dispatch = vec![Complex128::default(); n];
            let mut wide_stockham = vec![Complex128::default(); n];
            unsafe {
                fft_c128(&wide, &mut wide_dispatch, inverse, 0.5);
                stockham_fft_c128(&wide, &mut wide_stockham, inverse, 0.5);
            }
            for i in 0..n {
                assert_eq!(
                    wide_dispatch[i].re.to_bits(),
                    wide_stockham[i].re.to_bits(),
                    "c128 re bits differ at n={} inverse={} bin {}",
                    n,
                    inverse,
                    i
                );
                assert_eq!(
                    wide_dispatch[i].im.to_bits(),
                    wide_stockham[i].im.to_bits(),
                    "c128 im bits differ at n={} inverse={} bin {}",
                    n,
                    inverse,
                    i
                );
            }
        }
    }
}

#[test]
fn test_power_of_two_pinned_values() {
    // FFT of the ramp x[j] = j + 1 at N = 8.
    // Closed form: X[0] = 36, X[k] = -4 + 4i*cot(pi*k/8).
    let input: Vec<Complex64> = (0..8)
        .map(|j| Complex64::new(j as f32 + 1.0, 0.0))
        .collect();
    let mut output = vec![Complex64::default(); 8];
    unsafe {
        fft_c64(&input, &mut output, false, 1.0);
    }

    let expected: [(f32, f32); 8] = [
        (36.0, 0.0),
        (-4.0, 9.656_854),
        (-4.0, 4.0),
        (-4.0, 1.656_854_2),
        (-4.0, 0.0),
        (-4.0, -1.656_854_2),
        (-4.0, -4.0),
        (-4.0, -9.656_854),
    ];

    for (i, &(re, im)) in expected.iter().enumerate() {
        assert!(
            (output[i].re - re).abs() < 1e-4,
            "bin {} re: got {}, want {}",
            i,
            output[i].re,
            re
        );
        assert!(
            (output[i].im - im).abs() < 1e-4,
            "bin {} im: got {}, want {}",
            i,
            output[i].im,
            im
        );
    }
}

#[test]
fn test_fft_ifft_roundtrip_400_c128() {
    let n = 400;
    let input = deterministic_samples(n, 0xdead_beef);
    let mut spectrum = vec![Complex128::default(); n];
    let mut recovered = vec![Complex128::default(); n];

    unsafe {
        fft_c128(&input, &mut spectrum, false, 1.0);
        fft_c128(&spectrum, &mut recovered, true, 1.0 / n as f64);
    }

    assert_close_c128(&recovered, &input, 1e-12, "fft/ifft roundtrip n=400 c128");
}

#[test]
fn test_fft_ifft_roundtrip_400_c64() {
    let n = 400;
    let wide = deterministic_samples(n, 0xfeed_face);
    let input: Vec<Complex64> = wide
        .iter()
        .map(|c| Complex64::new(c.re as f32, c.im as f32))
        .collect();
    let expected: Vec<Complex128> = input
        .iter()
        .map(|c| Complex128::new(c.re as f64, c.im as f64))
        .collect();

    let mut spectrum = vec![Complex64::default(); n];
    let mut recovered = vec![Complex64::default(); n];

    unsafe {
        fft_c64(&input, &mut spectrum, false, 1.0);
        fft_c64(&spectrum, &mut recovered, true, 1.0 / n as f32);
    }

    assert_close_c64(&recovered, &expected, 1e-4, "fft/ifft roundtrip n=400 c64");
}

#[test]
fn test_rfft_arbitrary_size_matches_naive_dft() {
    for &n in &ARBITRARY_SIZES {
        let real_f64: Vec<f64> = deterministic_samples(n, 0xabcd_0001 ^ n as u64)
            .iter()
            .map(|c| c.re)
            .collect();
        let as_complex: Vec<Complex128> =
            real_f64.iter().map(|&x| Complex128::new(x, 0.0)).collect();
        let full = naive_dft(&as_complex, false);
        let expected = &full[..n / 2 + 1];
        let scale = signal_scale(&as_complex);

        let mut out_c128 = vec![Complex128::default(); n / 2 + 1];
        unsafe {
            rfft_c128(&real_f64, &mut out_c128, 1.0);
        }
        assert_close_c128(
            &out_c128,
            expected,
            1e-11 * scale + 1e-11,
            &format!("rfft c128 n={}", n),
        );

        let real_f32: Vec<f32> = real_f64.iter().map(|&x| x as f32).collect();
        let as_complex_f32: Vec<Complex128> = real_f32
            .iter()
            .map(|&x| Complex128::new(x as f64, 0.0))
            .collect();
        let full_f32 = naive_dft(&as_complex_f32, false);
        let expected_f32 = &full_f32[..n / 2 + 1];

        let mut out_c64 = vec![Complex64::default(); n / 2 + 1];
        unsafe {
            rfft_c64(&real_f32, &mut out_c64, 1.0);
        }
        assert_close_c64(
            &out_c64,
            expected_f32,
            1e-6 * scale + 1e-5,
            &format!("rfft c64 n={}", n),
        );
    }
}

#[test]
fn test_rfft_irfft_roundtrip_400() {
    let n = 400;
    let original_f64: Vec<f64> = deterministic_samples(n, 0x1122_3344)
        .iter()
        .map(|c| c.re)
        .collect();

    let mut spectrum_c128 = vec![Complex128::default(); n / 2 + 1];
    let mut recovered_f64 = vec![0.0f64; n];
    unsafe {
        rfft_c128(&original_f64, &mut spectrum_c128, 1.0);
        irfft_c128(&spectrum_c128, &mut recovered_f64, 1.0 / n as f64);
    }
    for i in 0..n {
        assert!(
            (recovered_f64[i] - original_f64[i]).abs() < 1e-12,
            "c128 sample {}: got {}, want {}",
            i,
            recovered_f64[i],
            original_f64[i]
        );
    }

    let original_f32: Vec<f32> = original_f64.iter().map(|&x| x as f32).collect();
    let mut spectrum_c64 = vec![Complex64::default(); n / 2 + 1];
    let mut recovered_f32 = vec![0.0f32; n];
    unsafe {
        rfft_c64(&original_f32, &mut spectrum_c64, 1.0);
        irfft_c64(&spectrum_c64, &mut recovered_f32, 1.0 / n as f32);
    }
    for i in 0..n {
        assert!(
            (recovered_f32[i] - original_f32[i]).abs() < 1e-5,
            "c64 sample {}: got {}, want {}",
            i,
            recovered_f32[i],
            original_f32[i]
        );
    }
}

#[test]
fn test_rfft_irfft_roundtrip_odd_sizes() {
    // Odd N cannot be inferred from the spectrum length, so the kernels take
    // N from the output slice. Cover both parities of `N/2`.
    for &n in &[3usize, 5, 7, 101] {
        let original: Vec<f64> = deterministic_samples(n, 0x9988_7766 ^ n as u64)
            .iter()
            .map(|c| c.re)
            .collect();

        let mut spectrum = vec![Complex128::default(); n / 2 + 1];
        let mut recovered = vec![0.0f64; n];
        unsafe {
            rfft_c128(&original, &mut spectrum, 1.0);
            irfft_c128(&spectrum, &mut recovered, 1.0 / n as f64);
        }

        for i in 0..n {
            assert!(
                (recovered[i] - original[i]).abs() < 1e-12,
                "n={} sample {}: got {}, want {}",
                n,
                i,
                recovered[i],
                original[i]
            );
        }
    }
}

#[test]
fn test_bluestein_size_one() {
    let input = [Complex128::new(2.5, -1.25)];
    let mut output = [Complex128::default(); 1];
    unsafe {
        fft_c128(&input, &mut output, false, 1.0);
    }
    assert!((output[0].re - 2.5).abs() < 1e-15);
    assert!((output[0].im + 1.25).abs() < 1e-15);

    // A 3-point plan used directly, checked against the closed-form DFT.
    let plan = BluesteinPlan::new(3, false);
    assert_eq!(plan.n, 3);
    let x = [
        Complex128::new(1.0, 0.0),
        Complex128::new(2.0, 0.0),
        Complex128::new(3.0, 0.0),
    ];
    let mut y = [Complex128::default(); 3];
    plan.execute_c128(&x, &mut y, 1.0);

    // numpy.fft.fft([1,2,3]) == [6, -1.5+0.8660254j, -1.5-0.8660254j]
    assert!((y[0].re - 6.0).abs() < 1e-12 && y[0].im.abs() < 1e-12);
    assert!((y[1].re + 1.5).abs() < 1e-12 && (y[1].im - 0.866_025_403_784_438_6).abs() < 1e-12);
    assert!((y[2].re + 1.5).abs() < 1e-12 && (y[2].im + 0.866_025_403_784_438_6).abs() < 1e-12);
}
