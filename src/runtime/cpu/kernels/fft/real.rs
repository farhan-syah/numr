//! Real FFT kernels (rfft/irfft)

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

use super::bluestein::BluesteinPlan;
use super::stockham::{stockham_fft_c64, stockham_fft_c128};
use super::{fft_c64, fft_c128};

// ============================================================================
// Real FFT Kernels (rfft/irfft)
// ============================================================================

/// Real-to-complex FFT using the "packing" trick
///
/// For N real inputs, we pack them as N/2 complex values:
///   z[k] = x[2k] + i*x[2k+1]
///
/// Then compute N/2-point complex FFT and unpack to get N/2+1 complex outputs.
///
/// # Arguments
///
/// * `input` - Real input data of length N (must be power of 2)
/// * `output` - Complex output buffer of length N/2 + 1
///
/// # Safety
///
/// * N must be >= 1
/// * `output` must have length N/2 + 1
///
/// Sizes that are not a power of two >= 2 fall back to a full complex Bluestein
/// transform, keeping the first N/2 + 1 bins.
pub unsafe fn rfft_c64(input: &[f32], output: &mut [Complex64], normalize_factor: f32) {
    let n = input.len();
    debug_assert!(n >= 1, "N must be >= 1");
    debug_assert_eq!(output.len(), n / 2 + 1);

    if !(n >= 2 && n.is_power_of_two()) {
        BluesteinPlan::new(n, false).execute_rfft_f32(input, output, normalize_factor);
        return;
    }

    let half_n = n / 2;

    // Step 1: Pack real values into complex
    let mut packed: Vec<Complex64> = Vec::with_capacity(half_n);
    for k in 0..half_n {
        packed.push(Complex64::new(input[2 * k], input[2 * k + 1]));
    }

    // Step 2: Compute half-size complex FFT (no normalization yet)
    let mut fft_result = vec![Complex64::default(); half_n];
    stockham_fft_c64(&packed, &mut fft_result, false, 1.0);

    // Step 3: Unpack to get full rfft output
    // X[0] = Z[0].re + Z[0].im (DC component)
    // X[N/2] = Z[0].re - Z[0].im (Nyquist component)
    // X[k] = (Z[k] + conj(Z[N/2-k])) / 2 - i * (Z[k] - conj(Z[N/2-k])) / 2 * W_N^k
    //
    // Simplified unpack formula:
    // Xe[k] = (Z[k] + conj(Z[N/2-k])) / 2
    // Xo[k] = (Z[k] - conj(Z[N/2-k])) / 2i
    // X[k] = Xe[k] + W_N^(-k) * Xo[k]

    // DC component (k=0)
    output[0] = Complex64::new(
        (fft_result[0].re + fft_result[0].im) * normalize_factor,
        0.0,
    );

    // Middle components (k = 1 to N/2 - 1)
    for k in 1..half_n {
        let z_k = fft_result[k];
        let z_nk = fft_result[half_n - k].conj();

        let x_even = (z_k + z_nk) * Complex64::new(0.5, 0.0);
        let x_odd = (z_k - z_nk) * Complex64::new(0.0, -0.5);

        // Twiddle factor W_N^(-k)
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let twiddle = Complex64::new(theta.cos() as f32, theta.sin() as f32);

        let result = x_even + x_odd * twiddle;
        output[k] = Complex64::new(result.re * normalize_factor, result.im * normalize_factor);
    }

    // Nyquist component (k = N/2)
    output[half_n] = Complex64::new(
        (fft_result[0].re - fft_result[0].im) * normalize_factor,
        0.0,
    );
}

/// Complex-to-real inverse FFT
///
/// Takes Hermitian-symmetric complex input (N/2+1 values) and produces N real values.
///
/// # Safety
///
/// * `input` must have length N/2 + 1
/// * `output` must have length N
/// * N must be >= 1
///
/// The output length is authoritative: N is taken from `output`, so odd N (which
/// cannot be recovered from `input.len()`) is handled correctly.
pub unsafe fn irfft_c64(input: &[Complex64], output: &mut [f32], normalize_factor: f32) {
    let n = output.len();
    debug_assert!(n >= 1, "N must be >= 1");
    let half_n = n / 2;
    debug_assert_eq!(input.len(), half_n + 1);

    // Step 1: Extend Hermitian-symmetric input to full complex spectrum.
    // For even N the Nyquist bin (k == N - k) is stored once, without conjugation.
    let mut full_spectrum = vec![Complex64::default(); n];
    full_spectrum[0] = input[0];
    for k in 1..=half_n {
        full_spectrum[k] = input[k];
        if n - k != k {
            full_spectrum[n - k] = input[k].conj();
        }
    }

    // Step 2: Compute inverse FFT (with normalization 1/N built-in)
    let mut ifft_result = vec![Complex64::default(); n];
    fft_c64(&full_spectrum, &mut ifft_result, true, normalize_factor);

    // Step 3: Extract real parts
    for i in 0..n {
        output[i] = ifft_result[i].re;
    }
}

/// Real-to-complex FFT (f64 precision)
///
/// Sizes that are not a power of two >= 2 fall back to a full complex Bluestein
/// transform, keeping the first N/2 + 1 bins.
///
/// # Safety
///
/// * N must be >= 1
/// * `output` must have length N/2 + 1
pub unsafe fn rfft_c128(input: &[f64], output: &mut [Complex128], normalize_factor: f64) {
    let n = input.len();
    debug_assert!(n >= 1, "N must be >= 1");
    debug_assert_eq!(output.len(), n / 2 + 1);

    if !(n >= 2 && n.is_power_of_two()) {
        BluesteinPlan::new(n, false).execute_rfft_f64(input, output, normalize_factor);
        return;
    }

    let half_n = n / 2;

    // Pack real values into complex
    let mut packed: Vec<Complex128> = Vec::with_capacity(half_n);
    for k in 0..half_n {
        packed.push(Complex128::new(input[2 * k], input[2 * k + 1]));
    }

    // Compute half-size complex FFT
    let mut fft_result = vec![Complex128::default(); half_n];
    stockham_fft_c128(&packed, &mut fft_result, false, 1.0);

    // Unpack to get full rfft output
    output[0] = Complex128::new(
        (fft_result[0].re + fft_result[0].im) * normalize_factor,
        0.0,
    );

    for k in 1..half_n {
        let z_k = fft_result[k];
        let z_nk = fft_result[half_n - k].conj();

        let x_even = (z_k + z_nk) * Complex128::new(0.5, 0.0);
        let x_odd = (z_k - z_nk) * Complex128::new(0.0, -0.5);

        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let twiddle = Complex128::new(theta.cos(), theta.sin());

        let result = x_even + x_odd * twiddle;
        output[k] = Complex128::new(result.re * normalize_factor, result.im * normalize_factor);
    }

    output[half_n] = Complex128::new(
        (fft_result[0].re - fft_result[0].im) * normalize_factor,
        0.0,
    );
}

/// Complex-to-real inverse FFT (f64 precision)
///
/// The output length is authoritative: N is taken from `output`, so odd N (which
/// cannot be recovered from `input.len()`) is handled correctly.
///
/// # Safety
///
/// * `input` must have length N/2 + 1
/// * `output` must have length N >= 1
pub unsafe fn irfft_c128(input: &[Complex128], output: &mut [f64], normalize_factor: f64) {
    let n = output.len();
    debug_assert!(n >= 1, "N must be >= 1");
    let half_n = n / 2;
    debug_assert_eq!(input.len(), half_n + 1);

    // Extend Hermitian-symmetric input.
    // For even N the Nyquist bin (k == N - k) is stored once, without conjugation.
    let mut full_spectrum = vec![Complex128::default(); n];
    full_spectrum[0] = input[0];
    for k in 1..=half_n {
        full_spectrum[k] = input[k];
        if n - k != k {
            full_spectrum[n - k] = input[k].conj();
        }
    }

    // Compute inverse FFT
    let mut ifft_result = vec![Complex128::default(); n];
    fft_c128(&full_spectrum, &mut ifft_result, true, normalize_factor);

    // Extract real parts
    for i in 0..n {
        output[i] = ifft_result[i].re;
    }
}
