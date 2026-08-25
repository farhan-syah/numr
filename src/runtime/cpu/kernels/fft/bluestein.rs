//! Bluestein (Chirp-Z) FFT for arbitrary sizes

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

use super::stockham::stockham_fft_c128;

/// Precomputed chirp sequence and kernel spectrum for a Bluestein transform.
///
/// Bluestein rewrites an N-point DFT as a cyclic convolution of length
/// `M = (2N - 1).next_power_of_two()`, which the existing radix-2 Stockham
/// kernel evaluates directly. This makes every size `N >= 1` available without
/// padding the signal (which would change the frequency grid).
///
/// With `w[k] = exp(sign * i * pi * k^2 / N)` (`sign = -1` forward, `+1` inverse):
///
/// ```text
/// a[j] = x[j] * w[j]                (j < N, zero padded to M)
/// b[t] = conj(w[t]), b[M - t] = b[t] (1 <= t < N, zero elsewhere)
/// c    = ifft_M(fft_M(a) * fft_M(b))
/// X[k] = w[k] * c[k]
/// ```
///
/// The chirp phase uses `pi * (k^2 mod 2N) / N`, with `k^2 mod 2N` accumulated in
/// integer arithmetic. Evaluating `pi * k * k / N` directly destroys all precision
/// once `k^2` exceeds the f64 mantissa.
///
/// Everything is accumulated in f64 (`Complex128`) regardless of the caller's dtype;
/// the f32 entry points narrow only on the final store.
pub(super) struct BluesteinPlan {
    pub(super) n: usize,
    m: usize,
    /// `w[k]` for `k < N`
    chirp: Vec<Complex128>,
    /// `fft_M(b)` — the convolution kernel spectrum, length M
    kernel_spectrum: Vec<Complex128>,
}

impl BluesteinPlan {
    /// Build a plan for an N-point transform.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub(super) fn new(n: usize, inverse: bool) -> Self {
        assert!(n >= 1, "Bluestein FFT requires N >= 1");

        let m = (2 * n - 1).next_power_of_two();
        let sign = if inverse { 1.0f64 } else { -1.0f64 };
        let two_n = 2 * n;

        // k^2 mod 2N accumulated incrementally: q_{k+1} = q_k + 2k + 1 (mod 2N)
        let mut chirp = Vec::with_capacity(n);
        let mut q = 0usize;
        for k in 0..n {
            let theta = sign * PI * (q as f64) / (n as f64);
            chirp.push(Complex128::new(theta.cos(), theta.sin()));
            q = (q + 2 * k + 1) % two_n;
        }

        // Kernel b: conj(chirp), mirrored into the tail of the M-point buffer.
        // M >= 2N - 1 guarantees the head (t < N) and tail (M - t >= N) never collide.
        let mut kernel = vec![Complex128::default(); m];
        kernel[0] = chirp[0].conj();
        for t in 1..n {
            let v = chirp[t].conj();
            kernel[t] = v;
            kernel[m - t] = v;
        }

        let mut kernel_spectrum = vec![Complex128::default(); m];
        unsafe {
            stockham_fft_c128(&kernel, &mut kernel_spectrum, false, 1.0);
        }

        Self {
            n,
            m,
            chirp,
            kernel_spectrum,
        }
    }

    /// Build the zero-padded, chirp-premultiplied input buffer.
    fn premultiply<I: Iterator<Item = Complex128>>(&self, samples: I) -> Vec<Complex128> {
        let mut a = vec![Complex128::default(); self.m];
        for (k, x) in samples.enumerate().take(self.n) {
            a[k] = x * self.chirp[k];
        }
        a
    }

    /// Cyclic convolution of `a` with the precomputed kernel, via the radix-2 FFT.
    fn convolve(&self, a: &[Complex128]) -> Vec<Complex128> {
        let m = self.m;

        let mut spectrum = vec![Complex128::default(); m];
        unsafe {
            stockham_fft_c128(a, &mut spectrum, false, 1.0);
        }

        for (s, k) in spectrum.iter_mut().zip(self.kernel_spectrum.iter()) {
            *s *= *k;
        }

        let mut conv = vec![Complex128::default(); m];
        unsafe {
            stockham_fft_c128(&spectrum, &mut conv, true, 1.0 / m as f64);
        }
        conv
    }

    /// Full unnormalized N-point transform, in f64.
    fn transform<I: Iterator<Item = Complex128>>(&self, samples: I) -> Vec<Complex128> {
        let a = self.premultiply(samples);
        let conv = self.convolve(&a);
        (0..self.n).map(|k| self.chirp[k] * conv[k]).collect()
    }

    /// Complex64 (f32) transform. Input and output must both have length N.
    pub(super) fn execute_c64(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        normalize_factor: f32,
    ) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);

        let result = self.transform(
            input
                .iter()
                .map(|c| Complex128::new(c.re as f64, c.im as f64)),
        );

        let nf = normalize_factor as f64;
        for k in 0..self.n {
            output[k] = Complex64::new((result[k].re * nf) as f32, (result[k].im * nf) as f32);
        }
    }

    /// Complex128 (f64) transform. Input and output must both have length N.
    pub(super) fn execute_c128(
        &self,
        input: &[Complex128],
        output: &mut [Complex128],
        normalize_factor: f64,
    ) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);

        let result = self.transform(input.iter().copied());

        for k in 0..self.n {
            output[k] = Complex128::new(
                result[k].re * normalize_factor,
                result[k].im * normalize_factor,
            );
        }
    }

    /// Real-to-complex forward transform, keeping the first `N/2 + 1` bins (f32).
    pub(super) fn execute_rfft_f32(
        &self,
        input: &[f32],
        output: &mut [Complex64],
        normalize_factor: f32,
    ) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n / 2 + 1);

        let result = self.transform(input.iter().map(|&x| Complex128::new(x as f64, 0.0)));

        let nf = normalize_factor as f64;
        for k in 0..self.n / 2 + 1 {
            output[k] = Complex64::new((result[k].re * nf) as f32, (result[k].im * nf) as f32);
        }
    }

    /// Real-to-complex forward transform, keeping the first `N/2 + 1` bins (f64).
    pub(super) fn execute_rfft_f64(
        &self,
        input: &[f64],
        output: &mut [Complex128],
        normalize_factor: f64,
    ) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n / 2 + 1);

        let result = self.transform(input.iter().map(|&x| Complex128::new(x, 0.0)));

        for k in 0..self.n / 2 + 1 {
            output[k] = Complex128::new(
                result[k].re * normalize_factor,
                result[k].im * normalize_factor,
            );
        }
    }
}
