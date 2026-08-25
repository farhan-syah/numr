//! Bluestein (Chirp-Z) FFT for arbitrary sizes.
//!
//! The chirp and kernel-spectrum TABLES live in
//! [`crate::algorithm::fft_bluestein::BluesteinTables`], shared with every other
//! backend so a CPU and a GPU transform of the same size cannot disagree about
//! the chirp. This file is the CPU execution around them.
//!
//! Everything is accumulated in f64 (`Complex128`) regardless of the caller's
//! dtype; the f32 entry points narrow only on the final store.

use crate::algorithm::fft_bluestein::BluesteinTables;
use crate::dtype::{Complex64, Complex128};

use super::stockham::stockham_fft_c128;

/// A [`BluesteinTables`] plus the CPU convolution that executes it.
///
/// Bluestein rewrites an N-point DFT as a cyclic convolution of length
/// `M = (2N - 1).next_power_of_two()`, which the radix-2 Stockham kernel
/// evaluates directly. This makes every size `N >= 1` available without padding
/// the signal, which would change the frequency grid.
pub(super) struct BluesteinPlan {
    tables: BluesteinTables,
}

impl BluesteinPlan {
    /// Build a plan for an N-point transform.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    /// Build a plan for an N-point transform.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub(super) fn new(n: usize, inverse: bool) -> Self {
        let tables = BluesteinTables::new(n, inverse, |kernel| {
            let mut spectrum = vec![Complex128::default(); kernel.len()];
            // SAFETY: `kernel.len()` is a power of two by construction of M, and
            // `spectrum` is allocated to exactly that length.
            unsafe {
                stockham_fft_c128(kernel, &mut spectrum, false, 1.0);
            }
            spectrum
        });
        Self { tables }
    }

    /// Transform length. Kept as an accessor because the tables own the field.
    pub(super) fn n(&self) -> usize {
        self.tables.n
    }

    /// Build the zero-padded, chirp-premultiplied input buffer.
    fn premultiply<I: Iterator<Item = Complex128>>(&self, samples: I) -> Vec<Complex128> {
        let mut a = vec![Complex128::default(); self.tables.m];
        for (k, x) in samples.enumerate().take(self.n()) {
            a[k] = x * self.tables.chirp[k];
        }
        a
    }

    /// Cyclic convolution of `a` with the precomputed kernel, via the radix-2 FFT.
    fn convolve(&self, a: &[Complex128]) -> Vec<Complex128> {
        let m = self.tables.m;

        let mut spectrum = vec![Complex128::default(); m];
        unsafe {
            stockham_fft_c128(a, &mut spectrum, false, 1.0);
        }

        for (s, k) in spectrum.iter_mut().zip(self.tables.kernel_spectrum.iter()) {
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
        (0..self.n())
            .map(|k| self.tables.chirp[k] * conv[k])
            .collect()
    }

    /// Complex64 (f32) transform. Input and output must both have length N.
    pub(super) fn execute_c64(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
        normalize_factor: f32,
    ) {
        debug_assert_eq!(input.len(), self.n());
        debug_assert_eq!(output.len(), self.n());

        let result = self.transform(
            input
                .iter()
                .map(|c| Complex128::new(c.re as f64, c.im as f64)),
        );

        let nf = normalize_factor as f64;
        for k in 0..self.n() {
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
        debug_assert_eq!(input.len(), self.n());
        debug_assert_eq!(output.len(), self.n());

        let result = self.transform(input.iter().copied());

        for k in 0..self.n() {
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
        debug_assert_eq!(input.len(), self.n());
        debug_assert_eq!(output.len(), self.n() / 2 + 1);

        let result = self.transform(input.iter().map(|&x| Complex128::new(x as f64, 0.0)));

        let nf = normalize_factor as f64;
        for k in 0..self.n() / 2 + 1 {
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
        debug_assert_eq!(input.len(), self.n());
        debug_assert_eq!(output.len(), self.n() / 2 + 1);

        let result = self.transform(input.iter().map(|&x| Complex128::new(x, 0.0)));

        for k in 0..self.n() / 2 + 1 {
            output[k] = Complex128::new(
                result[k].re * normalize_factor,
                result[k].im * normalize_factor,
            );
        }
    }
}
