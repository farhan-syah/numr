//! FFT kernels using Stockham autosort algorithm
//!
//! This module provides CPU implementations of FFT operations.
//! The Stockham algorithm is used for its:
//! - No bit-reversal permutation (Cooley-Tukey's main bottleneck)
//! - Sequential memory access patterns
//! - Natural double-buffering
//!
//! # Algorithm: Stockham Radix-2 FFT
//!
//! ```text
//! For each stage s = 0..log2(N):
//!     half_m = 2^s
//!     m = 2^(s+1)
//!     For each group g = 0..(N/m):
//!         For each butterfly b = 0..half_m:
//!             twiddle = exp(sign * 2πi * b / m)
//!             even = src[g * half_m + b]
//!             odd = src[N/2 + g * half_m + b] * twiddle
//!             dst[g * m + b] = even + odd
//!             dst[g * m + b + half_m] = even - odd
//!     swap(src, dst)
//! ```

mod bluestein;
mod real;
mod shift;
mod stockham;
#[cfg(test)]
mod tests;

use crate::dtype::{Complex64, Complex128};
use bluestein::BluesteinPlan;
use stockham::{stockham_fft_c64, stockham_fft_c128};

pub use real::{irfft_c64, irfft_c128, rfft_c64, rfft_c128};
pub use shift::{fftshift_c64, fftshift_c128, ifftshift_c64, ifftshift_c128};
pub use stockham::{stockham_fft_batched_c64, stockham_fft_batched_c128};

// ============================================================================
// Size-Dispatching FFT Entry Points
// ============================================================================

/// Complex64 FFT for any size `N >= 1`.
///
/// Power-of-two sizes take the Stockham path unchanged; every other size uses
/// Bluestein's algorithm built on that same Stockham kernel.
///
/// # Safety
///
/// * `input` and `output` must be valid slices of the same length `N >= 1`
unsafe fn fft_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    inverse: bool,
    normalize_factor: f32,
) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    debug_assert!(n >= 1, "N must be >= 1");

    if n.is_power_of_two() {
        stockham_fft_c64(input, output, inverse, normalize_factor);
    } else {
        BluesteinPlan::new(n, inverse).execute_c64(input, output, normalize_factor);
    }
}

/// Complex128 FFT for any size `N >= 1`.
///
/// # Safety
///
/// * `input` and `output` must be valid slices of the same length `N >= 1`
unsafe fn fft_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    inverse: bool,
    normalize_factor: f64,
) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    debug_assert!(n >= 1, "N must be >= 1");

    if n.is_power_of_two() {
        stockham_fft_c128(input, output, inverse, normalize_factor);
    } else {
        BluesteinPlan::new(n, inverse).execute_c128(input, output, normalize_factor);
    }
}
