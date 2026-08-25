//! FFT shift kernels

use crate::dtype::{Complex64, Complex128};

/// Shift zero-frequency component to center
///
/// For 1D array of length N, swaps [0..N/2] with [N/2..N]
#[allow(clippy::manual_memcpy)]
pub unsafe fn fftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = n / 2;

    // Copy second half to first half of output
    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    // Copy first half to second half of output
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (undo fftshift)
#[allow(clippy::manual_memcpy, clippy::manual_div_ceil)]
pub unsafe fn ifftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = (n + 1) / 2; // For odd lengths, first half is larger

    // For ifftshift: swap [0..ceil(N/2)] with [ceil(N/2)..N]
    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
    }
}

/// Shift zero-frequency component to center (f64)
#[allow(clippy::manual_memcpy)]
pub unsafe fn fftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = n / 2;

    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (f64)
#[allow(clippy::manual_memcpy, clippy::manual_div_ceil)]
pub unsafe fn ifftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = (n + 1) / 2;

    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
    }
}
