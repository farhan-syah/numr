//! CPU kernels for quasi-random sequence generation.
//!
//! Implements Sobol, Halton, and Latin Hypercube Sampling sequences.

use super::rng;
use super::sobol_data::{MAX_SOBOL_DIMENSION, get_polynomial};
use crate::ops::common::quasirandom::{
    SOBOL_BITS, compute_direction_vectors, dimension_zero_vectors,
};

/// Generate Sobol sequence points (F32 version).
///
/// Supports up to 21,201 dimensions using Joe & Kuo (2008) direction numbers.
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension must be <= 21,201
#[inline]
pub unsafe fn sobol_f32(out: *mut f32, n_points: usize, dimension: usize, skip: usize) {
    assert!(
        dimension <= MAX_SOBOL_DIMENSION,
        "Sobol sequence supports up to {} dimensions, got {}",
        MAX_SOBOL_DIMENSION,
        dimension
    );

    for d in 0..dimension {
        sobol_dimension_f32(out.add(d), n_points, dimension, d, skip);
    }
}

/// Generate Sobol sequence for a single dimension.
#[inline]
unsafe fn sobol_dimension_f32(
    out: *mut f32,
    n_points: usize,
    stride: usize,
    dimension: usize,
    skip: usize,
) {
    // Get direction vectors for this dimension
    let v = if dimension == 0 {
        dimension_zero_vectors()
    } else {
        // Dimension was validated by caller, so polynomial must exist.
        // If missing, the embedded sobol_data.bin is corrupted.
        let poly = get_polynomial(dimension + 1)
            .expect("INTERNAL: sobol_data.bin corrupted - missing polynomial");
        compute_direction_vectors(poly)
    };

    let scale = 1.0f32 / (1u64 << 32) as f32;

    for i in 0..n_points {
        let index = skip + i;

        // Gray code: i XOR (i >> 1)
        let gray = index as u32 ^ (index as u32 >> 1);

        // Compute Sobol point using direction vectors
        let mut x = 0u32;
        for bit in 0..SOBOL_BITS {
            if (gray & (1 << bit)) != 0 {
                x ^= v[bit];
            }
        }

        // Convert to float in [0, 1)
        *out.add(i * stride) = (x as f32) * scale;
    }
}

/// Generate Sobol sequence points (F64 version).
///
/// Supports up to 21,201 dimensions using Joe & Kuo (2008) direction numbers.
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension must be <= 21,201
#[inline]
pub unsafe fn sobol_f64(out: *mut f64, n_points: usize, dimension: usize, skip: usize) {
    assert!(
        dimension <= MAX_SOBOL_DIMENSION,
        "Sobol sequence supports up to {} dimensions, got {}",
        MAX_SOBOL_DIMENSION,
        dimension
    );

    for d in 0..dimension {
        sobol_dimension_f64(out.add(d), n_points, dimension, d, skip);
    }
}

#[inline]
unsafe fn sobol_dimension_f64(
    out: *mut f64,
    n_points: usize,
    stride: usize,
    dimension: usize,
    skip: usize,
) {
    let v = if dimension == 0 {
        dimension_zero_vectors()
    } else {
        // Dimension was validated by caller, so polynomial must exist.
        // If missing, the embedded sobol_data.bin is corrupted.
        let poly = get_polynomial(dimension + 1)
            .expect("INTERNAL: sobol_data.bin corrupted - missing polynomial");
        compute_direction_vectors(poly)
    };

    let scale = 1.0f64 / (1u64 << 32) as f64;

    for i in 0..n_points {
        let index = skip + i;
        let gray = index as u32 ^ (index as u32 >> 1);

        let mut x = 0u32;
        for bit in 0..SOBOL_BITS {
            if (gray & (1 << bit)) != 0 {
                x ^= v[bit];
            }
        }

        *out.add(i * stride) = (x as f64) * scale;
    }
}

// ============================================================================
// Halton Sequence
// ============================================================================

/// First 100 prime numbers for Halton sequence
const PRIMES: [u32; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// Generate Halton sequence points (F32 version).
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_points * dimension`
/// - Dimension must be <= 100 (limited by prime numbers)
#[inline]
pub unsafe fn halton_f32(out: *mut f32, n_points: usize, dimension: usize, skip: usize) {
    for d in 0..dimension {
        let base = PRIMES[d];
        van_der_corput_f32(out.add(d), n_points, dimension, base);

        // Apply skip offset
        if skip > 0 {
            for i in 0..n_points {
                let skipped_value = van_der_corput_single_f32(skip + i, base);
                *out.add(i * dimension + d) = skipped_value;
            }
        }
    }
}

/// van der Corput sequence in given base (F32).
#[inline]
unsafe fn van_der_corput_f32(out: *mut f32, n_points: usize, stride: usize, base: u32) {
    for i in 0..n_points {
        *out.add(i * stride) = van_der_corput_single_f32(i, base);
    }
}

/// Compute single van der Corput value.
#[inline]
fn van_der_corput_single_f32(mut index: usize, base: u32) -> f32 {
    let base_f = base as f32;
    let mut result = 0.0f32;
    let mut f = 1.0 / base_f;

    while index > 0 {
        result += f * (index % base as usize) as f32;
        index /= base as usize;
        f /= base_f;
    }

    result
}

/// Generate Halton sequence points (F64 version).
#[inline]
pub unsafe fn halton_f64(out: *mut f64, n_points: usize, dimension: usize, skip: usize) {
    for d in 0..dimension {
        let base = PRIMES[d];
        van_der_corput_f64(out.add(d), n_points, dimension, base);

        if skip > 0 {
            for i in 0..n_points {
                let skipped_value = van_der_corput_single_f64(skip + i, base);
                *out.add(i * dimension + d) = skipped_value;
            }
        }
    }
}

#[inline]
unsafe fn van_der_corput_f64(out: *mut f64, n_points: usize, stride: usize, base: u32) {
    for i in 0..n_points {
        *out.add(i * stride) = van_der_corput_single_f64(i, base);
    }
}

#[inline]
fn van_der_corput_single_f64(mut index: usize, base: u32) -> f64 {
    let base_f = base as f64;
    let mut result = 0.0f64;
    let mut f = 1.0 / base_f;

    while index > 0 {
        result += f * (index % base as usize) as f64;
        index /= base as usize;
        f /= base_f;
    }

    result
}

// ============================================================================
// Latin Hypercube Sampling
// ============================================================================

/// Generate Latin Hypercube samples (F32 version).
///
/// # Safety
///
/// - `out` must point to valid memory of length `n_samples * dimension`
#[inline]
pub unsafe fn latin_hypercube_f32(out: *mut f32, n_samples: usize, dimension: usize) {
    let mut prng = rng::thread_rng();

    for d in 0..dimension {
        // Create stratified intervals
        let mut intervals: Vec<usize> = (0..n_samples).collect();

        // Shuffle intervals
        rng::shuffle(&mut prng, &mut intervals);

        // Generate random point within each interval
        for (i, &interval) in intervals.iter().enumerate() {
            let lower = interval as f32 / n_samples as f32;
            let upper = (interval + 1) as f32 / n_samples as f32;
            let random_offset = rng::clamp_uniform_sample(rng::sample_uniform(&mut prng) as f32);

            *out.add(i * dimension + d) = lower + random_offset * (upper - lower);
        }
    }
}

/// Generate Latin Hypercube samples (F64 version).
#[inline]
pub unsafe fn latin_hypercube_f64(out: *mut f64, n_samples: usize, dimension: usize) {
    let mut prng = rng::thread_rng();

    for d in 0..dimension {
        let mut intervals: Vec<usize> = (0..n_samples).collect();
        rng::shuffle(&mut prng, &mut intervals);

        for (i, &interval) in intervals.iter().enumerate() {
            let lower = interval as f64 / n_samples as f64;
            let upper = (interval + 1) as f64 / n_samples as f64;
            let random_offset = rng::sample_uniform(&mut prng);

            *out.add(i * dimension + d) = lower + random_offset * (upper - lower);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobol_f64_basic() {
        let n = 10;
        let d = 2;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            sobol_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // First point should be 0
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn test_sobol_f64_known_values() {
        // Test against known Sobol sequence values
        // First few points of dimension 0 (Gray code based):
        // index 0: gray=0 -> x=0
        // index 1: gray=1 -> x=v[0] = 0.5
        // index 2: gray=3 -> x=v[0]^v[1] = 0.5^0.25 = 0.75
        // index 3: gray=2 -> x=v[1] = 0.25
        let n = 4;
        let d = 1;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            sobol_f64(out.as_mut_ptr(), n, d, 0);
        }

        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 0.5).abs() < 1e-10);
        assert!((out[2] - 0.75).abs() < 1e-10);
        assert!((out[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_sobol_high_dimension() {
        // Test that high dimensions work (using Joe & Kuo data)
        let n = 10;
        let d = 100; // Much higher than the old 6-dimension limit
        let mut out = vec![0.0f64; n * d];

        unsafe {
            sobol_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }
    }

    #[test]
    fn test_sobol_very_high_dimension() {
        // Test dimensions up to 1000
        let n = 5;
        let d = 1000;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            sobol_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // Check that different dimensions produce different sequences
        // (not all the same value)
        let dim0_vals: Vec<f64> = (0..n).map(|i| out[i * d]).collect();
        let dim999_vals: Vec<f64> = (0..n).map(|i| out[i * d + 999]).collect();
        assert_ne!(dim0_vals, dim999_vals);
    }

    #[test]
    fn test_halton_f64_basic() {
        let n = 10;
        let d = 3;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            halton_f64(out.as_mut_ptr(), n, d, 0);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // First Halton values are known:
        // base 2: 0, 0.5, 0.25, 0.75, ...
        // base 3: 0, 1/3, 2/3, 1/9, ...
        // base 5: 0, 0.2, 0.4, 0.6, ...
        assert_eq!(out[0], 0.0); // H(0, 2)
        assert!((out[d] - 0.5).abs() < 1e-10); // H(1, 2)
    }

    #[test]
    fn test_latin_hypercube_f64_stratification() {
        let n = 20;
        let d = 2;
        let mut out = vec![0.0f64; n * d];

        unsafe {
            latin_hypercube_f64(out.as_mut_ptr(), n, d);
        }

        // Check all values in [0, 1)
        for &val in &out {
            assert!(val >= 0.0 && val < 1.0, "Value out of range: {}", val);
        }

        // Check stratification: each interval should have exactly one sample
        for dim in 0..d {
            let mut bins = vec![0; n];
            for i in 0..n {
                let val = out[i * d + dim];
                let bin = (val * n as f64).floor() as usize;
                let bin = bin.min(n - 1); // Handle edge case where val is very close to 1.0
                bins[bin] += 1;
            }

            // Each bin should have exactly 1 sample
            for &count in &bins {
                assert_eq!(count, 1, "Latin hypercube not properly stratified");
            }
        }
    }

    #[test]
    fn test_van_der_corput_base2() {
        // van der Corput in base 2:
        // 0 -> 0
        // 1 -> 0.5
        // 2 -> 0.25
        // 3 -> 0.75
        // 4 -> 0.125
        assert_eq!(van_der_corput_single_f64(0, 2), 0.0);
        assert!((van_der_corput_single_f64(1, 2) - 0.5).abs() < 1e-10);
        assert!((van_der_corput_single_f64(2, 2) - 0.25).abs() < 1e-10);
        assert!((van_der_corput_single_f64(3, 2) - 0.75).abs() < 1e-10);
        assert!((van_der_corput_single_f64(4, 2) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_van_der_corput_base3() {
        // van der Corput in base 3:
        // 0 -> 0
        // 1 -> 1/3
        // 2 -> 2/3
        // 3 -> 1/9
        assert_eq!(van_der_corput_single_f64(0, 3), 0.0);
        assert!((van_der_corput_single_f64(1, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput_single_f64(2, 3) - 2.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput_single_f64(3, 3) - 1.0 / 9.0).abs() < 1e-10);
    }
}
