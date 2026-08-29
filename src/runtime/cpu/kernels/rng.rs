//! Shared PRNG and distribution sampling for CPU kernels.
//!
//! Provides Xoshiro256++ as the standard PRNG and distribution samplers
//! that replace the `rand` and `rand_distr` crate dependencies.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Xoshiro256++ PRNG
// ---------------------------------------------------------------------------

/// Xoshiro256++ state (Blackman & Vigna 2018).
#[derive(Clone)]
pub(crate) struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Create from seed using SplitMix64 to expand the seed.
    #[inline(always)]
    pub(crate) fn from_seed(seed: u64) -> Self {
        let mut sm_state = seed;
        let mut splitmix = || {
            sm_state = sm_state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };

        Self {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    /// Generate next u64.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

// ---------------------------------------------------------------------------
// Entropy-based seeding (no getrandom / no rand crate)
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(not(target_arch = "wasm32"))]
fn get_thread_entropy() -> u64 {
    let id = std::thread::current().id();
    let s = format!("{:?}", id);
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(target_arch = "wasm32")]
fn get_thread_entropy() -> u64 {
    // No threads on wasm, use a different mixing constant.
    0xd1342543de82ef95
}

/// Create a new Xoshiro256++ seeded from available entropy.
///
/// Uses a combination of address-space randomisation (ASLR), an atomic
/// counter, and thread ID to generate unique seeds without OS entropy.
pub(crate) fn thread_rng() -> Xoshiro256 {
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let thread_id = get_thread_entropy();
    // Mix in a stack address for ASLR entropy.
    let stack_addr = &counter as *const _ as u64;
    let seed = counter
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(thread_id)
        .wrapping_add(stack_addr);
    Xoshiro256::from_seed(seed)
}

// ---------------------------------------------------------------------------
// Primitive samplers
// ---------------------------------------------------------------------------

/// Convert a raw u64 to a uniform f64 in [0, 1) using 53 bits.
#[inline(always)]
pub(crate) fn u64_to_uniform(u: u64) -> f64 {
    (u >> 11) as f64 / (1u64 << 53) as f64
}

/// Sample a uniform f64 in [0, 1).
#[inline(always)]
pub(crate) fn sample_uniform(rng: &mut Xoshiro256) -> f64 {
    u64_to_uniform(rng.next())
}

/// Clamp a narrowed uniform sample to keep `rand`'s `[0, 1)` contract.
///
/// Narrowing an f64 sample near 1.0 to `T` can round it up to exactly 1.0
/// for low-precision floats (FP8, F16, BF16). When that happens, replace it
/// with `T::DTYPE`'s largest value strictly below 1.0
/// ([`DType::largest_value_below_one`]). A no-op for F32/F64 and integers.
#[inline(always)]
pub(crate) fn clamp_uniform_sample<T: crate::dtype::Element>(val: T) -> T {
    if val.to_f64() >= 1.0
        && let Some(bound) = T::DTYPE.largest_value_below_one()
    {
        return T::from_f64(bound);
    }
    val
}

/// Sample a standard-normal f64 (mean 0, std 1) via Box-Muller.
///
/// Generates a pair and discards the second value for simplicity.
#[inline(always)]
pub(crate) fn sample_normal(rng: &mut Xoshiro256) -> f64 {
    let u1 = sample_uniform(rng).clamp(1e-10, 1.0 - 1e-10);
    let u2 = sample_uniform(rng);
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * PI * u2).cos()
}

/// Sample a uniform integer in [low, high).
///
/// Uses rejection sampling to avoid modulo bias: we reject values from the
/// incomplete final bucket of size `u64::MAX % range` at the top of the range.
#[inline(always)]
pub(crate) fn sample_uniform_int(rng: &mut Xoshiro256, low: i64, high: i64) -> i64 {
    debug_assert!(low < high);
    let range = (high - low) as u64;
    // Largest multiple of `range` that fits in u64: reject anything >= limit.
    // For power-of-2 ranges, limit == 0 (wraps), so the loop always accepts on first try.
    let limit = range.wrapping_neg() % range; // = (2^64 - range) % range = 2^64 % range
    loop {
        let raw = rng.next();
        if raw >= limit {
            return low + (raw % range) as i64;
        }
    }
}

/// Sample from Exponential(rate) via inverse transform.
#[inline(always)]
pub(crate) fn sample_exponential(rng: &mut Xoshiro256, rate: f64) -> f64 {
    let u = sample_uniform(rng).clamp(1e-300, 1.0 - 1e-10);
    -u.ln() / rate
}

/// Sample from Gamma(shape, scale) using Marsaglia & Tsang (2000).
pub(crate) fn sample_gamma(rng: &mut Xoshiro256, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        // Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        let g = sample_gamma(rng, shape + 1.0, 1.0);
        let u = sample_uniform(rng).clamp(1e-300, 1.0);
        return g * u.powf(1.0 / shape) * scale;
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_normal(rng);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = sample_uniform(rng).clamp(1e-300, 1.0);

        // Squeeze test (fast path)
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v * scale;
        }
        // Full test
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

/// Sample from Beta(alpha, beta) via two Gamma samples.
#[inline]
pub(crate) fn sample_beta(rng: &mut Xoshiro256, alpha: f64, beta: f64) -> f64 {
    let x = sample_gamma(rng, alpha, 1.0);
    let y = sample_gamma(rng, beta, 1.0);
    x / (x + y)
}

/// Sample from Poisson(lambda).
///
/// Knuth's algorithm for small lambda (<30), normal approximation for large.
pub(crate) fn sample_poisson(rng: &mut Xoshiro256, lambda: f64) -> u64 {
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k: u64 = 0;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= sample_uniform(rng);
            if p < l {
                return k - 1;
            }
        }
    } else {
        // Normal approximation
        let val = lambda + lambda.sqrt() * sample_normal(rng);
        val.round().max(0.0) as u64
    }
}

/// Sample from Binomial(n, p).
///
/// For small n, sum of Bernoulli trials. For large n, normal approximation.
pub(crate) fn sample_binomial(rng: &mut Xoshiro256, n: u64, p: f64) -> u64 {
    if n <= 64 {
        let mut successes = 0u64;
        for _ in 0..n {
            if sample_uniform(rng) < p {
                successes += 1;
            }
        }
        successes
    } else {
        // Normal approximation: N(np, np(1-p))
        let mean = n as f64 * p;
        let std = (mean * (1.0 - p)).sqrt();
        let val = mean + std * sample_normal(rng);
        val.round().clamp(0.0, n as f64) as u64
    }
}

/// Fisher-Yates shuffle of a mutable slice.
pub(crate) fn shuffle<T>(rng: &mut Xoshiro256, slice: &mut [T]) {
    let n = slice.len();
    for i in (1..n).rev() {
        let bound = i as u64 + 1;
        let j = sample_uniform_int(rng, 0, bound as i64) as usize;
        slice.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_range() {
        let mut rng = Xoshiro256::from_seed(42);
        for _ in 0..10_000 {
            let v = sample_uniform(&mut rng);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_normal_statistics() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.1, "var = {var}");
    }

    #[test]
    fn test_uniform_int() {
        let mut rng = Xoshiro256::from_seed(42);
        for _ in 0..10_000 {
            let v = sample_uniform_int(&mut rng, -5, 10);
            assert!((-5..10).contains(&v));
        }
    }

    #[test]
    fn test_exponential_positive() {
        let mut rng = Xoshiro256::from_seed(42);
        for _ in 0..1_000 {
            assert!(sample_exponential(&mut rng, 1.0) > 0.0);
        }
    }

    #[test]
    fn test_gamma_statistics() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 10_000;
        let shape = 2.0;
        let scale = 1.0;
        let samples: Vec<f64> = (0..n)
            .map(|_| sample_gamma(&mut rng, shape, scale))
            .collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert!(samples.iter().all(|&x| x > 0.0));
        assert!((mean - shape * scale).abs() < 0.3, "mean = {mean}");
    }

    #[test]
    fn test_gamma_small_shape() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 5_000;
        let samples: Vec<f64> = (0..n).map(|_| sample_gamma(&mut rng, 0.5, 1.0)).collect();
        assert!(samples.iter().all(|&x| x > 0.0));
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.2, "mean = {mean}");
    }

    #[test]
    fn test_beta_range() {
        let mut rng = Xoshiro256::from_seed(42);
        for _ in 0..1_000 {
            let v = sample_beta(&mut rng, 2.0, 5.0);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_poisson_small() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 10_000;
        let lambda = 5.0;
        let samples: Vec<u64> = (0..n).map(|_| sample_poisson(&mut rng, lambda)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        assert!((mean - lambda).abs() < 0.5, "mean = {mean}");
    }

    #[test]
    fn test_poisson_large() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 10_000;
        let lambda = 100.0;
        let samples: Vec<u64> = (0..n).map(|_| sample_poisson(&mut rng, lambda)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        assert!((mean - lambda).abs() < 5.0, "mean = {mean}");
    }

    #[test]
    fn test_binomial_small() {
        let mut rng = Xoshiro256::from_seed(42);
        let n = 10_000;
        let trials = 10u64;
        let p = 0.5;
        let samples: Vec<u64> = (0..n)
            .map(|_| sample_binomial(&mut rng, trials, p))
            .collect();
        assert!(samples.iter().all(|&x| x <= trials));
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        assert!((mean - trials as f64 * p).abs() < 0.5, "mean = {mean}");
    }

    #[test]
    fn test_shuffle() {
        let mut rng = Xoshiro256::from_seed(42);
        let mut v: Vec<usize> = (0..100).collect();
        shuffle(&mut rng, &mut v);
        // Should still contain all elements
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        // Should not be in original order (extremely unlikely)
        assert_ne!(v, (0..100).collect::<Vec<_>>());
    }
}
