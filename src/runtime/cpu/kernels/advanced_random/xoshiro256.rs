//! Xoshiro256++ PRNG kernel
//!
//! Shift-register generator from Blackman & Vigna "Scrambled Linear Pseudorandom Number Generators" (2018)
//! This is the default PRNG used by Rust's `rand` crate.

use super::box_muller;
use crate::dtype::Element;

/// Xoshiro256++ state
#[derive(Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Create from seed using SplitMix64
    #[inline(always)]
    fn from_seed(seed: u64) -> Self {
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

    /// Generate next u64
    #[inline(always)]
    fn next(&mut self) -> u64 {
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

/// Convert u64 to uniform float in [0, 1)
#[inline(always)]
fn u64_to_uniform(u: u64) -> f64 {
    // Use 53 bits for full double precision
    (u >> 11) as f64 / (1u64 << 53) as f64
}

/// Generate uniform random values in [0, 1) using Xoshiro256++
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn xoshiro256_uniform_kernel<T: Element>(out: *mut T, n: usize, seed: u64) {
    let mut rng = Xoshiro256::from_seed(seed);
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    for elem in out_slice.iter_mut() {
        let u = rng.next();
        let val = u64_to_uniform(u);
        *elem = crate::runtime::cpu::kernels::rng::clamp_uniform_sample(T::from_f64(val));
    }
}

/// Generate standard normal random values using Xoshiro256++ + Box-Muller
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn xoshiro256_randn_kernel<T: Element>(out: *mut T, n: usize, seed: u64) {
    let mut rng = Xoshiro256::from_seed(seed);
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    let mut i = 0;
    while i < n {
        let u1 = u64_to_uniform(rng.next());
        let u2 = u64_to_uniform(rng.next());
        let (z0, z1) = box_muller(u1, u2);

        out_slice[i] = T::from_f64(z0);
        if i + 1 < n {
            out_slice[i + 1] = T::from_f64(z1);
            i += 2;
        } else {
            i += 1;
        }
    }
}
