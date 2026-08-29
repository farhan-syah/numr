//! PCG64 PRNG kernel
//!
//! Permuted congruential generator from O'Neill "PCG: A Family of Simple Fast
//! Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014)

use super::box_muller;
use crate::dtype::Element;

const PCG64_MULTIPLIER: u128 = 0x2360ed051fc65da44385df649fccf645u128;

/// PCG64 state advance with XSL-RR output function
#[inline(always)]
fn pcg64_step(state: &mut u128) -> u64 {
    let old_state = *state;
    *state = old_state.wrapping_mul(PCG64_MULTIPLIER).wrapping_add(1);

    // XSL-RR output function
    let xorshifted = (((old_state >> 64) ^ old_state) >> 59) as u64;
    let rot = (old_state >> 122) as u32;
    xorshifted.rotate_right(rot)
}

/// Convert u64 to uniform float in [0, 1)
#[inline(always)]
fn u64_to_uniform(u: u64) -> f64 {
    // Use 53 bits for full double precision
    (u >> 11) as f64 / (1u64 << 53) as f64
}

/// Generate uniform random values in [0, 1) using PCG64
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn pcg64_uniform_kernel<T: Element>(out: *mut T, n: usize, seed: u64, stream: u64) {
    // Initialize state
    let mut state = ((seed as u128) << 64) | (stream as u128);
    state = state.wrapping_mul(PCG64_MULTIPLIER).wrapping_add(1);

    let out_slice = std::slice::from_raw_parts_mut(out, n);

    for elem in out_slice.iter_mut() {
        let u = pcg64_step(&mut state);
        let val = u64_to_uniform(u);
        *elem = crate::runtime::cpu::kernels::rng::clamp_uniform_sample(T::from_f64(val));
    }
}

/// Generate standard normal random values using PCG64 + Box-Muller
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn pcg64_randn_kernel<T: Element>(out: *mut T, n: usize, seed: u64, stream: u64) {
    let mut state = ((seed as u128) << 64) | (stream as u128);
    state = state.wrapping_mul(PCG64_MULTIPLIER).wrapping_add(1);

    let out_slice = std::slice::from_raw_parts_mut(out, n);

    let mut i = 0;
    while i < n {
        let u1 = u64_to_uniform(pcg64_step(&mut state));
        let u2 = u64_to_uniform(pcg64_step(&mut state));
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
