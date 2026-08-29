//! Philox4x32-10 PRNG kernel
//!
//! 10-round Feistel cipher from Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use super::box_muller;
use crate::dtype::Element;

const PHILOX_M2X32_0: u32 = 0xD2511F53;
const PHILOX_M2X32_1: u32 = 0xCD9E8D57;
const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;

/// Philox4x32 round function
#[inline(always)]
fn philox_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let prod0 = (ctr[0] as u64).wrapping_mul(PHILOX_M2X32_0 as u64);
    let prod1 = (ctr[2] as u64).wrapping_mul(PHILOX_M2X32_1 as u64);

    [
        ((prod1 >> 32) as u32) ^ ctr[1] ^ key[0],
        prod1 as u32,
        ((prod0 >> 32) as u32) ^ ctr[3] ^ key[1],
        prod0 as u32,
    ]
}

/// Philox4x32-10: 10-round Feistel cipher
#[inline(always)]
fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;

    for _ in 0..10 {
        c = philox_round(c, k);
        k[0] = k[0].wrapping_add(PHILOX_W32_0);
        k[1] = k[1].wrapping_add(PHILOX_W32_1);
    }

    c
}

/// Convert u32 to uniform float in [0, 1)
#[inline(always)]
fn u32_to_uniform(u: u32) -> f64 {
    // Use top 24 bits for good distribution
    (u >> 8) as f64 / (1u64 << 24) as f64
}

/// Generate uniform random values in [0, 1) using Philox4x32-10
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn philox_uniform_kernel<T: Element>(
    out: *mut T,
    n: usize,
    key: u64,
    counter_base: u64,
) {
    let key_split = [(key & 0xFFFFFFFF) as u32, (key >> 32) as u32];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    for i in (0..n).step_by(4) {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [(counter & 0xFFFFFFFF) as u32, (counter >> 32) as u32, 0, 0];

        let random = philox4x32_10(ctr, key_split);

        for j in 0..4 {
            if i + j < n {
                let val = u32_to_uniform(random[j]);
                out_slice[i + j] =
                    crate::runtime::cpu::kernels::rng::clamp_uniform_sample(T::from_f64(val));
            }
        }
    }
}

/// Generate standard normal random values using Philox4x32-10 + Box-Muller
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn philox_randn_kernel<T: Element>(out: *mut T, n: usize, key: u64, counter_base: u64) {
    let key_split = [(key & 0xFFFFFFFF) as u32, (key >> 32) as u32];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    let mut i = 0;
    while i < n {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [(counter & 0xFFFFFFFF) as u32, (counter >> 32) as u32, 0, 0];

        let random = philox4x32_10(ctr, key_split);

        // Box-Muller: generate pairs of normal values from pairs of uniform
        for j in (0..4).step_by(2) {
            if i + j >= n {
                break;
            }

            let u1 = u32_to_uniform(random[j]);
            let u2 = u32_to_uniform(random[j + 1]);
            let (z0, z1) = box_muller(u1, u2);

            out_slice[i + j] = T::from_f64(z0);
            if i + j + 1 < n {
                out_slice[i + j + 1] = T::from_f64(z1);
            }
        }

        i += 4;
    }
}
