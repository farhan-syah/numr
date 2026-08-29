//! ThreeFry4x64-20 PRNG kernel
//!
//! 20-round Threefish-based cipher from Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

use super::box_muller;
use crate::dtype::Element;

const THREEFRY_ROTATION: [[u32; 4]; 8] = [
    [14, 16, 52, 57],
    [23, 40, 5, 37],
    [33, 48, 46, 12],
    [17, 34, 22, 32],
    [13, 50, 10, 17],
    [25, 29, 39, 43],
    [26, 24, 20, 10],
    [37, 38, 19, 22],
];

const THREEFRY_PARITY64: u64 = 0x1BD11BDAA9FC1A22;

/// ThreeFry round function
#[inline(always)]
fn threefry_round(x: &mut [u64; 4], ks: &[u64; 5], r: usize) {
    // Add round key every 4 rounds
    if r.is_multiple_of(4) {
        let d = r / 4;
        x[0] = x[0].wrapping_add(ks[d % 5]);
        x[1] = x[1].wrapping_add(ks[(d + 1) % 5]);
        x[2] = x[2].wrapping_add(ks[(d + 2) % 5]);
        x[3] = x[3].wrapping_add(ks[(d + 3) % 5]).wrapping_add(d as u64);
    }

    // MIX: add + rotate
    let rot = &THREEFRY_ROTATION[r % 8];

    x[0] = x[0].wrapping_add(x[1]);
    x[1] = x[1].rotate_left(rot[0]) ^ x[0];

    x[2] = x[2].wrapping_add(x[3]);
    x[3] = x[3].rotate_left(rot[1]) ^ x[2];

    // Permute
    x.swap(1, 3);
}

/// ThreeFry4x64-20: 20-round Threefish cipher
#[inline(always)]
fn threefry4x64_20(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    // Extend key with parity
    let ks = [key[0], key[1], 0, 0, key[0] ^ key[1] ^ THREEFRY_PARITY64];

    let mut x = ctr;

    for r in 0..20 {
        threefry_round(&mut x, &ks, r);
    }

    // Final key addition
    x[0] = x[0].wrapping_add(ks[0]);
    x[1] = x[1].wrapping_add(ks[1]);
    x[2] = x[2].wrapping_add(ks[2]);
    x[3] = x[3].wrapping_add(ks[3]).wrapping_add(5);

    x
}

/// Convert u64 to uniform float in [0, 1)
/// Convert u64 to uniform float in [0, 1) using the top 53 bits (full f64
/// precision; correctly rounded when cast to f32).
#[inline(always)]
fn u64_to_uniform(u: u64) -> f64 {
    (u >> 11) as f64 / (1u64 << 53) as f64
}

/// Generate uniform random values in [0, 1) using ThreeFry4x64-20
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn threefry_uniform_kernel<T: Element>(
    out: *mut T,
    n: usize,
    key: u64,
    counter_base: u64,
) {
    let key_arr = [key, 0];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    for i in (0..n).step_by(4) {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [counter, 0, 0, 0];

        let random = threefry4x64_20(ctr, key_arr);

        for j in 0..4 {
            if i + j < n {
                let val = u64_to_uniform(random[j]);
                out_slice[i + j] =
                    crate::runtime::cpu::kernels::rng::clamp_uniform_sample(T::from_f64(val));
            }
        }
    }
}

/// Generate standard normal random values using ThreeFry4x64-20 + Box-Muller
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements
pub unsafe fn threefry_randn_kernel<T: Element>(
    out: *mut T,
    n: usize,
    key: u64,
    counter_base: u64,
) {
    let key_arr = [key, 0];
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    let mut i = 0;
    while i < n {
        let counter = counter_base.wrapping_add((i / 4) as u64);
        let ctr = [counter, 0, 0, 0];

        let random = threefry4x64_20(ctr, key_arr);

        // Box-Muller: generate pairs
        for j in (0..4).step_by(2) {
            if i + j >= n {
                break;
            }

            let u1 = u64_to_uniform(random[j]);
            let u2 = u64_to_uniform(random[j + 1]);
            let (z0, z1) = box_muller(u1, u2);

            out_slice[i + j] = T::from_f64(z0);
            if i + j + 1 < n {
                out_slice[i + j + 1] = T::from_f64(z1);
            }
        }

        i += 4;
    }
}
