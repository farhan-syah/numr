use super::*;

/// Naive O(M^2) DFT, used only to give the tables a reference that shares no
/// code with the backends under test.
fn naive_dft(x: &[Complex128]) -> Vec<Complex128> {
    let m = x.len();
    (0..m)
        .map(|k| {
            let mut acc = Complex128::new(0.0, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let theta = -2.0 * PI * (k as f64) * (j as f64) / (m as f64);
                acc += xj * Complex128::new(theta.cos(), theta.sin());
            }
            acc
        })
        .collect()
}

#[test]
fn convolution_length_is_the_next_power_of_two_past_2n_minus_1() {
    for (n, expected_m) in [(1, 1), (3, 8), (5, 16), (20, 64), (1920, 4096)] {
        let t = BluesteinTables::new(n, false, naive_dft);
        assert_eq!(t.m, expected_m, "n = {n}");
        assert!(t.m >= 2 * n - 1, "n = {n}: M must cover 2N-1");
    }
}

#[test]
fn chirp_has_unit_magnitude_at_every_index() {
    // The chirp is a pure phase. A magnitude drift means the integer recurrence
    // for k^2 mod 2N has gone wrong, which is silent otherwise.
    for n in [1, 2, 7, 64, 1920] {
        for &inverse in &[false, true] {
            for (k, c) in chirp_sequence(n, inverse).iter().enumerate() {
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                assert!((mag - 1.0).abs() < 1e-12, "n={n} k={k} |w|={mag}");
            }
        }
    }
}

#[test]
fn forward_and_inverse_chirps_are_conjugates() {
    let n = 97;
    let f = chirp_sequence(n, false);
    let i = chirp_sequence(n, true);
    for k in 0..n {
        assert!((f[k].re - i[k].re).abs() < 1e-12, "k={k}");
        assert!((f[k].im + i[k].im).abs() < 1e-12, "k={k}");
    }
}

#[test]
fn integer_recurrence_beats_direct_k_squared_at_large_n() {
    // The reason `chirp_sequence` accumulates `k^2 mod 2N` in integers: at
    // n = 1920 and k near the end, `pi * k * k / n` in f64 has already lost the
    // low bits of k^2, so the phase is wrong. This pins WHY the recurrence is
    // there — a "simplification" back to the direct formula fails here.
    let n = 1_000_003; // prime, so 2N shares no factors that would mask the error
    let chirp = chirp_sequence(n, false);
    let k = n - 1;
    let q = (k as u128 * k as u128) % (2 * n as u128);
    let exact = PI * (q as f64) / (n as f64);
    let expected = Complex128::new(exact.cos(), -exact.sin());
    assert!(
        (chirp[k].re - expected.re).abs() < 1e-9 && (chirp[k].im - expected.im).abs() < 1e-9,
        "recurrence diverged at k={k}: {:?} vs {:?}",
        chirp[k],
        expected
    );
}

#[test]
fn kernel_is_mirrored_without_self_collision() {
    let n = 5;
    let chirp = chirp_sequence(n, false);
    let m = (2 * n - 1).next_power_of_two();
    let kernel = kernel_from_chirp(&chirp, m);
    for t in 1..n {
        assert_eq!(kernel[t].re, kernel[m - t].re, "t={t}");
        assert_eq!(kernel[t].im, kernel[m - t].im, "t={t}");
    }
    // Everything between the head and the mirrored tail stays zero; a collision
    // would show up here as a nonzero entry.
    for slot in kernel.iter().take(m - n + 1).skip(n) {
        assert_eq!(slot.re, 0.0);
        assert_eq!(slot.im, 0.0);
    }
}

#[test]
fn narrowing_preserves_length_and_interleaving() {
    let t = BluesteinTables::new(7, false, naive_dft);
    assert_eq!(t.chirp_f32().len(), 2 * t.n);
    assert_eq!(t.chirp_f64().len(), 2 * t.n);
    assert_eq!(t.kernel_spectrum_f32().len(), 2 * t.m);
    assert_eq!(t.kernel_spectrum_f64().len(), 2 * t.m);
    let f64s = t.chirp_f64();
    assert_eq!(f64s[0], t.chirp[0].re);
    assert_eq!(f64s[1], t.chirp[0].im);
}
