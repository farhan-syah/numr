use super::*;

/// Reference: exact i128 accumulation, clamped once — the same contract the
/// scalar wide-accumulator kernel implements.
fn reference(a: &[i32], b: &[i32], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i128;
            for kk in 0..k {
                acc += a[i * k + kk] as i128 * b[kk * n + j] as i128;
            }
            out[i * n + j] = acc.clamp(i32::MIN as i128, i32::MAX as i128) as i32;
        }
    }
    out
}

fn run_avx2(a: &[i32], b: &[i32], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    unsafe {
        matmul_i32_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, n, k, k, n, n);
    }
    out
}

fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[test]
fn matches_the_exact_reference_on_mixed_signs() {
    if !avx2_available() {
        return;
    }
    let (m, n, k) = (5usize, 13usize, 17usize);
    let a: Vec<i32> = (0..m * k).map(|i| (i as i32 % 21) - 10).collect();
    let b: Vec<i32> = (0..k * n).map(|i| (i as i32 % 17) - 8).collect();
    assert_eq!(run_avx2(&a, &b, m, n, k), reference(&a, &b, m, n, k));
}

#[test]
fn the_column_tail_is_not_dropped() {
    if !avx2_available() {
        return;
    }
    // n values that are not multiples of the 4-lane width: the scalar tail loop
    // is the part a chunked kernel most often gets wrong.
    for n in [1usize, 2, 3, 4, 5, 7, 9] {
        let (m, k) = (3usize, 6usize);
        let a: Vec<i32> = (0..m * k).map(|i| (i as i32 % 7) - 3).collect();
        let b: Vec<i32> = (0..k * n).map(|i| (i as i32 % 5) - 2).collect();
        assert_eq!(
            run_avx2(&a, &b, m, n, k),
            reference(&a, &b, m, n, k),
            "n = {n}"
        );
    }
}

#[test]
fn a_zero_row_of_a_is_skipped_without_changing_the_result() {
    if !avx2_available() {
        return;
    }
    // The kernel short-circuits on `a_val == 0`. That must be an optimization,
    // not a behaviour change.
    let (m, n, k) = (2usize, 8usize, 4usize);
    let a = vec![0i32; m * k];
    let b: Vec<i32> = (0..k * n).map(|i| i as i32 - 16).collect();
    assert_eq!(run_avx2(&a, &b, m, n, k), vec![0i32; m * n]);
}

#[test]
fn results_saturate_rather_than_wrap() {
    if !avx2_available() {
        return;
    }
    // Each product is 2^30, and k = 8 sums to 2^33 — past i32 but far inside
    // i64, so the guard admits it and the kernel must clamp.
    let (m, n, k) = (1usize, 4usize, 8usize);
    let a = vec![1 << 15; m * k];
    let b = vec![1 << 15; k * n];
    assert!(unsafe { matmul_i32_fits_i64(a.as_ptr(), b.as_ptr(), m, n, k, k, n) });
    assert_eq!(run_avx2(&a, &b, m, n, k), vec![i32::MAX; m * n]);
}

#[test]
fn the_guard_admits_small_operands_and_rejects_overflowing_ones() {
    let (m, n, k) = (4usize, 4usize, 1024usize);

    let small = vec![1000i32; m.max(k) * n.max(k)];
    assert!(
        unsafe { matmul_i32_fits_i64(small.as_ptr(), small.as_ptr(), m, n, k, k, n) },
        "1000 * 1000 * 1024 is nowhere near i64::MAX"
    );

    // i32::MIN squared is 2^62; 1024 of them is 2^72, well past i64.
    let big = vec![i32::MIN; m.max(k) * n.max(k)];
    assert!(
        !unsafe { matmul_i32_fits_i64(big.as_ptr(), big.as_ptr(), m, n, k, k, n) },
        "the guard must reject operands whose partial sums leave i64"
    );
}

#[test]
fn the_guard_uses_magnitude_not_sign() {
    // `-i32::MIN` does not fit i32, so a guard that took the absolute value in
    // i32 would overflow inside the check itself. `max_abs` returns i64 for
    // exactly this input.
    let one = [i32::MIN];
    // 2^31 * 2^31 * 1 = 2^62, which is inside i64, so a single product of the
    // two most negative i32 values IS admissible.
    assert!(unsafe { matmul_i32_fits_i64(one.as_ptr(), one.as_ptr(), 1, 1, 1, 1, 1) });
    // Two of them is 2^63, which is not.
    let two = [i32::MIN, i32::MIN];
    assert!(!unsafe { matmul_i32_fits_i64(two.as_ptr(), two.as_ptr(), 1, 1, 2, 2, 1) });
}

#[test]
fn an_empty_k_is_admissible_and_yields_zeros() {
    if !avx2_available() {
        return;
    }
    let a: Vec<i32> = Vec::new();
    let b: Vec<i32> = Vec::new();
    assert!(unsafe { matmul_i32_fits_i64(a.as_ptr(), b.as_ptr(), 2, 3, 0, 0, 3) });
    let mut out = vec![7i32; 6];
    unsafe {
        matmul_i32_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 2, 3, 0, 0, 3, 3);
    }
    assert_eq!(
        out,
        vec![0i32; 6],
        "k = 0 must zero the output, not leave it"
    );
}

#[test]
fn the_avx2_path_and_the_i128_path_agree_through_the_real_dispatch() {
    // The kernel above is only reached via `matmul_kernel`, which picks between
    // it and the exact i128 scalar path. That choice must be invisible: this
    // runs the same operands through the public entry point and against the
    // reference, so a divergence between the two paths shows up here rather
    // than as a silently different answer in a caller.
    use crate::runtime::cpu::kernels::matmul_kernel;

    // Sizes chosen to straddle the 4-lane width and to include a k large enough
    // that partial sums matter.
    for (m, n, k) in [(1usize, 1usize, 1usize), (3, 4, 8), (2, 7, 33), (6, 16, 64)] {
        let a: Vec<i32> = (0..m * k).map(|i| ((i * 37) as i32 % 601) - 300).collect();
        let b: Vec<i32> = (0..k * n).map(|i| ((i * 53) as i32 % 409) - 204).collect();

        let mut got = vec![0i32; m * n];
        unsafe {
            matmul_kernel(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), m, n, k, k, n, n);
        }
        assert_eq!(got, reference(&a, &b, m, n, k), "m={m} n={n} k={k}");
    }
}

#[test]
fn operands_the_guard_rejects_still_produce_the_exact_answer() {
    // i32::MIN squared is 2^62, so k = 4 of them leaves i64 and the guard sends
    // this to the i128 path. The result must still be the exact clamped sum,
    // not whatever a wrapped i64 accumulator would give.
    use crate::runtime::cpu::kernels::matmul_kernel;

    let (m, n, k) = (1usize, 4usize, 4usize);
    let a = vec![i32::MIN; m * k];
    let b = vec![i32::MIN; k * n];
    assert!(
        !unsafe { matmul_i32_fits_i64(a.as_ptr(), b.as_ptr(), m, n, k, k, n) },
        "these operands must be rejected by the guard for this test to mean anything"
    );

    let mut got = vec![0i32; m * n];
    unsafe {
        matmul_kernel(a.as_ptr(), b.as_ptr(), got.as_mut_ptr(), m, n, k, k, n, n);
    }
    // 4 * (-2^31)^2 = 2^64, positive, so it clamps to i32::MAX.
    assert_eq!(got, vec![i32::MAX; m * n]);
}
