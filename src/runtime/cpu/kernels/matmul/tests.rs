use super::super::{gemv_bt_kernel, matmul_bias_kernel, matmul_kernel};

#[test]
fn test_matmul_i32_basic() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    let a = [1i32, 2, 3, 4];
    let b = [5i32, 6, 7, 8];
    let mut c = [0i32; 4];

    unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2) };
    assert_eq!(c, [19, 22, 43, 50]);
}

#[test]
fn test_matmul_i32_non_square() {
    // A(3x2) @ B(2x4) = C(3x4)
    let a = [1i32, 2, 3, 4, 5, 6];
    let b = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let mut c = [0i32; 12];

    unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4) };
    assert_eq!(c, [11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68]);
}

#[test]
fn test_matmul_i32_wide() {
    // n > 8: the width that used to select the AVX2 i32 microkernel.
    let (m, n, k) = (2, 16, 3);
    let a: Vec<i32> = (0..m * k).map(|i| (i + 1) as i32).collect();
    let b: Vec<i32> = (0..k * n).map(|i| (i + 1) as i32).collect();
    let mut c = vec![0i32; m * n];

    unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

    let mut expected = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            for kk in 0..k {
                expected[i * n + j] += a[i * k + kk] * b[kk * n + j];
            }
        }
    }
    assert_eq!(c, expected);
}

/// Catches an i32 matmul accumulator.
///
/// Column 0's dot product is 4_000_000_000, which i32 cannot hold. An i32
/// accumulator panics on the overflow in a debug build, and in a release
/// build wraps to -294_967_296 where the documented answer is the saturated
/// `i32::MAX`. Column 1 stays in range and pins that ordinary results are
/// untouched.
#[test]
fn test_matmul_i32_saturates_instead_of_wrapping() {
    let a = [2_000_000_000i32, 2_000_000_000];
    let b = [1i32, 1, 1, -1];
    let mut c = [0i32; 2];

    unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 2, 2, 2, 2, 2) };
    assert_eq!(c, [i32::MAX, 0]);
}

/// Catches an FP8 matmul accumulator.
///
/// A length-32 dot product of ones is 32. Accumulated in FP8E4M3 the
/// running sum stalls at 16, because above 16 the format's spacing is 2 and
/// `16 + 1` rounds back to 16.
#[test]
fn test_matmul_fp8_accumulates_in_f32() {
    use crate::dtype::FP8E4M3;

    let a = [FP8E4M3::from_f32(1.0); 32];
    let b = [FP8E4M3::from_f32(1.0); 32];
    let mut c = [FP8E4M3::from_f32(0.0); 1];

    unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 1, 32, 32, 1, 1) };
    assert_eq!(c[0].to_f32(), 32.0);
}

/// Same accumulator defect through the fused bias kernel.
///
/// The bias is only the starting value of a dot product that still has to
/// be accumulated wide, so an i32 accumulator fails here for exactly the
/// reason it fails in `matmul_kernel`.
#[test]
fn test_matmul_bias_i32_saturates_instead_of_wrapping() {
    let a = [2_000_000_000i32, 2_000_000_000];
    let b = [1i32, 1, 1, -1];
    let bias = [7i32, 7];
    let mut c = [0i32; 2];

    unsafe {
        matmul_bias_kernel(
            a.as_ptr(),
            b.as_ptr(),
            bias.as_ptr(),
            c.as_mut_ptr(),
            1,
            2,
            2,
            2,
            2,
            2,
        )
    };
    assert_eq!(c, [i32::MAX, 7]);
}

/// Same accumulator defect through the GEMV-BT decode fast path, which has
/// its own dot-product loop and its own accumulator.
#[test]
fn test_gemv_bt_i32_saturates_instead_of_wrapping() {
    // B is stored as [N, K] = [[1, 1], [1, -1]].
    let a = [2_000_000_000i32, 2_000_000_000];
    let b_nk = [1i32, 1, 1, -1];
    let mut c = [0i32; 2];

    unsafe { gemv_bt_kernel(a.as_ptr(), b_nk.as_ptr(), c.as_mut_ptr(), 1, 2, 2, 2) };
    assert_eq!(c, [i32::MAX, 0]);
}

/// Catches an FP8 accumulator in the GEMV-BT dot product.
#[test]
fn test_gemv_bt_fp8_accumulates_in_f32() {
    use crate::dtype::FP8E4M3;

    let a = [FP8E4M3::from_f32(1.0); 32];
    let b_nk = [FP8E4M3::from_f32(1.0); 32];
    let mut c = [FP8E4M3::from_f32(0.0); 1];

    unsafe { gemv_bt_kernel(a.as_ptr(), b_nk.as_ptr(), c.as_mut_ptr(), 1, 1, 32, 1) };
    assert_eq!(c[0].to_f32(), 32.0);
}
