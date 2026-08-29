//! i8 × i8 → i32 matrix multiplication using SIMD dot product kernels
//!
//! Each output element C[i][j] = sum_k(A[i][k] * B[k][j]) where A,B are i8
//! and accumulation is in i32. Uses the SIMD dot product from `simd::dot`.

use super::super::dot::{i8xi8_dot_i32, saturate_i64_to_i32};

/// i8 × i8 → i32 matmul: C[m×n] = A[m×k] @ B[k×n]
///
/// Packs columns of B into a contiguous scratch buffer so each dot product
/// operates on contiguous memory.
///
/// # Safety
/// - `a` must be valid for m*lda i8 elements
/// - `b` must be valid for k*ldb i8 elements
/// - `out` must be valid for m*ldc i32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_to_i32(
    a: *const i8,
    b: *const i8,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Pack column j of B into contiguous memory for efficient dot products
    let mut b_col = vec![0i8; k];

    for j in 0..n {
        // Pack column j
        for kk in 0..k {
            *b_col.as_mut_ptr().add(kk) = *b.add(kk * ldb + j);
        }

        // Compute dot product of each row of A with packed column
        for i in 0..m {
            let a_row = a.add(i * lda);
            *out.add(i * ldc + j) = i8xi8_dot_i32(a_row, b_col.as_ptr(), k);
        }
    }
}

/// i8 × i8 → i32 matmul with a fused i32 bias: C[m×n] = A[m×k] @ B[k×n] + bias
///
/// The bias joins the wide sum before the single narrowing store, so it is part
/// of the accumulation rather than a value added to an already-narrowed result.
/// This is what the CUDA `I8ToI32Acc` policy does, and what keeps the two
/// backends reporting the same number at the boundary.
///
/// The dot product itself clamps at i32 internally, which can only differ from
/// a fully exact seed past 2^31 - about 2^17 terms at full i8 magnitude - and
/// the same clamp fires on CUDA there.
///
/// # Safety
/// - `a` must be valid for m*lda i8 elements
/// - `b` must be valid for k*ldb i8 elements
/// - `bias` must be valid for n i32 elements
/// - `out` must be valid for m*ldc i32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_to_i32_bias(
    a: *const i8,
    b: *const i8,
    bias: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Same column packing as the plain form: one contiguous copy of column j
    // feeds every row's dot product.
    let mut b_col = vec![0i8; k];

    for j in 0..n {
        for kk in 0..k {
            *b_col.as_mut_ptr().add(kk) = *b.add(kk * ldb + j);
        }

        let bias_j = *bias.add(j) as i64;
        for i in 0..m {
            let a_row = a.add(i * lda);
            let acc = bias_j + i8xi8_dot_i32(a_row, b_col.as_ptr(), k) as i64;
            *out.add(i * ldc + j) = saturate_i64_to_i32(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_i8_to_i32_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let mut c = [0i32; 4];

        unsafe {
            matmul_i8_to_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2);
        }
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c, [19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_i8_to_i32_negative() {
        let a: Vec<i8> = vec![-1, 2, 3, -4];
        let b: Vec<i8> = vec![5, -6, -7, 8];
        let mut c = [0i32; 4];

        unsafe {
            matmul_i8_to_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2);
        }
        // [[-1,2],[3,-4]] @ [[5,-6],[-7,8]] = [[-19,22],[43,-50]]
        assert_eq!(c, [-19, 22, 43, -50]);
    }

    /// The bias reaches the widened output: 12 * 11 = 132 with a bias of 5 is
    /// 137, a value I8 could not hold at either step.
    #[test]
    fn test_matmul_i8_to_i32_bias_past_i8_range() {
        let a: Vec<i8> = vec![12];
        let b: Vec<i8> = vec![11];
        let bias = [5i32];
        let mut c = [0i32; 1];

        unsafe {
            matmul_i8_to_i32_bias(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                1,
                1,
                1,
                1,
                1,
                1,
            );
        }
        assert_eq!(c, [137]);
    }

    #[test]
    fn test_matmul_i8_to_i32_bias_per_column() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let bias = [1i32, -2];
        let mut c = [0i32; 4];

        unsafe {
            matmul_i8_to_i32_bias(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
            );
        }
        // [[19,22],[43,50]] + [1,-2]
        assert_eq!(c, [20, 20, 44, 48]);
    }

    #[test]
    fn test_matmul_i8_to_i32_wide() {
        // Test with larger k to exercise SIMD dot product path
        let (m, n, k) = (2, 3, 64);
        let a: Vec<i8> = (0..m * k)
            .map(|i| ((i % 127) as i8).wrapping_sub(64))
            .collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 3 % 127) as i8).wrapping_sub(64))
            .collect();
        let mut c = vec![0i32; m * n];

        unsafe {
            matmul_i8_to_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n);
        }

        // Reference
        let mut expected = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += a[i * k + kk] as i32 * b[kk * n + j] as i32;
                }
            }
        }
        assert_eq!(c, expected);
    }
}
