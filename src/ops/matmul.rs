//! Matrix multiplication helpers
//!
//! This module contains helper types and functions for matrix multiplication.
//! The actual operations are defined in the `TensorOps` trait.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Matrix multiplication parameters
#[derive(Copy, Clone, Debug)]
pub struct MatmulParams {
    /// Number of rows in A (M)
    pub m: usize,
    /// Number of columns in A / rows in B (K)
    pub k: usize,
    /// Number of columns in B (N)
    pub n: usize,
    /// Whether A is transposed
    pub trans_a: bool,
    /// Whether B is transposed
    pub trans_b: bool,
    /// Batch size (1 for regular matmul)
    pub batch: usize,
}

impl MatmulParams {
    /// Create params for standard matmul: C`[M,N]` = A`[M,K]` @ B`[K,N]`
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self {
            m,
            k,
            n,
            trans_a: false,
            trans_b: false,
            batch: 1,
        }
    }

    /// Create params for batched matmul
    pub fn batched(batch: usize, m: usize, k: usize, n: usize) -> Self {
        Self {
            m,
            k,
            n,
            trans_a: false,
            trans_b: false,
            batch,
        }
    }

    /// Set A transposition
    pub fn with_trans_a(mut self, trans: bool) -> Self {
        self.trans_a = trans;
        self
    }

    /// Set B transposition
    pub fn with_trans_b(mut self, trans: bool) -> Self {
        self.trans_b = trans;
        self
    }

    /// Output shape
    pub fn output_shape(&self) -> Vec<usize> {
        if self.batch > 1 {
            vec![self.batch, self.m, self.n]
        } else {
            vec![self.m, self.n]
        }
    }
}

/// Validate matmul shapes and return dimensions (m, k, n)
///
/// Returns None if shapes are incompatible.
pub fn validate_matmul_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Option<(usize, usize, usize)> {
    // Handle 1D vectors
    let (a_rows, a_cols) = match a_shape.len() {
        0 => return None,
        1 => (1, a_shape[0]),
        _ => {
            let ndim = a_shape.len();
            (a_shape[ndim - 2], a_shape[ndim - 1])
        }
    };

    let (b_rows, b_cols) = match b_shape.len() {
        0 => return None,
        1 => (b_shape[0], 1),
        _ => {
            let ndim = b_shape.len();
            (b_shape[ndim - 2], b_shape[ndim - 1])
        }
    };

    // Inner dimensions must match
    if a_cols != b_rows {
        return None;
    }

    Some((a_rows, a_cols, b_cols))
}

/// Compute output shape for matmul
pub fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Option<Vec<usize>> {
    let (m, _k, n) = validate_matmul_shapes(a_shape, b_shape)?;

    // Handle batched matmul
    let a_batch: Vec<_> = a_shape
        .iter()
        .take(a_shape.len().saturating_sub(2))
        .copied()
        .collect();
    let b_batch: Vec<_> = b_shape
        .iter()
        .take(b_shape.len().saturating_sub(2))
        .copied()
        .collect();

    // Broadcast batch dimensions
    let batch = super::broadcast_shape(&a_batch, &b_batch)?;

    let mut result = batch;
    result.push(m);
    result.push(n);
    Some(result)
}

/// Map each output batch to the batch index each operand should read.
///
/// Batch dims broadcast per dimension, so an operand's batch count is not enough
/// to locate its data: with `A[2, 4, m, k] @ B[2, 1, k, n]` the output has 8
/// batches while `B` has 2, and `B`'s index advances only every 4 outputs.
/// Treating an operand as "broadcast everything" or "same batch as output" reads
/// out of bounds for any case in between.
///
/// Returns `(a_indices, b_indices)`, both of length `prod(out_shape[..-2])`.
pub fn matmul_batch_indices(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let out_batch = &out_shape[..out_shape.len().saturating_sub(2)];
    let a_batch = &a_shape[..a_shape.len().saturating_sub(2)];
    let b_batch = &b_shape[..b_shape.len().saturating_sub(2)];

    let total: usize = out_batch.iter().product::<usize>().max(1);
    let mut a_indices = Vec::with_capacity(total);
    let mut b_indices = Vec::with_capacity(total);
    let mut coord = vec![0usize; out_batch.len()];

    // Operand batch dims are right-aligned against the output's, and a size-1 dim
    // holds index 0 while the output coordinate advances.
    let project = |coord: &[usize], batch: &[usize]| -> usize {
        let offset = coord.len() - batch.len();
        let mut idx = 0;
        for (d, &size) in batch.iter().enumerate() {
            let c = if size == 1 { 0 } else { coord[offset + d] };
            idx = idx * size + c;
        }
        idx
    };

    for flat in 0..total {
        let mut rem = flat;
        for d in (0..out_batch.len()).rev() {
            coord[d] = rem % out_batch[d];
            rem /= out_batch[d];
        }
        a_indices.push(project(&coord, a_batch));
        b_indices.push(project(&coord, b_batch));
    }

    (a_indices, b_indices)
}

/// Is `B` a plain transpose of a contiguous `[.., N, K]` buffer?
///
/// A `[K, N]` operand with strides `[1, K]` is the transposed view of a
/// contiguous `[N, K]` weight matrix — the layout every `Linear` weight has.
/// Backends that can read that layout directly skip materializing the view,
/// which otherwise copies the whole weight matrix on every call.
///
/// Returns true only when the underlying buffer is densely packed as
/// `[.., N, K]`, so a batch `i` of the operand starts at element `i * N * K`:
/// - the last two strides are exactly `[1, K]` (this rejects stride 0 and every
///   negative stride, neither of which is a simple transpose)
/// - every batch dim of size > 1 carries the dense stride for that layout
///   (a size-1 dim always projects to index 0, so its stride is irrelevant)
/// - `K` and `N` are both non-zero, so the stride pattern is unambiguous
pub fn is_transposed_b(b_shape: &[usize], b_strides: &[isize], k: usize, n: usize) -> bool {
    let ndim = b_shape.len();
    if ndim < 2 || b_strides.len() != ndim || k == 0 || n == 0 {
        return false;
    }
    if b_strides[ndim - 2] != 1 || b_strides[ndim - 1] != k as isize {
        return false;
    }

    // Batch dims, right to left: dense packing of `[.., N, K]`.
    let mut expected = (n * k) as isize;
    for d in (0..ndim - 2).rev() {
        if b_shape[d] != 1 && b_strides[d] != expected {
            return false;
        }
        expected *= b_shape[d] as isize;
    }
    true
}

/// Validate matmul_bias shapes and return dimensions (m, k, n)
///
/// Checks that:
/// - A and B are compatible for matmul (inner dimensions match)
/// - bias is 1D
/// - bias length matches output columns (N)
///
/// Returns None if shapes are incompatible.
pub fn validate_matmul_bias_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
    bias_shape: &[usize],
) -> Option<(usize, usize, usize)> {
    // First validate matmul shapes
    let (m, k, n) = validate_matmul_shapes(a_shape, b_shape)?;

    // Bias must be 1D
    if bias_shape.len() != 1 {
        return None;
    }

    // Bias length must match output columns (N)
    if bias_shape[0] != n {
        return None;
    }

    Some((m, k, n))
}

/// Compute output shape for matmul_bias
///
/// Same as matmul_output_shape - bias doesn't change output dimensions.
pub fn matmul_bias_output_shape(
    a_shape: &[usize],
    b_shape: &[usize],
    bias_shape: &[usize],
) -> Option<Vec<usize>> {
    // Validate bias shape
    validate_matmul_bias_shapes(a_shape, b_shape, bias_shape)?;

    // Output shape is same as matmul
    matmul_output_shape(a_shape, b_shape)
}

/// Validate that all three tensors (A, B, bias) have the same dtype for matmul_bias.
///
/// This is the **canonical** dtype validation for matmul_bias - use this function
/// in all backend implementations to ensure consistent error handling.
///
/// # Returns
/// - `Ok(dtype)` if all dtypes match
/// - `Err(DTypeMismatch)` if any dtypes differ
pub fn validate_matmul_bias_dtypes(
    a_dtype: DType,
    b_dtype: DType,
    bias_dtype: DType,
) -> Result<DType> {
    if a_dtype != b_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: b_dtype,
        });
    }
    if a_dtype != bias_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: bias_dtype,
        });
    }
    Ok(a_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A middle batch dim broadcasting under a leading batch > 1: B's index must
    /// advance once every 4 outputs, which a batch count alone cannot express.
    #[test]
    fn test_matmul_batch_indices_middle_broadcast() {
        let (a, b) = matmul_batch_indices(&[2, 4, 3, 2], &[2, 1, 2, 1], &[2, 4, 3, 1]);
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }

    /// A leading broadcast dim does cycle, which is why wrapping by a count
    /// appeared to work.
    #[test]
    fn test_matmul_batch_indices_leading_broadcast() {
        let (a, b) = matmul_batch_indices(&[2, 4, 3, 2], &[1, 4, 2, 3], &[2, 4, 3, 3]);
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_matmul_batch_indices_fewer_batch_dims() {
        // B's batch dims are right-aligned against the output's.
        let (a, b) = matmul_batch_indices(&[2, 4, 3, 2], &[4, 2, 3], &[2, 4, 3, 3]);
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(b, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_matmul_batch_indices_both_broadcast() {
        let (a, b) = matmul_batch_indices(&[2, 1, 3, 2], &[1, 4, 2, 3], &[2, 4, 3, 3]);
        assert_eq!(a, vec![0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(b, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn test_matmul_batch_indices_unbatched() {
        let (a, b) = matmul_batch_indices(&[3, 2], &[2, 4], &[3, 4]);
        assert_eq!(a, vec![0]);
        assert_eq!(b, vec![0]);
    }

    #[test]
    fn test_validate_matmul_shapes() {
        // Valid 2D matmul
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3, 4]), Some((2, 3, 4)));

        // Invalid: inner dimensions don't match
        assert_eq!(validate_matmul_shapes(&[2, 3], &[4, 5]), None);

        // 1D vectors
        assert_eq!(validate_matmul_shapes(&[3], &[3, 4]), Some((1, 3, 4)));
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3]), Some((2, 3, 1)));
    }

    #[test]
    fn test_matmul_output_shape() {
        // Basic 2D matmul
        assert_eq!(matmul_output_shape(&[2, 3], &[3, 4]), Some(vec![2, 4]));

        // Batched matmul
        assert_eq!(
            matmul_output_shape(&[5, 2, 3], &[5, 3, 4]),
            Some(vec![5, 2, 4])
        );

        // Broadcast batches
        assert_eq!(
            matmul_output_shape(&[5, 2, 3], &[3, 4]),
            Some(vec![5, 2, 4])
        );
    }

    #[test]
    fn test_validate_matmul_bias_shapes() {
        // Valid: 2D matmul with 1D bias
        assert_eq!(
            validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[4]),
            Some((2, 3, 4))
        );

        // Invalid: inner dimensions don't match
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[4, 5], &[5]), None);

        // Invalid: bias is 2D
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[2, 4]), None);

        // Invalid: bias length doesn't match N
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[3]), None);

        // Valid: batched matmul with 1D bias
        assert_eq!(
            validate_matmul_bias_shapes(&[5, 2, 3], &[5, 3, 4], &[4]),
            Some((2, 3, 4))
        );
    }

    #[test]
    fn test_matmul_bias_output_shape() {
        // Basic 2D matmul_bias
        assert_eq!(
            matmul_bias_output_shape(&[2, 3], &[3, 4], &[4]),
            Some(vec![2, 4])
        );

        // Batched matmul_bias
        assert_eq!(
            matmul_bias_output_shape(&[5, 2, 3], &[5, 3, 4], &[4]),
            Some(vec![5, 2, 4])
        );

        // Invalid bias shape returns None
        assert_eq!(matmul_bias_output_shape(&[2, 3], &[3, 4], &[3]), None);
    }

    #[test]
    fn test_validate_matmul_bias_dtypes() {
        // All same dtype - should succeed
        assert!(validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).is_ok());
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F64, DType::F64, DType::F64).unwrap(),
            DType::F64
        );

        // A and B mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F64, DType::F32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::F64);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }

        // A and bias mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::I32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::I32);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }
    }

    #[test]
    fn test_is_transposed_b_simple_transpose() {
        // [4, 3] view of a contiguous [3, 4] buffer.
        assert!(is_transposed_b(&[4, 3], &[1, 4], 4, 3));
    }

    #[test]
    fn test_is_transposed_b_rejects_contiguous() {
        // Row-major [4, 3] is not a transpose (unless K == 1, where both agree).
        assert!(!is_transposed_b(&[4, 3], &[3, 1], 4, 3));
    }

    #[test]
    fn test_is_transposed_b_rejects_zero_and_negative_strides() {
        assert!(!is_transposed_b(&[4, 3], &[0, 4], 4, 3));
        assert!(!is_transposed_b(&[4, 3], &[1, -4], 4, 3));
        assert!(!is_transposed_b(&[4, 3], &[-1, 4], 4, 3));
    }

    #[test]
    fn test_is_transposed_b_rejects_degenerate_dims() {
        assert!(!is_transposed_b(&[0, 3], &[1, 0], 0, 3));
        assert!(!is_transposed_b(&[4, 0], &[1, 4], 4, 0));
        assert!(!is_transposed_b(&[3], &[1], 3, 1));
    }

    #[test]
    fn test_is_transposed_b_batched_dense() {
        // [2, 4, 3] view of a contiguous [2, 3, 4] buffer: batch stride 12.
        assert!(is_transposed_b(&[2, 4, 3], &[12, 1, 4], 4, 3));
        assert!(is_transposed_b(&[5, 2, 4, 3], &[24, 12, 1, 4], 4, 3));
    }

    #[test]
    fn test_is_transposed_b_rejects_sliced_batch() {
        // Batch dim striding over a larger buffer: batch i does NOT start at
        // i * N * K, so the flat batch offset would read the wrong matrix.
        assert!(!is_transposed_b(&[2, 4, 3], &[24, 1, 4], 4, 3));
    }

    #[test]
    fn test_is_transposed_b_ignores_size_one_batch_stride() {
        // A size-1 batch dim always projects to index 0.
        assert!(is_transposed_b(&[1, 4, 3], &[999, 1, 4], 4, 3));
    }
}
