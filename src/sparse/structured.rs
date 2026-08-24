//! 2:4 Structured sparsity format
//!
//! NVIDIA Ampere+ format where exactly 2 of every 4 consecutive elements are zero,
//! enabling 2x GEMM throughput via sparse tensor cores.
//!
//! The compressed representation stores only the 2 non-zero values per group of 4,
//! plus 2-bit metadata indicating which positions were kept.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// 2:4 structured sparse tensor
///
/// Stores a matrix in compressed 2:4 format where exactly 2 out of every 4
/// consecutive elements along the K dimension are non-zero.
///
/// # Layout
///
/// For an `[M, K]` dense matrix:
/// - `compressed_values`: `[M, K/2]` — the 2 kept values per group of 4
/// - `metadata`: `[M, K/16]` as U32 — 2-bit indices packed into 32-bit words
///   (each U32 holds metadata for 16 groups of 4 = 64 elements)
///
/// # Metadata encoding
///
/// For each group of 4 elements, 2 bits encode which 2 of 4 positions are kept.
/// There are C(4,2) = 6 valid patterns, encoded as:
///   - 0b00: positions 0,1
///   - 0b01: positions 0,2
///   - 0b10: positions 0,3
///   - 0b11: positions 1,2
///   - 0b100: positions 1,3  (but we only use 2 bits, so we need a different encoding)
///
/// Actually, NVIDIA uses a different encoding: each group stores a 4-bit mask where
/// exactly 2 bits are set, indicating which positions are kept. We pack 8 such masks
/// per U32 (8 × 4 bits = 32 bits), so metadata shape is `[M, ceil(K/4/8)]` = `[M, K/32]`.
///
/// Revised: We use 4 bits per group (bitmask with exactly 2 bits set).
/// 8 groups per U32 → metadata shape `[M, K/32]` (since K/4 groups, 8 groups per U32).
/// If K is not divisible by 32, the last U32 is partially used.
#[derive(Debug, Clone)]
pub struct Sparse24Tensor<R: Runtime> {
    /// Compressed non-zero values, shape [M, K/2]
    pub(crate) compressed_values: Tensor<R>,
    /// Packed metadata bitmasks, shape [M, ceil(K/4 / 8)] as U32
    /// Each U32 contains 8 groups × 4 bits = 32 bits
    pub(crate) metadata: Tensor<R>,
    /// Original dense shape [M, K]
    pub(crate) original_shape: [usize; 2],
    /// Data type of the compressed values
    pub(crate) dtype: DType,
}

impl<R: Runtime<DType = DType>> Sparse24Tensor<R> {
    /// Create a Sparse24Tensor from pre-built components
    ///
    /// # Arguments
    /// * `compressed_values` - Shape [M, K/2], the non-zero values
    /// * `metadata` - Shape [M, meta_cols] as U32, packed bitmasks
    /// * `original_shape` - The original dense shape [M, K]
    pub fn new(
        compressed_values: Tensor<R>,
        metadata: Tensor<R>,
        original_shape: [usize; 2],
    ) -> Result<Self> {
        let [m, k] = original_shape;

        // K must be divisible by 4
        if k % 4 != 0 {
            return Err(Error::InvalidArgument {
                arg: "original_shape",
                reason: format!("K dimension ({k}) must be divisible by 4 for 2:4 sparsity"),
            });
        }

        // Validate compressed_values shape
        let expected_val_shape = [m, k / 2];
        if compressed_values.shape() != expected_val_shape {
            return Err(Error::ShapeMismatch {
                expected: expected_val_shape.to_vec(),
                got: compressed_values.shape().to_vec(),
            });
        }

        // Validate metadata shape
        let num_groups = k / 4;
        let meta_cols = (num_groups + 7) / 8; // ceil(num_groups / 8)
        let expected_meta_shape = [m, meta_cols];
        if metadata.shape() != expected_meta_shape {
            return Err(Error::ShapeMismatch {
                expected: expected_meta_shape.to_vec(),
                got: metadata.shape().to_vec(),
            });
        }

        // Metadata must be U32
        if metadata.dtype() != DType::U32 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U32,
                rhs: metadata.dtype(),
            });
        }

        let dtype = compressed_values.dtype();

        Ok(Self {
            compressed_values,
            metadata,
            original_shape,
            dtype,
        })
    }

    /// Returns the original dense shape [M, K]
    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        self.original_shape
    }

    /// Returns M (number of rows)
    #[inline]
    pub fn nrows(&self) -> usize {
        self.original_shape[0]
    }

    /// Returns K (original number of columns)
    #[inline]
    pub fn ncols(&self) -> usize {
        self.original_shape[1]
    }

    /// Returns the data type
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns a reference to the compressed values tensor [M, K/2]
    #[inline]
    pub fn compressed_values(&self) -> &Tensor<R> {
        &self.compressed_values
    }

    /// Returns a reference to the metadata tensor [M, meta_cols] as U32
    #[inline]
    pub fn metadata(&self) -> &Tensor<R> {
        &self.metadata
    }

    /// Returns the number of non-zero elements (always M * K/2)
    #[inline]
    pub fn nnz(&self) -> usize {
        self.original_shape[0] * (self.original_shape[1] / 2)
    }

    /// Returns the compression ratio (always 2.0 for 2:4)
    #[inline]
    pub fn compression_ratio(&self) -> f64 {
        2.0
    }

    /// Number of groups of 4 per row
    #[inline]
    pub fn groups_per_row(&self) -> usize {
        self.original_shape[1] / 4
    }

    /// Number of U32 metadata words per row
    #[inline]
    pub fn meta_cols(&self) -> usize {
        (self.groups_per_row() + 7) / 8
    }

    /// Validate that the 2:4 structure is correct:
    /// each metadata group has exactly 2 bits set in its 4-bit nibble
    pub fn is_valid(&self) -> bool
    where
        R: Runtime<DType = DType>,
    {
        let meta_data: Vec<u32> = self.metadata.to_vec();
        let num_groups = self.groups_per_row();

        for row in 0..self.nrows() {
            for g in 0..num_groups {
                let word_idx = g / 8;
                let nibble_idx = g % 8;
                let word = meta_data[row * self.meta_cols() + word_idx];
                let nibble = (word >> (nibble_idx * 4)) & 0xF;
                if nibble.count_ones() != 2 {
                    return false;
                }
            }
        }
        true
    }
}

/// Compute the metadata column count for a given K dimension
#[inline]
pub fn meta_cols_for_k(k: usize) -> usize {
    let num_groups = k / 4;
    (num_groups + 7) / 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_cols_for_k() {
        assert_eq!(meta_cols_for_k(4), 1); // 1 group, 1 word
        assert_eq!(meta_cols_for_k(8), 1); // 2 groups, 1 word
        assert_eq!(meta_cols_for_k(32), 1); // 8 groups, 1 word
        assert_eq!(meta_cols_for_k(36), 2); // 9 groups, 2 words
        assert_eq!(meta_cols_for_k(64), 2); // 16 groups, 2 words
    }

    #[test]
    fn test_k_must_be_divisible_by_4() {
        use crate::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();

        // K=5 should fail
        let vals = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap();
        let meta = Tensor::<CpuRuntime>::try_from_slice(&[0u32], &[1, 1], &device).unwrap();
        let result = Sparse24Tensor::new(vals, meta, [1, 5]);
        assert!(result.is_err());
    }
}
