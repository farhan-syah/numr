//! Sparse matrix merge algorithms
//!
//! This module implements parameterized merge patterns for sparse matrix
//! element-wise operations, eliminating code duplication across CSR, CSC, and COO formats.

use super::super::CpuRuntime;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// Re-export zero_tolerance from shared utilities module
// See runtime::sparse_utils::zero_tolerance for full documentation
pub(crate) use crate::runtime::common::sparse_utils::zero_tolerance;

// =============================================================================
// Merge Strategy and Operation Semantics
// =============================================================================
//
// This section implements a parameterized merge pattern that eliminates
// code duplication across sparse matrix element-wise operations.
//
// # Design Pattern
//
// Instead of having separate implementations for add, subtract, multiply, and
// divide, we have:
// - ONE generic merge function per format (CSR, CSC, COO)
// - Strategy enum to control which positions to keep (union vs intersection)
// - Semantics enum to control empty matrix handling
// - Operation closures to define element-wise computation
//
// # Benefits
//
// 1. **Code Reuse**: Single merge implementation handles all operations
//    - Before: ~900 lines (6 specialized functions)
//    - After: ~300 lines (2 parameterized functions)
//
// 2. **Consistency**: All operations use identical merge logic, reducing bugs
//
// 3. **Extensibility**: New operations (e.g., max, min) require only:
//    - Adding semantics to enum
//    - Writing operation closure
//    - No new merge code needed
//
// 4. **Testability**: Testing merge logic once covers all operations
//
// # Architecture
//
// ```text
// User-facing trait method (add_csr, sub_csr, mul_csr, div_csr)
//       │
//       ├─> Calls merge_csr_impl with:
//       │   - MergeStrategy (Union or Intersection)
//       │   - OperationSemantics (Add, Subtract, Multiply, Divide)
//       │   - Operation closure: |a, b| a + b
//       │   - Transform closures: |a| a, |b| -b
//       │
//       └─> merge_csr_impl:
//           1. Check empty matrices (using semantics)
//           2. Merge based on strategy (union/intersection)
//           3. Apply operations via closures
//           4. Filter near-zeros
// ```
//
// # Example Usage
//
// ```ignore
// // Addition: union merge, identity transforms
// fn add_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Union,
//         OperationSemantics::Add,
//         |a, b| a + b,  // Both exist
//         |a| a,         // A-only: keep
//         |b| b,         // B-only: keep
//     )
// }
//
// // Subtraction: union merge, negate B
// fn sub_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Union,
//         OperationSemantics::Subtract,
//         |a, b| a - b,  // Both exist
//         |a| a,         // A-only: keep
//         |b| -b,        // B-only: negate
//     )
// }
//
// // Multiplication: intersection merge (0 * x = 0)
// fn mul_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Intersection,
//         OperationSemantics::Multiply,
//         |a, b| a * b,  // Both exist
//         |a| a,         // Unused (intersection only)
//         |b| b,         // Unused (intersection only)
//     )
// }
// ```
//
// # Empty Matrix Handling
//
// OperationSemantics controls what happens when one or both matrices are empty:
//
// | Operation | A empty, B not | A not, B empty | Both empty |
// |-----------|----------------|----------------|------------|
// | Add       | Return B       | Return A       | Empty      |
// | Subtract  | Return -B      | Return A       | Empty      |
// | Multiply  | Empty (0*x=0)  | Empty (x*0=0)  | Empty      |
// | Divide    | Empty (0/x=0)  | Error (x/0)    | Empty      |
//
// This logic is centralized in handle_empty_compressed() and avoids
// duplicating empty-check code across operations.

/// Strategy for merging two sparse matrices
///
/// Determines which positions to keep in the result:
/// - Union: Keep positions that exist in EITHER matrix (OR semantics)
/// - Intersection: Keep positions that exist in BOTH matrices (AND semantics)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum MergeStrategy {
    /// Union semantics: Keep all positions from both matrices
    ///
    /// Example: Addition, Subtraction
    /// - Position in A only: keep A's value
    /// - Position in B only: keep B's value (or -B for subtraction)
    /// - Position in both: apply operation
    Union,

    /// Intersection semantics: Keep only positions where both matrices have values
    ///
    /// Example: Multiplication, Division
    /// - Position in A only: skip (0 * B = 0)
    /// - Position in B only: skip (A * 0 = 0)
    /// - Position in both: apply operation
    Intersection,
}

/// Operation semantics for element-wise operations
///
/// Defines how to handle empty matrices and special cases
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum OperationSemantics {
    /// Addition: 0 + x = x, x + 0 = x
    Add,
    /// Subtraction: x - 0 = x, 0 - x = -x
    Subtract,
    /// Multiplication: 0 * x = 0, x * 0 = 0
    Multiply,
    /// Division: 0 / x = 0, x / 0 = error
    Divide,
}

/// Generic centralized empty matrix handling for CSR/CSC operations
///
/// Returns Some(result) if the operation can be short-circuited due to empty inputs,
/// None if both inputs are non-empty and normal processing should continue.
///
/// # Type Parameters
///
/// * `format_is_csr` - If true, uses row-based pointers (CSR), else column-based (CSC)
fn handle_empty_compressed<T: Element>(
    a_nnz: usize,
    b_nnz: usize,
    shape: [usize; 2],
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    a_ptrs: &Tensor<CpuRuntime>,
    a_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_ptrs: &Tensor<CpuRuntime>,
    b_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    semantics: OperationSemantics,
    format_is_csr: bool,
) -> Option<Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
    // CSR uses nrows, CSC uses ncols for pointer dimension
    let ptr_dim = if format_is_csr { shape[0] } else { shape[1] };

    let empty_result =
        || -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
            let empty_ptrs = Tensor::from_slice(&vec![0i64; ptr_dim + 1], &[ptr_dim + 1], device)?;
            let empty_indices = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
            let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
            Ok((empty_ptrs, empty_indices, empty_vals))
        };

    match (a_nnz, b_nnz, semantics) {
        // Both empty - always return empty
        (0, 0, _) => Some(empty_result()),

        // A empty, B not empty
        (0, _, OperationSemantics::Add) => {
            // 0 + B = B
            Some(Ok((b_ptrs.clone(), b_indices.clone(), b_values.clone())))
        }
        (0, _, OperationSemantics::Subtract) => {
            // 0 - B = -B
            let b_vals: Vec<T> = b_values.to_vec();
            let negated_vals: Vec<T> = b_vals.iter().map(|&v| T::from_f64(-v.to_f64())).collect();
            Some(
                Tensor::from_slice(&negated_vals, &[negated_vals.len()], device)
                    .map(|out_vals| (b_ptrs.clone(), b_indices.clone(), out_vals)),
            )
        }
        (0, _, OperationSemantics::Multiply) => {
            // 0 * B = 0
            Some(empty_result())
        }
        (0, _, OperationSemantics::Divide) => {
            // 0 / B = 0 (mathematically)
            Some(empty_result())
        }

        // A not empty, B empty
        (_, 0, OperationSemantics::Add) => {
            // A + 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Subtract) => {
            // A - 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Multiply) => {
            // A * 0 = 0
            Some(empty_result())
        }
        (_, 0, OperationSemantics::Divide) => {
            // A / 0 = undefined (division by zero matrix)
            Some(Err(Error::Internal(
                "Division by zero - B matrix is empty".to_string(),
            )))
        }

        // Both non-empty - continue with normal processing
        (_, _, _) => None,
    }
}

/// Generic CSR merge operation for element-wise ops (add, sub, mul, div)
///
/// This function implements the sorted-merge algorithm for combining two
/// CSR matrices element-wise. The operation is specified by the `op` function
/// and the merge strategy determines which positions to keep.
///
/// # Arguments
///
/// * `strategy` - Whether to use Union (keep all positions) or Intersection (keep only common positions)
/// * `semantics` - Operation semantics for handling empty matrices
/// * `op` - Operation to apply when both matrices have values at a position
/// * `only_a_op` - Transformation for values that only exist in A
/// * `only_b_op` - Transformation for values that only exist in B
///
/// # Algorithm
///
/// For each row:
/// 1. Merge the two sorted lists of column indices
/// 2. Union strategy: Keep positions from either matrix
/// 3. Intersection strategy: Keep only positions where both have values
/// 4. Apply operation when both matrices have values
pub(crate) fn merge_csr_impl<T: Element, F, FA, FB>(
    a_row_ptrs: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_ptrs: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
    strategy: MergeStrategy,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let [nrows, _ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_row_ptrs,
        a_col_indices,
        a_values,
        b_row_ptrs,
        b_col_indices,
        b_values,
        semantics,
        true, // CSR format
    ) {
        return result;
    }

    // Read CSR data
    let a_row_ptrs_data: Vec<i64> = a_row_ptrs.to_vec();
    let a_col_indices_data: Vec<i64> = a_col_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_row_ptrs_data: Vec<i64> = b_row_ptrs.to_vec();
    let b_col_indices_data: Vec<i64> = b_col_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSR
    let mut out_row_ptrs: Vec<i64> = Vec::with_capacity(nrows + 1);
    let mut out_col_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_row_ptrs.push(0);

    for row in 0..nrows {
        let a_start = a_row_ptrs_data[row] as usize;
        let a_end = a_row_ptrs_data[row + 1] as usize;
        let b_start = b_row_ptrs_data[row] as usize;
        let b_end = b_row_ptrs_data[row + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_col = if i < a_end {
                        a_col_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_col = if j < b_end {
                        b_col_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_col < b_col {
                        // Only A has value at this column - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value at this column - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(b_col);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            MergeStrategy::Intersection => {
                // Intersection: Keep only positions where both have values (&& semantics)
                while i < a_end && j < b_end {
                    let a_col = a_col_indices_data[i];
                    let b_col = b_col_indices_data[j];

                    if a_col < b_col {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_row_ptrs.push(out_col_indices.len() as i64);
    }

    // Create result tensors
    let result_row_ptrs = Tensor::from_slice(&out_row_ptrs, &[nrows + 1], device)?;
    let result_col_indices =
        Tensor::from_slice(&out_col_indices, &[out_col_indices.len()], device)?;
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device)?;

    Ok((result_row_ptrs, result_col_indices, result_values))
}

/// Generic merge implementation for CSC element-wise operations
///
/// Merges two CSC matrices element-wise. The operation is specified by the `op` function.
///
/// # Algorithm
///
/// For each column:
/// 1. Merge the two sorted lists of row indices
/// 2. Apply operation when both matrices have a value at that row
/// 3. Keep values from only one matrix when the other is zero
pub(crate) fn merge_csc_impl<T: Element, F, FA, FB>(
    a_col_ptrs: &Tensor<CpuRuntime>,
    a_row_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_col_ptrs: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
    strategy: MergeStrategy,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let [_nrows, ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_col_ptrs,
        a_row_indices,
        a_values,
        b_col_ptrs,
        b_row_indices,
        b_values,
        semantics,
        false, // CSC format
    ) {
        return result;
    }

    // Read CSC data
    let a_col_ptrs_data: Vec<i64> = a_col_ptrs.to_vec();
    let a_row_indices_data: Vec<i64> = a_row_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_col_ptrs_data: Vec<i64> = b_col_ptrs.to_vec();
    let b_row_indices_data: Vec<i64> = b_row_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSC
    let mut out_col_ptrs: Vec<i64> = Vec::with_capacity(ncols + 1);
    let mut out_row_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_col_ptrs.push(0);

    for col in 0..ncols {
        let a_start = a_col_ptrs_data[col] as usize;
        let a_end = a_col_ptrs_data[col + 1] as usize;
        let b_start = b_col_ptrs_data[col] as usize;
        let b_end = b_col_ptrs_data[col + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_row = if i < a_end {
                        a_row_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_row = if j < b_end {
                        b_row_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_row < b_row {
                        // Only A has value at this row - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value at this row - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(b_row);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            MergeStrategy::Intersection => {
                // Intersection: Keep only positions where both have values (&& semantics)
                while i < a_end && j < b_end {
                    let a_row = a_row_indices_data[i];
                    let b_row = b_row_indices_data[j];

                    if a_row < b_row {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_col_ptrs.push(out_row_indices.len() as i64);
    }

    // Create result tensors
    let result_col_ptrs = Tensor::from_slice(&out_col_ptrs, &[ncols + 1], device)?;
    let result_row_indices =
        Tensor::from_slice(&out_row_indices, &[out_row_indices.len()], device)?;
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device)?;

    Ok((result_col_ptrs, result_row_indices, result_values))
}

/// Generic merge implementation for COO element-wise operations
///
/// Merges two COO matrices element-wise with full control over value transformations.
///
/// # Arguments
///
/// * `op` - Operation to apply when both matrices have a value at the same position
/// * `only_a_op` - Transformation for values that only exist in A (identity for add, identity for sub)
/// * `only_b_op` - Transformation for values that only exist in B (identity for add, negate for sub)
///
/// # Algorithm
///
/// 1. Concatenate both matrices' triplets (applying transformations)
/// 2. Sort by (row, col)
/// 3. Merge duplicate positions by applying the operation
pub(crate) fn merge_coo_impl<T: Element, F, FA, FB>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    _semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs early to avoid bytemuck alignment errors
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 && b_nnz == 0 {
        // Both empty - return empty result (always empty regardless of semantics)
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    } else if a_nnz == 0 {
        // A is empty, B is not - apply B transformation
        let b_vals: Vec<T> = b_values.to_vec();
        let transformed_vals: Vec<T> = b_vals.iter().map(|&v| only_b_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device)?;
        return Ok((b_row_indices.clone(), b_col_indices.clone(), out_vals));
    } else if b_nnz == 0 {
        // B is empty, A is not - apply A transformation (usually identity)
        let a_vals: Vec<T> = a_values.to_vec();
        let transformed_vals: Vec<T> = a_vals.iter().map(|&v| only_a_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device)?;
        return Ok((a_row_indices.clone(), a_col_indices.clone(), out_vals));
    }

    // Both non-empty - proceed with merge
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Concatenate triplets with source tag (true=from A, false=from B)
    let mut triplets: Vec<(i64, i64, T, bool)> = Vec::new();
    for i in 0..a_rows.len() {
        triplets.push((a_rows[i], a_cols[i], a_vals[i], true));
    }
    for i in 0..b_rows.len() {
        triplets.push((b_rows[i], b_cols[i], b_vals[i], false));
    }

    // Sort by (row, col)
    triplets.sort_by_key(|&(r, c, _, _)| (r, c));

    // Merge duplicates
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    if triplets.is_empty() {
        // Empty result
        let empty_rows = Tensor::from_slice(&result_rows, &[0], device)?;
        let empty_cols = Tensor::from_slice(&result_cols, &[0], device)?;
        let empty_vals = Tensor::from_slice(&result_vals, &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    let mut current_row = triplets[0].0;
    let mut current_col = triplets[0].1;
    let mut current_val = triplets[0].2;
    let mut current_from_a = triplets[0].3;
    let mut current_merged = false; // Track if this value resulted from merging

    for i in 1..triplets.len() {
        let (row, col, val, from_a) = triplets[i];

        if row == current_row && col == current_col {
            // Same position - apply operation (merge)
            current_val = op(current_val, val);
            current_merged = true; // This value is now a merged result
        } else {
            // Different position - save current (with transformation if not merged)
            let final_val = if current_merged {
                // Already merged, no transformation needed
                current_val
            } else {
                // Not merged, apply transformation based on source
                if current_from_a {
                    only_a_op(current_val)
                } else {
                    only_b_op(current_val)
                }
            };

            if final_val.to_f64().abs() > zero_tolerance::<T>() {
                result_rows.push(current_row);
                result_cols.push(current_col);
                result_vals.push(final_val);
            }

            // Start new accumulation
            current_row = row;
            current_col = col;
            current_val = val;
            current_from_a = from_a;
            current_merged = false;
        }
    }

    // Don't forget the last triplet
    let final_val = if current_merged {
        // Already merged, no transformation needed
        current_val
    } else {
        // Not merged, apply transformation based on source
        if current_from_a {
            only_a_op(current_val)
        } else {
            only_b_op(current_val)
        }
    };

    if final_val.to_f64().abs() > zero_tolerance::<T>() {
        result_rows.push(current_row);
        result_cols.push(current_col);
        result_vals.push(final_val);
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device)?;
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device)?;
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device)?;

    Ok((out_rows, out_cols, out_vals))
}

/// Intersection-based merge for COO element-wise operations
///
/// Only keeps positions where BOTH matrices have non-zero values.
/// This is the correct semantics for sparse element-wise multiplication and division.
///
/// Uses a merge-join algorithm on sorted triplets for O(nnz_a + nnz_b) complexity.
pub(crate) fn intersect_coo_impl<T: Element, F>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    op: F,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs - multiplication by zero gives zero (empty)
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 || b_nnz == 0 {
        // If either is empty, result is empty
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device)?;
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device)?;
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    // Read COO data
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Sort both by (row, col) if not already sorted
    let mut a_triplets: Vec<(i64, i64, T)> = a_rows
        .iter()
        .zip(a_cols.iter())
        .zip(a_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    a_triplets.sort_by_key(|&(r, c, _)| (r, c));

    let mut b_triplets: Vec<(i64, i64, T)> = b_rows
        .iter()
        .zip(b_cols.iter())
        .zip(b_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    b_triplets.sort_by_key(|&(r, c, _)| (r, c));

    // Merge-join to find intersection
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    let mut i = 0;
    let mut j = 0;

    while i < a_triplets.len() && j < b_triplets.len() {
        let (a_row, a_col, a_val) = a_triplets[i];
        let (b_row, b_col, b_val) = b_triplets[j];

        match (a_row.cmp(&b_row), a_col.cmp(&b_col)) {
            (std::cmp::Ordering::Less, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Less) => {
                // A's position is before B's - skip it (not in intersection)
                i += 1;
            }
            (std::cmp::Ordering::Greater, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Greater) => {
                // B's position is before A's - skip it (not in intersection)
                j += 1;
            }
            (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                // Same position - apply operation and keep result
                let result = op(a_val, b_val);
                if result.to_f64().abs() > zero_tolerance::<T>() {
                    result_rows.push(a_row);
                    result_cols.push(a_col);
                    result_vals.push(result);
                }
                i += 1;
                j += 1;
            }
        }
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device)?;
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device)?;
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device)?;

    Ok((out_rows, out_cols, out_vals))
}
