//! Shared reduction epilogue for CUDA: sum over dims, then divide by a count.
//!
//! `mean` and `var` both end in "sum the elements, divide by how many there
//! were". Built naively out of `sum` + `div_scalar` that epilogue overflows
//! twice on a narrow float dtype:
//!
//! 1. `sum` accumulates in F32 but writes its result back in the tensor's own
//!    dtype, so a total above the dtype's largest finite value saturates to
//!    infinity (F16 tops out at 65504).
//! 2. `div_scalar` narrows the scalar to the tensor's dtype, so any element
//!    count above that same limit becomes infinity and every result becomes 0.
//!
//! CPU defines the intended semantics: it divides inside its wide accumulator
//! and narrows exactly once at write-out. This module reproduces that on CUDA
//! by promoting a narrow float to F32, doing both the sum and the division
//! there, and demoting once at the end.
//!
//! `div_scalar` itself is left alone — narrowing the scalar is correct for a
//! general elementwise-by-scalar op. The defect was building a reduction
//! epilogue out of it.

use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::error::Result;
use crate::ops::{ReduceOps, ScalarOps};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Sum `a` over `dims`, then divide by `divisor`.
///
/// Narrow float dtypes (F16, BF16, FP8) are promoted to F32 for both the sum
/// and the division, then demoted once. F32, F64, and non-float dtypes take
/// the direct `sum` + `div_scalar` path unchanged — no promote/demote round
/// trip, and integer means keep their integer division semantics.
pub(crate) fn sum_then_divide(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    divisor: f64,
) -> Result<Tensor<CudaRuntime>> {
    if !a.dtype().is_narrow_float() {
        let sum = client.sum(a, dims, keepdim)?;
        return client.div_scalar(&sum, divisor);
    }

    let (a_promoted, original_dtype) = linalg_promote(client, a)?;
    let sum = client.sum(&a_promoted, dims, keepdim)?;
    let scaled = client.div_scalar(&sum, divisor)?;
    linalg_demote(client, scaled, original_dtype)
}
