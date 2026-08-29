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
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ReduceOps, ScalarOps, TypeConversionOps, reduce_output_shape};
use crate::runtime::cuda::ops::helpers::native_reduce_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// The number of elements `dims` reduces away, or `None` if any dim is out of
/// range for `a`.
fn reduced_element_count(a: &Tensor<CudaRuntime>, dims: &[usize]) -> Option<usize> {
    let shape = a.shape();
    dims.iter()
        .try_fold(1usize, |acc, &d| Some(acc * *shape.get(d)?))
}

/// Sum `a` over `dims`, then divide by `divisor`.
///
/// Three paths, by dtype:
///
/// - Narrow floats (F16, BF16, FP8) promote to F32 for both the sum and the
///   division, then demote once. A F16 sum above 65504 otherwise saturates to
///   infinity, and so does a F16 divisor above it.
/// - Integers with `divisor` equal to the reduced element count run the fused
///   `mean` kernel, which sums in a 128-bit accumulator, divides there, and
///   narrows once. Built out of `sum` + `div_scalar` the sum lands back in the
///   element type first, so an overflowing total saturates and the division
///   then divides the clamped value — the wrong answer even when the true mean
///   fits. This mirrors the CPU epilogue in `cpu/kernels/reduce/int_acc.rs`.
/// - Integers whose `divisor` is NOT the reduced element count (an unbiased
///   variance's `n - correction`, for instance) cannot use that fused kernel,
///   which only ever divides by the axis it reduced. [`int_sum_then_divide`]
///   gives them the same divide-once guarantee for an arbitrary divisor.
/// - Everything else (F32, F64, Bool) takes the direct `sum` + `div_scalar`
///   path unchanged.
pub(crate) fn sum_then_divide(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    divisor: f64,
) -> Result<Tensor<CudaRuntime>> {
    if a.dtype().is_int() {
        // The fused kernel divides by the elements it reduced, so it only
        // answers this call when the caller asked for exactly that divisor.
        if reduced_element_count(a, dims).is_some_and(|count| divisor == count as f64) {
            return int_mean(client, a, dims, keepdim);
        }
        return int_sum_then_divide(client, a, dims, keepdim, divisor);
    }

    if !a.dtype().is_narrow_float() {
        let sum = client.sum(a, dims, keepdim)?;
        return client.div_scalar(&sum, divisor);
    }

    let (a_promoted, original_dtype) = linalg_promote(client, a)?;
    let sum = client.sum(&a_promoted, dims, keepdim)?;
    let scaled = client.div_scalar(&sum, divisor)?;
    linalg_demote(client, scaled, original_dtype)
}

/// Integer sum-then-divide for a `divisor` that is not the reduced element
/// count, so the fused `mean` kernel (which only divides by the axis it
/// reduced) cannot serve it — the unbiased variance's `n - correction` is the
/// caller today, reached only if a future change stops promoting integer
/// input to F32 before this epilogue runs (see the module doc).
///
/// `client.sum` narrows its 128-bit accumulator back to `a`'s own dtype
/// before returning, so dividing that result is wrong whenever the true total
/// leaves the dtype's range while the final quotient would not have — the
/// same failure [`int_mean`] exists to avoid, just for a divisor the fused
/// kernel cannot take. `I32` has a lossless wider dtype CUDA can cast to
/// (`I64`), so cast up first, sum and divide there with a truncating integer
/// division, and narrow back once, saturating on overflow.
///
/// `I64` and `U32` take the direct `sum` + `div_scalar` path instead, which
/// saturates on the same inputs `sum` alone already would:
/// - `I64` has no wider tensor dtype in numr to cast up to — the accepted
///   limit `WideAcc` documents for 64-bit integers.
/// - `U32` has a wider dtype (`I64`) that would losslessly hold it, but
///   CUDA's `cast` kernel does not support `U32` as a source or destination
///   at all (only `F32, F64, F16, BF16, FP8E4M3, FP8E5M2, I32, I64, Bool` —
///   see `runtime/cuda/kernels/cast.rs`), so casting up is not available.
fn int_sum_then_divide(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    divisor: f64,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    if dtype != DType::I32 {
        let sum = client.sum(a, dims, keepdim)?;
        return client.div_scalar(&sum, divisor);
    }

    let wide = client.cast(a, DType::I64)?;
    let sum = client.sum(&wide, dims, keepdim)?;
    let scaled = client.div_scalar(&sum, divisor)?;
    client.cast(&scaled, dtype)
}

/// Integer `mean` over `dims`, dividing exactly once.
///
/// Chaining a per-dimension `mean` is wrong for integers: each step truncates,
/// so `mean([[0, 3], [0, 3], [0, 0]])` over both dims reports 0 instead of 1.
/// The reduced dimensions are therefore gathered into one axis and reduced by a
/// single kernel launch, which accumulates the whole total in 128 bits and
/// truncates once — the same shape as the CPU's fused integer mean.
fn int_mean(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    if dims.len() <= 1 {
        return native_reduce_op(client, a, "mean", dims, keepdim, None);
    }

    let shape = a.shape().to_vec();
    let out_shape = reduce_output_shape(&shape, dims, keepdim);

    // Kept axes stay in their original order, which is the order
    // `reduce_output_shape` reports them in, so the final reshape is a
    // relabelling and never a transpose.
    let kept: Vec<usize> = (0..shape.len()).filter(|d| !dims.contains(d)).collect();
    let mut reduced: Vec<usize> = dims.to_vec();
    reduced.sort_unstable();

    let order: Vec<usize> = kept.iter().chain(reduced.iter()).copied().collect();
    let permuted = ensure_contiguous(&a.permute(&order)?)?;

    let mut flat_shape: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    flat_shape.push(reduced.iter().map(|&d| shape[d]).product());
    let reduce_axis = flat_shape.len() - 1;

    let flat = permuted.reshape(&flat_shape)?;
    let reduced_tensor = native_reduce_op(client, &flat, "mean", &[reduce_axis], false, None)?;

    reduced_tensor.reshape(&out_shape)
}
