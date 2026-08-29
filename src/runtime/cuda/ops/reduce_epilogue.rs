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
//! there, and demoting once at the end. Integers get the same divide-once
//! guarantee from the fused `reduce_mean_dim` kernel, which takes the divisor
//! as an explicit parameter rather than assuming it is the reduced axis
//! length (see `reduce_int.cu`).
//!
//! `div_scalar` itself is left alone — narrowing the scalar is correct for a
//! general elementwise-by-scalar op. The defect was building a reduction
//! epilogue out of it.

use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ScalarOps, reduce_output_shape};
use crate::runtime::cuda::kernels::launch_reduce_mean_dim_int_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Sum `a` over `dims`, then divide by `divisor`.
///
/// Two paths, by dtype:
///
/// - Narrow floats (F16, BF16, FP8) promote to F32 for both the sum and the
///   division, then demote once. A F16 sum above 65504 otherwise saturates to
///   infinity, and so does a F16 divisor above it.
/// - Integers run the fused `reduce_mean_dim` kernel via
///   [`int_sum_then_divide`], which sums in a 128-bit accumulator, divides by
///   `divisor` there, and narrows once. Built out of `sum` + `div_scalar` the
///   sum lands back in the element type first, so an overflowing total
///   saturates and the division then divides the clamped value — the wrong
///   answer even when the true quotient fits. This mirrors the CPU epilogue
///   in `cpu/kernels/reduce/int_acc.rs`, and answers both a plain mean
///   (`divisor` equal to the reduced element count) and an unbiased
///   variance's `n - correction` (which is not) the same way.
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

/// Convert a `divisor` produced by an epilogue caller (a plain element count,
/// or an unbiased variance's `n - correction`) into the `u64` the fused
/// integer kernel divides by.
///
/// `sum_then_divide` receives `divisor` as `f64` because the float path feeds
/// it straight to `div_scalar`, but an integer divisor is always a whole,
/// non-negative count. Reject anything else outright rather than truncating
/// or wrapping it into a silently wrong divisor.
fn int_divisor(divisor: f64) -> Result<u64> {
    let valid = divisor.is_finite()
        && divisor >= 0.0
        && divisor.fract() == 0.0
        && divisor <= u64::MAX as f64;

    if !valid {
        return Err(Error::InvalidArgument {
            arg: "divisor",
            reason: format!(
                "integer sum-then-divide requires a whole, non-negative divisor \
                 that fits in u64, got {divisor}"
            ),
        });
    }

    Ok(divisor as u64)
}

/// Integer sum-then-divide over `dims`, dividing exactly once by `divisor`
/// inside the fused kernel's 128-bit accumulator.
///
/// Chaining a per-dimension divide is wrong for integers: each step
/// truncates, so summing `[[0, 3], [0, 3], [0, 0]]` over both dims and
/// dividing by 6 at each step reports 0 instead of 1. The reduced dimensions
/// are therefore gathered into one axis and reduced by a single kernel
/// launch, which accumulates the whole total in 128 bits, divides by
/// `divisor` there, and truncates once — the same shape as the CPU's fused
/// integer mean, generalized to a `divisor` that need not be the reduced
/// element count.
fn int_sum_then_divide(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    divisor: f64,
) -> Result<Tensor<CudaRuntime>> {
    let divisor = int_divisor(divisor)?;

    if let [dim] = *dims {
        return native_reduce_mean_dim_divisor(client, a, dim, keepdim, divisor);
    }

    let shape = a.shape().to_vec();
    let out_shape = reduce_output_shape(&shape, dims, keepdim);

    if dims.is_empty() {
        // Nothing reduces away: every output element is exactly one input
        // element. Reshaping in a trailing size-1 axis and reducing over it
        // still divides each element by `divisor` exactly once, through the
        // same fused kernel, with no dedicated no-op path to keep in sync.
        let numel = a.numel();
        let flat = ensure_contiguous(a)?.reshape(&[numel, 1])?;
        let reduced = native_reduce_mean_dim_divisor(client, &flat, 1, false, divisor)?;
        return reduced.reshape(&out_shape);
    }

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
    let reduced_tensor =
        native_reduce_mean_dim_divisor(client, &flat, reduce_axis, false, divisor)?;

    reduced_tensor.reshape(&out_shape)
}

/// Launch the fused integer `reduce_mean_dim` kernel along one dimension,
/// dividing by an explicit `divisor` rather than assuming it equals the
/// reduced axis length.
///
/// Mirrors the single-dimension branch of `native_reduce_op` in
/// `runtime/cuda/ops/helpers.rs` (contiguity, shape math, the zero-output
/// early return), but calls [`launch_reduce_mean_dim_int_op`], whose kernel
/// takes `divisor` as a parameter instead. Callers needing more than one
/// reduced axis flatten them into one first, exactly as
/// [`int_sum_then_divide`] does.
fn native_reduce_mean_dim_divisor(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    keepdim: bool,
    divisor: u64,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let out_shape = reduce_output_shape(shape, &[dim], keepdim);

    let outer_size: usize = shape[..dim].iter().product::<usize>().max(1);
    let reduce_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>().max(1);

    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    // A zero-size output (some non-reduced dimension is 0) has nothing to
    // compute, and `outer_size`/`inner_size` were floored at 1 above, so
    // launching would write past the empty allocation.
    if out.numel() == 0 {
        return Ok(out);
    }

    unsafe {
        launch_reduce_mean_dim_int_op(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            outer_size,
            reduce_size,
            inner_size,
            divisor,
        )?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((device, client))
    }

    /// `sum_then_divide` with a `divisor` other than the reduced element
    /// count — an unbiased variance's `n - correction` shape — must still
    /// divide inside the wide accumulator, not after narrowing back to the
    /// element type. Three `I32` elements of `2_000_000_000` sum to
    /// `6_000_000_000`, which overflows `I32`, but dividing by `6` (not the
    /// element count `3`, so this is not a mean) lands back on
    /// `1_000_000_000`, which fits.
    ///
    /// Built out of `sum` + `div_scalar` directly, `sum` would first narrow
    /// its own saturating accumulator to `I32` — clamping to `i32::MAX`
    /// (`2_147_483_647`) — and only then divide by 6, giving `357_913_941`.
    /// This fails against that code and passes once the divide happens in a
    /// wider accumulator before any narrowing.
    #[test]
    fn test_int_sum_then_divide_with_non_count_divisor_does_not_saturate_first_i32() {
        let Some((device, client)) = setup() else {
            return;
        };

        let data = vec![2_000_000_000i32; 3];
        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).unwrap();

        let result = sum_then_divide(&client, &a, &[0], false, 6.0).unwrap();
        assert_eq!(result.dtype(), DType::I32);
        let got: Vec<i32> = result.to_vec();
        assert_eq!(got, vec![1_000_000_000]);
    }

    /// Same shape of bug as the `I32` case above, for `I64`: the old
    /// `int_sum_then_divide` had no wider tensor dtype to cast `I64` up to
    /// and fell back to plain `sum` + `div_scalar`, which saturates the
    /// total at `i64::MAX` before dividing. Three elements of
    /// `6_000_000_000_000_000_000` sum to `18_000_000_000_000_000_000`,
    /// which overflows `I64` (max `9_223_372_036_854_775_807`), but dividing
    /// by `6` lands back on `3_000_000_000_000_000_000`, which fits.
    #[test]
    fn test_int_sum_then_divide_with_non_count_divisor_does_not_saturate_first_i64() {
        let Some((device, client)) = setup() else {
            return;
        };

        let data = vec![6_000_000_000_000_000_000i64; 3];
        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).unwrap();

        let result = sum_then_divide(&client, &a, &[0], false, 6.0).unwrap();
        assert_eq!(result.dtype(), DType::I64);
        let got: Vec<i64> = result.to_vec();
        assert_eq!(got, vec![3_000_000_000_000_000_000]);
    }

    /// Same shape of bug again for `U32`: CUDA's `cast` kernel does not
    /// support `U32` as a source or destination at all, so the old code's
    /// only option for `U32` was the direct `sum` + `div_scalar` path, which
    /// saturates at `u32::MAX`. Three elements of `2_000_000_000u32` sum to
    /// `6_000_000_000`, which overflows `U32` (max `4_294_967_295`), but
    /// dividing by `6` lands back on `1_000_000_000`, which fits.
    #[test]
    fn test_int_sum_then_divide_with_non_count_divisor_does_not_saturate_first_u32() {
        let Some((device, client)) = setup() else {
            return;
        };

        let data = vec![2_000_000_000u32; 3];
        let a = Tensor::<CudaRuntime>::from_slice(&data, &[3], &device).unwrap();

        let result = sum_then_divide(&client, &a, &[0], false, 6.0).unwrap();
        assert_eq!(result.dtype(), DType::U32);
        let got: Vec<u32> = result.to_vec();
        assert_eq!(got, vec![1_000_000_000]);
    }

    /// An ordinary integer mean (`divisor` equal to the reduced element
    /// count) must still return exactly what it did before this kernel
    /// gained an explicit `divisor` parameter: `[1, 2, 3, 4]` sums to `10`
    /// and divides by the element count `4` for a mean of `2` (integer
    /// truncation), not `2.5`.
    #[test]
    fn test_int_sum_then_divide_with_count_divisor_matches_plain_mean() {
        let Some((device, client)) = setup() else {
            return;
        };

        let data = vec![1i32, 2, 3, 4];
        let a = Tensor::<CudaRuntime>::from_slice(&data, &[4], &device).unwrap();

        let result = sum_then_divide(&client, &a, &[0], false, 4.0).unwrap();
        assert_eq!(result.dtype(), DType::I32);
        let got: Vec<i32> = result.to_vec();
        assert_eq!(got, vec![2]);
    }
}
