//! Shared utilities for autograd backward implementations

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ensure a tensor is contiguous, making a copy if necessary.
///
/// Several `reshape` calls in backward implementations operate on tensors
/// that may be non-contiguous views (e.g. saved inputs, or grad_output after
/// a transpose/permute). `Tensor::reshape` intentionally errors on
/// non-contiguous tensors rather than silently materializing them, so
/// backward passes must call this helper first.
///
/// This intentionally duplicates [`crate::runtime::ensure_contiguous`] rather
/// than reusing it: that helper requires `R: Runtime<DType = DType>`, a bound
/// most `GradFn` impls in this module do not carry (and adding it would leak
/// into every `GradFn<R>` signature just to satisfy this one helper). Do NOT
/// collapse the two without first removing that bound from the callers here.
#[inline]
pub(crate) fn ensure_contiguous<R: Runtime>(tensor: &Tensor<R>) -> Result<Tensor<R>> {
    if tensor.is_contiguous() {
        Ok(tensor.clone())
    } else {
        tensor.contiguous()
    }
}
