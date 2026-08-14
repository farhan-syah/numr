//! Shared utilities for reduction backward implementations

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ensure a tensor is contiguous, making a copy if necessary.
///
/// Thin by-value wrapper around the crate-wide
/// [`crate::autograd::ops::ensure_contiguous`] helper, kept because callers
/// in this module pass freshly-produced owned tensors (e.g. from
/// `broadcast_to`) rather than borrows.
#[inline]
pub(super) fn ensure_contiguous<R: Runtime>(tensor: Tensor<R>) -> Result<Tensor<R>> {
    super::super::ensure_contiguous(&tensor)
}
