//! PTX module selection for kernel families whose integer dtypes live in a
//! separate translation unit.
//!
//! Integer reductions and integer `cumsum`/`cumprod` accumulate in `Numr128`
//! rather than a float register, so they are compiled into their own module
//! under the same kernel names. Selecting the module is a straight swap.

use crate::dtype::DType;

use super::names::kernel_names;

/// The PTX module holding this dtype's cumulative kernels.
///
/// Integer `cumsum` and `cumprod` live in their own translation unit: they
/// accumulate in `Numr128` instead of a float register, and there is no integer
/// `logsumexp`. `cumulative_int.cu` uses the same kernel names and the same
/// launch ABI as `cumulative.cu`, so this is a straight swap of module, never of
/// kernel name.
#[inline]
pub(crate) fn cumulative_module(dtype: DType) -> &'static str {
    if dtype.is_int() {
        kernel_names::CUMULATIVE_INT_MODULE
    } else {
        kernel_names::CUMULATIVE_MODULE
    }
}

/// The PTX module holding this dtype's reduction kernels.
///
/// Integer reductions live in their own translation unit: they accumulate in
/// `Numr128` instead of a float register, and `reduce.cu` is already at its size
/// limit. `reduce_int.cu` instantiates every integer dtype and every reduction
/// name that `reduce.cu` does, so this is a straight swap of module, never of
/// kernel name.
#[inline]
pub(crate) fn reduce_module(dtype: DType) -> &'static str {
    if dtype.is_int() {
        kernel_names::REDUCE_INT_MODULE
    } else {
        kernel_names::REDUCE_MODULE
    }
}
