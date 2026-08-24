//! Generic implementations of shape operations (unfold, repeat_interleave).
//!
//! These are composite operations built from primitive shape operations (narrow, stack, cat).
//! All backends delegate to these implementations for numerical parity.

use crate::error::{Error, Result};
use crate::ops::ShapeOps;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Generic unfold implementation.
///
/// Extracts sliding windows along a dimension. Composed from narrow + stack + permute.
pub fn unfold_impl<R: Runtime, C: ShapeOps<R>>(
    client: &C,
    tensor: &Tensor<R>,
    dim: isize,
    size: usize,
    step: usize,
) -> Result<Tensor<R>> {
    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "cannot unfold a scalar tensor".to_string(),
        });
    }
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: "size must be greater than zero".to_string(),
        });
    }
    if step == 0 {
        return Err(Error::InvalidArgument {
            arg: "step",
            reason: "step must be greater than zero".to_string(),
        });
    }

    let dim_idx = if dim < 0 {
        let adjusted = ndim as isize + dim;
        if adjusted < 0 {
            return Err(Error::InvalidDimension { dim, ndim });
        }
        adjusted as usize
    } else {
        dim as usize
    };
    if dim_idx >= ndim {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    let dim_size = tensor.shape()[dim_idx];
    if size > dim_size {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!(
                "size ({}) must be <= dimension {} size ({})",
                size, dim_idx, dim_size
            ),
        });
    }

    let num_windows = (dim_size - size) / step + 1;

    let mut windows = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let start = i * step;
        windows.push(tensor.narrow(dim_idx as isize, start, size)?);
    }

    let refs: Vec<&Tensor<R>> = windows.iter().collect();
    let stacked = client.stack(&refs, dim_idx as isize)?;

    // stack inserts the window-count axis at dim_idx, and the window-size axis
    // lands at dim_idx + 1. Move the size axis to the end to match unfold semantics.
    let size_axis = dim_idx + 1;
    let out_ndim = stacked.ndim();
    if size_axis + 1 == out_ndim {
        return Ok(stacked);
    }

    let mut perm = Vec::with_capacity(out_ndim);
    for axis in 0..out_ndim {
        if axis != size_axis {
            perm.push(axis);
        }
    }
    perm.push(size_axis);
    stacked.permute(&perm)
}

/// Generic repeat_interleave implementation.
///
/// Repeats each element along a dimension. Composed from narrow + cat.
pub fn repeat_interleave_impl<R: Runtime, C: ShapeOps<R>>(
    client: &C,
    tensor: &Tensor<R>,
    repeats: usize,
    dim: Option<isize>,
) -> Result<Tensor<R>> {
    if repeats == 0 {
        return Err(Error::InvalidArgument {
            arg: "repeats",
            reason: "repeats must be greater than zero".to_string(),
        });
    }

    let (input, dim_idx) = match dim {
        Some(d) => {
            let ndim = tensor.ndim();
            if ndim == 0 {
                return Err(Error::InvalidDimension { dim: d, ndim: 0 });
            }
            let dim_idx = if d < 0 {
                let adjusted = ndim as isize + d;
                if adjusted < 0 {
                    return Err(Error::InvalidDimension { dim: d, ndim });
                }
                adjusted as usize
            } else {
                d as usize
            };
            if dim_idx >= ndim {
                return Err(Error::InvalidDimension { dim: d, ndim });
            }
            (tensor.clone(), dim_idx)
        }
        None => (tensor.contiguous()?.flatten()?, 0usize),
    };

    let dim_size = input.shape()[dim_idx];
    if dim_size == 0 {
        let mut out_shape = input.shape().to_vec();
        out_shape[dim_idx] = 0;
        return Ok(Tensor::<R>::empty(
            &out_shape,
            input.dtype(),
            input.device(),
        )?);
    }

    let mut chunks: Vec<Tensor<R>> = Vec::with_capacity(dim_size * repeats);
    for i in 0..dim_size {
        let slice = input.narrow(dim_idx as isize, i, 1)?;
        for _ in 0..repeats {
            chunks.push(slice.clone());
        }
    }

    let refs: Vec<&Tensor<R>> = chunks.iter().collect();
    client.cat(&refs, dim_idx as isize)
}
