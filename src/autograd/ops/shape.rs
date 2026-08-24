//! Backward implementations for shape operations
//!
//! These implement gradient computation for shape-changing operations
//! like reshape, transpose, squeeze, and unsqueeze.
//!
//! # Key Property
//!
//! Shape operations have identity Jacobians - the gradient of reshaping
//! is just reshaping the gradient back to the original shape.

use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ReduceOps, ShapeOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// ReshapeBackward
// ============================================================================

/// Backward for reshape: z = reshape(a, new_shape)
///
/// Gradient: dL/da = reshape(dL/dz, original_shape)
/// The gradient simply reshapes back to the input's original shape.
pub struct ReshapeBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ReshapeBackward<R> {
    /// Create a new ReshapeBackward
    pub fn new(
        input_id: TensorId,
        input_shape: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for ReshapeBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input: the driver already skipped this node if its one
        // gradient were unwanted, so the mask carries no extra information.
        // Reshape gradient back to input shape. grad_output may be a
        // non-contiguous view (e.g. after a transpose/permute), and
        // `reshape` requires contiguity.
        let grad_output = super::ensure_contiguous(grad_output)?;
        let grad = grad_output.reshape(&self.input_shape)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Reshape gradient back to input shape
        // Reshape is a view operation with identity Jacobian, so we just
        // reshape and preserve the gradient function from grad_output
        let grad_output_tensor = super::ensure_contiguous(grad_output.tensor())?;
        let reshaped = grad_output_tensor.reshape(&self.input_shape)?;

        // Create ReshapeBackward to track the chain
        let grad_fn = ReshapeBackward::<R>::new(
            grad_output.id(),
            grad_output.shape().to_vec(),
            grad_output.grad_fn().cloned(),
        );

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(reshaped, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(reshaped, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

// ============================================================================
// TransposeBackward
// ============================================================================

/// Backward for transpose: z = transpose(a)
///
/// Gradient: dL/da = transpose(dL/dz)
/// The gradient is just the transpose of the output gradient.
pub struct TransposeBackward<R: Runtime> {
    input_id: TensorId,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> TransposeBackward<R> {
    /// Create a new TransposeBackward
    pub fn new(input_id: TensorId, input_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for TransposeBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        // Transpose gradient back
        let grad = grad_output.t()?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Transpose gradient back
        let transposed = grad_output.tensor().t()?;

        // Create TransposeBackward to track the chain
        let grad_fn = TransposeBackward::<R>::new(grad_output.id(), grad_output.grad_fn().cloned());

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(transposed, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(transposed, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

// ============================================================================
// PermuteBackward
// ============================================================================

/// Backward for permute: z = permute(a, dims)
///
/// Gradient: dL/da = permute(dL/dz, inverse_dims)
/// The gradient uses the inverse permutation.
pub struct PermuteBackward<R: Runtime> {
    input_id: TensorId,
    inverse_dims: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> PermuteBackward<R> {
    /// Create a new PermuteBackward
    ///
    /// # Arguments
    /// * `input_id` - ID of the input tensor
    /// * `dims` - The permutation dimensions used in forward pass
    /// * `input_grad_fn` - Gradient function of input
    pub fn new(
        input_id: TensorId,
        dims: &[usize],
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        // Compute inverse permutation
        let mut inverse_dims = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inverse_dims[d] = i;
        }

        Self {
            input_id,
            inverse_dims,
            input_grad_fn,
        }
    }

    /// Create from inverse dims directly (for backward_var chaining)
    fn from_inverse(
        input_id: TensorId,
        inverse_dims: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            inverse_dims,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for PermuteBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        // Apply inverse permutation
        let grad = grad_output.permute(&self.inverse_dims)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Apply inverse permutation
        let permuted = grad_output.tensor().permute(&self.inverse_dims)?;

        // The inverse of the inverse is the original permutation
        // So we pass self.inverse_dims as the "forward" dims for the new backward
        let grad_fn = PermuteBackward::<R>::from_inverse(
            grad_output.id(),
            self.inverse_dims.clone(),
            grad_output.grad_fn().cloned(),
        );

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(permuted, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(permuted, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

// ============================================================================
// ExpandBackward (for broadcast_to)
// ============================================================================

/// Backward for expand/broadcast_to: z = broadcast_to(a, new_shape)
///
/// Gradient: dL/da = sum(dL/dz, broadcast_dims)
/// The gradient sums over dimensions that were broadcast.
pub struct ExpandBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ExpandBackward<R> {
    /// Create a new ExpandBackward
    pub fn new(
        input_id: TensorId,
        input_shape: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for ExpandBackward<R>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ReduceOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // Find dimensions that were broadcast and need to be summed
        let output_shape = grad_output.shape();
        let input_ndim = self.input_shape.len();
        let output_ndim = output_shape.len();

        let mut result = grad_output.clone();

        // Sum over extra leading dimensions
        if output_ndim > input_ndim {
            let extra_dims: Vec<usize> = (0..(output_ndim - input_ndim)).collect();
            result = client.sum(&result, &extra_dims, false)?;
        }

        // Sum over dimensions where input had size 1
        let offset = output_ndim.saturating_sub(input_ndim);
        let mut reduce_dims = Vec::new();
        for (i, &input_dim) in self.input_shape.iter().enumerate() {
            let output_idx = offset + i;
            if input_dim == 1 && output_shape[output_idx] > 1 {
                reduce_dims.push(i);
            }
        }

        if !reduce_dims.is_empty() {
            result = client.sum(&result, &reduce_dims, true)?;
        }

        // Reshape to input shape if needed
        if result.shape() != self.input_shape.as_slice() {
            result = result.reshape(&self.input_shape)?;
        }

        Ok(vec![Some(result)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        use crate::autograd::var_sum;

        let client = R::default_client(grad_output.tensor().device());

        let output_shape = grad_output.shape();
        let input_ndim = self.input_shape.len();
        let output_ndim = output_shape.len();

        let mut result = grad_output.clone();

        // Sum over extra leading dimensions using var_sum
        if output_ndim > input_ndim {
            let extra_dims: Vec<usize> = (0..(output_ndim - input_ndim)).collect();
            result = var_sum(&result, &extra_dims, false, &client)?;
        }

        // Sum over dimensions where input had size 1
        let offset = output_ndim.saturating_sub(input_ndim);
        let mut reduce_dims = Vec::new();
        for (i, &input_dim) in self.input_shape.iter().enumerate() {
            let output_idx = offset + i;
            if output_idx < output_shape.len() && input_dim == 1 && output_shape[output_idx] > 1 {
                reduce_dims.push(i);
            }
        }

        if !reduce_dims.is_empty() {
            result = var_sum(&result, &reduce_dims, true, &client)?;
        }

        // Reshape to input shape if needed using var_reshape
        if result.shape() != self.input_shape.as_slice() {
            result = var_reshape(&result, &self.input_shape)?;
        }

        Ok(vec![Some(result)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

// ============================================================================
// Var Operations for Shape
// ============================================================================

/// Reshape a Var to a new shape
///
/// Creates ReshapeBackward for gradient computation.
pub fn var_reshape<R: Runtime>(a: &Var<R>, shape: &[usize]) -> Result<Var<R>> {
    let output = a.tensor().reshape(shape)?;

    if a.requires_grad() {
        let grad_fn = ReshapeBackward::<R>::new(a.id(), a.shape().to_vec(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Transpose a Var (swap last two dimensions)
///
/// Creates TransposeBackward for gradient computation.
pub fn var_transpose<R: Runtime>(a: &Var<R>) -> Result<Var<R>> {
    let output = a.tensor().t()?;

    if a.requires_grad() {
        let grad_fn = TransposeBackward::<R>::new(a.id(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Permute a Var's dimensions
///
/// Creates PermuteBackward for gradient computation.
pub fn var_permute<R: Runtime>(a: &Var<R>, dims: &[usize]) -> Result<Var<R>> {
    let output = a.tensor().permute(dims)?;

    if a.requires_grad() {
        let grad_fn = PermuteBackward::<R>::new(a.id(), dims, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Broadcast a Var to a new shape
///
/// Creates ExpandBackward for gradient computation.
pub fn var_broadcast_to<R: Runtime>(a: &Var<R>, shape: &[usize]) -> Result<Var<R>>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ReduceOps<R>,
{
    let output = a.tensor().broadcast_to(shape)?;

    if a.requires_grad() {
        let grad_fn = ExpandBackward::<R>::new(a.id(), a.shape().to_vec(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

// ============================================================================
// NarrowBackward
// ============================================================================

/// Backward for narrow: z = narrow(a, dim, start, length)
///
/// Gradient: dL/da is a zero tensor with dL/dz placed at the sliced region.
/// We use pad-with-zeros: create zeros of original shape, then add the gradient
/// into the narrow region.
pub struct NarrowBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dim: usize,
    start: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> NarrowBackward<R> {
    /// Create a new `NarrowBackward` node.
    ///
    /// - `input_id` — ID of the input tensor before narrowing
    /// - `input_shape` — original shape of the input tensor
    /// - `dim` — dimension that was narrowed
    /// - `start` — start index along `dim`
    /// - `input_grad_fn` — gradient function of the input, if it requires grad
    pub fn new(
        input_id: TensorId,
        input_shape: Vec<usize>,
        dim: usize,
        start: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape,
            dim,
            start,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for NarrowBackward<R>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ShapeOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // Pad gradient back to original size along the narrowed dimension.
        // Before: zeros of size [start], After: zeros of size [orig_dim - start - length]
        let length = grad_output.shape()[self.dim];
        let orig_dim_size = self.input_shape[self.dim];
        let end = self.start + length;

        let mut parts: Vec<Tensor<R>> = Vec::new();

        // Padding before the narrow region
        if self.start > 0 {
            let mut pad_shape = self.input_shape.clone();
            pad_shape[self.dim] = self.start;
            parts.push(Tensor::<R>::zeros(
                &pad_shape,
                grad_output.dtype(),
                grad_output.device(),
            ));
        }

        // The gradient itself (make contiguous for cat)
        parts.push(grad_output.contiguous()?);

        // Padding after the narrow region
        if end < orig_dim_size {
            let mut pad_shape = self.input_shape.clone();
            pad_shape[self.dim] = orig_dim_size - end;
            parts.push(Tensor::<R>::zeros(
                &pad_shape,
                grad_output.dtype(),
                grad_output.device(),
            ));
        }

        let refs: Vec<&Tensor<R>> = parts.iter().collect();
        let grad_input = client.cat(&refs, self.dim as isize)?;

        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());

        let length = grad_output.shape()[self.dim];
        let orig_dim_size = self.input_shape[self.dim];
        let end = self.start + length;

        let mut parts: Vec<Tensor<R>> = Vec::new();

        if self.start > 0 {
            let mut pad_shape = self.input_shape.clone();
            pad_shape[self.dim] = self.start;
            parts.push(Tensor::<R>::zeros(
                &pad_shape,
                grad_output.tensor().dtype(),
                grad_output.tensor().device(),
            ));
        }

        parts.push(grad_output.tensor().contiguous()?);

        if end < orig_dim_size {
            let mut pad_shape = self.input_shape.clone();
            pad_shape[self.dim] = orig_dim_size - end;
            parts.push(Tensor::<R>::zeros(
                &pad_shape,
                grad_output.tensor().dtype(),
                grad_output.tensor().device(),
            ));
        }

        let refs: Vec<&Tensor<R>> = parts.iter().collect();
        let grad_input = client.cat(&refs, self.dim as isize)?;

        Ok(vec![Some(Var::new(grad_input, false))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "NarrowBackward"
    }
}

// ============================================================================
// CatBackward
// ============================================================================

/// Backward for cat: z = cat([a, b, ...], dim)
///
/// Gradient: split dL/dz along dim, one slice per input.
pub struct CatBackward<R: Runtime> {
    input_ids: Vec<TensorId>,
    /// Size of each input along the cat dimension
    split_sizes: Vec<usize>,
    dim: usize,
    input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
}

impl<R: Runtime> CatBackward<R> {
    /// Create a new `CatBackward` node.
    ///
    /// - `input_ids` — IDs of the input tensors that were concatenated
    /// - `split_sizes` — size of each input along the cat dimension
    /// - `dim` — dimension along which the inputs were concatenated
    /// - `input_grad_fns` — gradient functions of each input, if they require grad
    pub fn new(
        input_ids: Vec<TensorId>,
        split_sizes: Vec<usize>,
        dim: usize,
        input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
    ) -> Self {
        Self {
            input_ids,
            split_sizes,
            dim,
            input_grad_fns,
        }
    }
}

impl<R: Runtime> GradFn<R> for CatBackward<R> {
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // Each slot is an independent `contiguous()` copy of its slice, so an
        // unwanted operand costs nothing. Concatenating a trainable tensor onto
        // a frozen one — a KV cache appended to cached history — skips the
        // frozen half's copy entirely.
        let mut grads = Vec::with_capacity(self.split_sizes.len());
        let mut offset = 0;
        for (i, &size) in self.split_sizes.iter().enumerate() {
            if needed[i] {
                let grad_slice = grad_output.narrow(self.dim as isize, offset, size)?;
                // Make contiguous so downstream ops get clean data
                grads.push(Some(grad_slice.contiguous()?));
            } else {
                grads.push(None);
            }
            offset += size;
        }
        Ok(grads)
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let mut grads = Vec::with_capacity(self.split_sizes.len());
        let mut offset = 0;
        for &size in &self.split_sizes {
            let grad_slice = grad_output
                .tensor()
                .narrow(self.dim as isize, offset, size)?
                .contiguous()?;
            grads.push(Some(Var::new(grad_slice, false)));
            offset += size;
        }
        Ok(grads)
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.clone()
    }

    fn name(&self) -> &'static str {
        "CatBackward"
    }
}

// ============================================================================
// Var Operations for Narrow and Cat
// ============================================================================

/// Narrow (slice) a Var along a dimension
///
/// Creates NarrowBackward for gradient computation.
pub fn var_narrow<R: Runtime<DType = DType>>(
    a: &Var<R>,
    dim: isize,
    start: usize,
    length: usize,
) -> Result<Var<R>>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ShapeOps<R>,
{
    let dim_idx =
        a.tensor()
            .layout()
            .normalize_dim(dim)
            .ok_or(crate::error::Error::InvalidDimension {
                dim,
                ndim: a.ndim(),
            })?;

    let output = a.tensor().narrow(dim, start, length)?;

    if a.requires_grad() {
        let grad_fn = NarrowBackward::<R>::new(
            a.id(),
            a.shape().to_vec(),
            dim_idx,
            start,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Concatenate Vars along a dimension
///
/// Creates CatBackward for gradient computation.
pub fn var_cat<R, C>(vars: &[&Var<R>], dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + crate::ops::ShapeOps<R>,
{
    if vars.is_empty() {
        return Err(crate::error::Error::InvalidArgument {
            arg: "vars",
            reason: "var_cat requires at least one input".into(),
        });
    }

    let tensors: Vec<&Tensor<R>> = vars.iter().map(|v| v.tensor()).collect();
    let output = client.cat(&tensors, dim)?;

    let any_requires_grad = vars.iter().any(|v| v.requires_grad());

    if any_requires_grad {
        // Normalize dim for split_sizes
        let dim_idx = vars[0].tensor().layout().normalize_dim(dim).ok_or(
            crate::error::Error::InvalidDimension {
                dim,
                ndim: vars[0].ndim(),
            },
        )?;

        let input_ids: Vec<TensorId> = vars.iter().map(|v| v.id()).collect();
        let split_sizes: Vec<usize> = vars.iter().map(|v| v.shape()[dim_idx]).collect();
        let input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>> =
            vars.iter().map(|v| v.grad_fn().cloned()).collect();

        let grad_fn = CatBackward::<R>::new(input_ids, split_sizes, dim_idx, input_grad_fns);
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_reshape_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3], reshape to [3, 2]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 2], DType::F32, &device);

        let backward = ReshapeBackward::<CpuRuntime>::new(input.id(), vec![2, 3], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_backward_non_contiguous_grad_output() {
        let device = CpuDevice::new();

        // grad_output is [3, 2] but transposed to a non-contiguous [2, 3]
        // view before being fed into backward(); reshape must still work.
        let grad_source =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
        let grad_out = grad_source.t().unwrap();
        assert!(!grad_out.is_contiguous());

        let backward = ReshapeBackward::<CpuRuntime>::new(TensorId::new(), vec![2, 3], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();
        // grad_source row-major [3,2] = [[1,2],[3,4],[5,6]]; transposed [2,3]
        // = [[1,3,5],[2,4,6]]; reshaped back to [2,3] reads the transposed
        // data in row-major order.
        assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_reshape_backward_var_non_contiguous_grad_output() {
        let device = CpuDevice::new();

        let grad_source =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
        let grad_out_tensor = grad_source.t().unwrap();
        assert!(!grad_out_tensor.is_contiguous());
        let grad_out_var = Var::new(grad_out_tensor, true);

        let backward = ReshapeBackward::<CpuRuntime>::new(TensorId::new(), vec![2, 3], None);
        let grads = backward.backward_var(&grad_out_var).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
        let values: Vec<f32> = grad.tensor().contiguous().unwrap().to_vec();
        assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3], transpose to [3, 2]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 2], DType::F32, &device);

        let backward = TransposeBackward::<CpuRuntime>::new(input.id(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
    }

    #[test]
    fn test_var_reshape() {
        let device = CpuDevice::new();

        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let x = Var::new(tensor, true);

        let y = var_reshape(&x, &[3, 2]).unwrap();

        assert_eq!(y.shape(), &[3, 2]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "ReshapeBackward");
    }

    #[test]
    fn test_var_transpose() {
        let device = CpuDevice::new();

        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let x = Var::new(tensor, true);

        let y = var_transpose(&x).unwrap();

        assert_eq!(y.shape(), &[3, 2]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "TransposeBackward");
    }

    #[test]
    fn test_permute_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3, 4], permute to [3, 4, 2] (dims = [1, 2, 0])
        let input = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 4, 2], DType::F32, &device);

        let backward = PermuteBackward::<CpuRuntime>::new(input.id(), &[1, 2, 0], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        // Inverse permutation of [1, 2, 0] is [2, 0, 1]
        // So output shape [3, 4, 2] -> [2, 3, 4]
        assert_eq!(grad.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_var_permute() {
        let device = CpuDevice::new();

        let tensor = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let x = Var::new(tensor, true);

        let y = var_permute(&x, &[2, 0, 1]).unwrap();

        assert_eq!(y.shape(), &[4, 2, 3]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "PermuteBackward");
    }

    #[test]
    fn test_expand_backward() {
        let device = CpuDevice::new();

        // Input shape [1, 3], expanded to [2, 3]
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device);

        let backward = ExpandBackward::<CpuRuntime>::new(input.id(), vec![1, 3], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 3]);

        // Each element should be summed over the broadcast dimension
        // grad_out is all 1s, broadcast dim had size 2, so sum = 2
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_var_broadcast_to() {
        let device = CpuDevice::new();

        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let x = Var::new(tensor, true);

        let y = var_broadcast_to(&x, &[2, 3]).unwrap();

        assert_eq!(y.shape(), &[2, 3]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "ExpandBackward");

        // Verify values are broadcast correctly (need contiguous for to_vec)
        let y_contiguous = y.tensor().contiguous().unwrap();
        let y_data: Vec<f32> = y_contiguous.to_vec();
        assert_eq!(y_data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reshape_backward_scalar() {
        let device = CpuDevice::new();

        // Scalar reshape to [1] and back
        let input = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device);

        let backward = ReshapeBackward::<CpuRuntime>::new(input.id(), vec![], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_transpose_backward_3d() {
        let device = CpuDevice::new();

        // 3D tensor transpose (swaps last two dims)
        let input = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 4, 3], DType::F32, &device);

        let backward = TransposeBackward::<CpuRuntime>::new(input.id(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_expand_backward_multiple_dims() {
        let device = CpuDevice::new();

        // Input shape [1, 1], expanded to [3, 4]
        let input = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 4], DType::F32, &device);

        let backward = ExpandBackward::<CpuRuntime>::new(input.id(), vec![1, 1], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 1]);

        // Sum over both dimensions: 3 * 4 = 12
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![12.0]);
    }

    /// `CatBackward` slices its gradient once per input. An unwanted operand's
    /// slice-and-copy must not happen, and the wanted ones must be unchanged.
    #[test]
    fn test_cat_backward_skips_unneeded_operands() {
        let device = CpuDevice::new();

        let ids = vec![TensorId::new(), TensorId::new(), TensorId::new()];
        let backward =
            CatBackward::<CpuRuntime>::new(ids, vec![1, 2, 1], 0, vec![None, None, None]);

        let grad_out = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            &device,
        );

        let all = backward.backward_all(&grad_out).unwrap();
        let masked = backward.backward(&grad_out, &[false, true, false]).unwrap();

        assert_eq!(masked.len(), 3, "one slot per input");
        assert!(masked[0].is_none(), "first operand is dead");
        assert!(masked[2].is_none(), "third operand is dead");

        // The kept slice must come from the right offset — a skipped operand
        // must not shift the ones after it.
        let want: Vec<f32> = all[1].as_ref().unwrap().to_vec();
        let got: Vec<f32> = masked[1].as_ref().unwrap().to_vec();
        assert_eq!(want, vec![3.0f32, 4.0, 5.0, 6.0]);
        assert_eq!(want, got);
    }

    #[test]
    fn test_permute_backward_identity() {
        let device = CpuDevice::new();

        // Identity permutation
        let input = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device);
        let grad_out =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let backward = PermuteBackward::<CpuRuntime>::new(input.id(), &[0, 1], None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);

        // Identity permutation should pass through unchanged
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_var_narrow() {
        let device = CpuDevice::new();

        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device);
        let x = Var::new(tensor, true);

        let y = var_narrow(&x, 0, 1, 3).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert!(y.requires_grad());
        assert_eq!(y.grad_fn().unwrap().name(), "NarrowBackward");

        let y_data: Vec<f32> = y.tensor().to_vec();
        assert_eq!(y_data, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_narrow_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device),
            true,
        );

        // narrow(dim=0, start=1, length=3) -> [2.0, 3.0, 4.0]
        let y = var_narrow(&x, 0, 1, 3).unwrap();
        let loss = crate::autograd::var_sum(&y, &[0], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        // Gradient should be [0, 1, 1, 1, 0] — ones in the narrow region, zeros outside
        assert_eq!(grad_x, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_var_cat() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &device),
            true,
        );

        let c = var_cat(&[&a, &b], 0, &client).unwrap();
        assert_eq!(c.shape(), &[5]);
        assert!(c.requires_grad());
        assert_eq!(c.grad_fn().unwrap().name(), "CatBackward");

        let c_data: Vec<f32> = c.tensor().to_vec();
        assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cat_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0], &[3], &device),
            true,
        );

        let c = var_cat(&[&a, &b], 0, &client).unwrap();
        let loss = crate::autograd::var_sum(&c, &[0], false, &client).unwrap();
        let grads = crate::autograd::backward(&loss, &client).unwrap();

        let grad_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();

        // Sum backward → all ones, split back to original sizes
        assert_eq!(grad_a, vec![1.0, 1.0]);
        assert_eq!(grad_b, vec![1.0, 1.0, 1.0]);
    }
}
