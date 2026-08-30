//! Backward implementation for Group Normalization

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::error::Result;
use crate::ops::common::group_norm_channels_per_group;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for Group Normalization.
///
/// Input shape: `[B, C, *spatial]`. Normalizes over (C/G, *spatial) per group.
///
/// Gradients:
/// - d_input: similar to layer_norm but per-group
/// - d_weight = sum(grad_out * x_norm, batch_and_spatial_dims)
/// - d_bias = sum(grad_out, batch_and_spatial_dims)
pub struct GroupNormBackward<R: Runtime> {
    input_ids: [TensorId; 3], // [input, weight, bias]
    saved_input: Tensor<R>,
    saved_weight: Tensor<R>,
    num_groups: usize,
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime> GroupNormBackward<R> {
    /// Create a new GroupNormBackward
    pub fn new(
        input_id: TensorId,
        weight_id: TensorId,
        bias_id: TensorId,
        input: Tensor<R>,
        weight: Tensor<R>,
        num_groups: usize,
        eps: f32,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [input_id, weight_id, bias_id],
            saved_input: input,
            saved_weight: weight,
            num_groups,
            eps,
            input_grad_fns: [input_grad_fn, weight_grad_fn, bias_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for GroupNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // `x_norm_flat` and `rstd` are shared. d_input, d_weight and d_bias
        // each cost their own extra passes on top, so each is guarded.
        if !needed.iter().any(|&n| n) {
            return Ok(vec![None, None, None]);
        }

        let client = R::default_client(grad_output.device());
        let input = &self.saved_input;
        let weight = &self.saved_weight;
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        // Shared guard: rejects `num_groups == 0` before dividing. The forward op
        // rejects it too, but `GroupNormBackward::new` is public, so a bad group
        // count can reach here without ever passing through the forward check.
        let cpg = group_norm_channels_per_group(channels, self.num_groups)?;
        // Unclamped: a 2D input already products to 1 over the empty trailing slice,
        // and a zero spatial dim must stay 0 so the flattened reshape below keeps the
        // element count the empty input actually has.
        let spatial: usize = shape[2..].iter().product();
        let group_size = cpg * spatial;

        // A zero-element input contributes to no gradient: every reduction below
        // would fold over an empty group and produce a value no element supports.
        // `d_input` matches the empty input, and both parameter gradients sum over
        // nothing, so both are the additive identity.
        if input.numel() == 0 {
            let dtype = input.dtype();
            let device = input.device();
            let d_input = if needed[0] {
                Some(Tensor::<R>::zeros_generic(shape, dtype, device)?)
            } else {
                None
            };
            let d_weight = if needed[1] {
                Some(Tensor::<R>::zeros_generic(&[channels], dtype, device)?)
            } else {
                None
            };
            let d_bias = if needed[2] {
                Some(Tensor::<R>::zeros_generic(&[channels], dtype, device)?)
            } else {
                None
            };
            return Ok(vec![d_input, d_weight, d_bias]);
        }

        // Flatten to [B, G, C/G * spatial] for per-group normalization.
        // `input`/`grad_output` may be non-contiguous views (saved_input is
        // a clone of whatever the caller passed; grad_output may come from
        // a transpose/permute upstream), and `reshape` requires contiguity.
        let flat_shape = [batch, self.num_groups, group_size];
        let input_contig = super::super::ensure_contiguous(input)?;
        let grad_output_contig = super::super::ensure_contiguous(grad_output)?;
        let input_flat = input_contig.reshape(&flat_shape)?;
        let grad_flat = grad_output_contig.reshape(&flat_shape)?;

        // Per-group mean and variance: reduce over dim 2
        let mu = client.mean(&input_flat, &[2], true)?;
        let x_centered = client.sub(&input_flat, &mu)?;
        let x_sq = client.mul(&x_centered, &x_centered)?;
        let variance = client.mean(&x_sq, &[2], true)?;
        let var_eps = client.add_scalar(&variance, self.eps as f64)?;
        let std = client.sqrt(&var_eps)?;
        let rstd = client.recip(&std)?;
        let x_norm_flat = client.mul(&x_centered, &rstd)?;

        // d_input (per-group layer norm backward). The weight broadcast below
        // feeds this slot only, so it lives inside the guard.
        let d_input = if needed[0] {
            // Reshape weight [C] → [1, G, cpg, 1] → broadcast → [1, G, cpg, spatial] → [1, G, group_size]
            // `weight` (saved_weight) is a clone of whatever the caller passed
            // in, so it may be a sliced/transposed view.
            let weight_contig = super::super::ensure_contiguous(weight)?;
            let weight_4d = weight_contig.reshape(&[1, self.num_groups, cpg, 1])?;
            let weight_bcast = weight_4d
                .broadcast_to(&[1, self.num_groups, cpg, spatial])?
                .contiguous()?;
            let weight_flat = weight_bcast.reshape(&[1, self.num_groups, group_size])?;

            let gw = client.mul(&grad_flat, &weight_flat)?;
            let mean_gw = client.mean(&gw, &[2], true)?;
            let gw_xn = client.mul(&gw, &x_norm_flat)?;
            let mean_gw_xn = client.mean(&gw_xn, &[2], true)?;
            let xn_correction = client.mul(&x_norm_flat, &mean_gw_xn)?;
            let inner = client.sub(&gw, &mean_gw)?;
            let inner = client.sub(&inner, &xn_correction)?;
            let d_input_flat = client.mul(&inner, &rstd)?;
            Some(d_input_flat.reshape(shape)?)
        } else {
            None
        };

        // Both parameter gradients read the gradient in [B, C, spatial] layout.
        let grad_bcs = if needed[1] || needed[2] {
            Some(grad_output_contig.reshape(&[batch, channels, spatial])?)
        } else {
            None
        };

        // d_weight = sum(grad * x_norm, dims=[0, 2]) → [C]
        let d_weight = match (needed[1], &grad_bcs) {
            (true, Some(grad_bcs)) => {
                // x_norm reshaped back to [B, C, spatial]
                let x_norm_bcs = x_norm_flat.reshape(&[batch, channels, spatial])?;
                let gxn = client.mul(grad_bcs, &x_norm_bcs)?;
                Some(client.sum(&gxn, &[0, 2], false)?)
            }
            _ => None,
        };

        // d_bias = sum(grad, dims=[0, 2]) → [C]
        let d_bias = match (needed[2], &grad_bcs) {
            (true, Some(grad_bcs)) => Some(client.sum(grad_bcs, &[0, 2], false)?),
            _ => None,
        };

        Ok(vec![d_input, d_weight, d_bias])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        // For higher-order gradients, fall back to tensor backward wrapped in Var
        // Second-order traversal keeps every node, so ask for every gradient.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "GroupNormBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Build a `[rows, cols]` tensor that is logically equal to `data`
    /// (row-major) but backed by a non-contiguous transposed view, by
    /// storing the transpose of `data` and calling `.t()` on it.
    fn non_contiguous_like(
        data: &[f32],
        rows: usize,
        cols: usize,
        device: &CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let mut transposed = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        let src = Tensor::<CpuRuntime>::from_slice(&transposed, &[cols, rows], device).unwrap();
        let view = src.t().unwrap();
        assert!(!view.is_contiguous());
        assert_eq!(view.shape(), &[rows, cols]);
        view
    }

    #[test]
    fn test_group_norm_backward_non_contiguous_grad_output() {
        let device = CpuDevice::new();

        // batch=2, channels=3, num_groups=1, no spatial dims.
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad_data = [0.1f32, 0.2, -0.3, 0.4, -0.5, 0.6];

        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[2, 3], &device).unwrap();
        let weight = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = GroupNormBackward::<CpuRuntime>::new(
            TensorId::new(),
            TensorId::new(),
            TensorId::new(),
            input,
            weight,
            1,
            1e-5,
            None,
            None,
            None,
        );

        let grad_contig = Tensor::<CpuRuntime>::from_slice(&grad_data, &[2, 3], &device).unwrap();
        let grads_contig = backward.backward_all(&grad_contig).unwrap();

        let grad_noncontig = non_contiguous_like(&grad_data, 2, 3, &device);
        let grads_noncontig = backward.backward_all(&grad_noncontig).unwrap();

        for (a, b) in grads_contig.iter().zip(grads_noncontig.iter()) {
            let a = a.as_ref().unwrap().contiguous().unwrap();
            let b = b.as_ref().unwrap().contiguous().unwrap();
            assert_eq!(a.shape(), b.shape());
            let a_vals: Vec<f32> = a.to_vec();
            let b_vals: Vec<f32> = b.to_vec();
            assert_eq!(a_vals, b_vals);
        }

        // Sanity: results are not trivially zero/empty.
        let d_input: Vec<f32> = grads_contig[0]
            .as_ref()
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec();
        assert_eq!(d_input.len(), 6);
        assert!(d_input.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_group_norm_backward_non_contiguous_saved_input() {
        let device = CpuDevice::new();

        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad_data = [0.1f32, 0.2, -0.3, 0.4, -0.5, 0.6];

        let weight_contig = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();
        let grad_output = Tensor::<CpuRuntime>::from_slice(&grad_data, &[2, 3], &device).unwrap();

        let input_contig = Tensor::<CpuRuntime>::from_slice(&input_data, &[2, 3], &device).unwrap();
        let backward_contig = GroupNormBackward::<CpuRuntime>::new(
            TensorId::new(),
            TensorId::new(),
            TensorId::new(),
            input_contig,
            weight_contig.clone(),
            1,
            1e-5,
            None,
            None,
            None,
        );
        let grads_contig = backward_contig.backward_all(&grad_output).unwrap();

        let input_noncontig = non_contiguous_like(&input_data, 2, 3, &device);
        let backward_noncontig = GroupNormBackward::<CpuRuntime>::new(
            TensorId::new(),
            TensorId::new(),
            TensorId::new(),
            input_noncontig,
            weight_contig,
            1,
            1e-5,
            None,
            None,
            None,
        );
        let grads_noncontig = backward_noncontig.backward_all(&grad_output).unwrap();

        for (a, b) in grads_contig.iter().zip(grads_noncontig.iter()) {
            let a = a.as_ref().unwrap().contiguous().unwrap();
            let b = b.as_ref().unwrap().contiguous().unwrap();
            assert_eq!(a.shape(), b.shape());
            let a_vals: Vec<f32> = a.to_vec();
            let b_vals: Vec<f32> = b.to_vec();
            assert_eq!(a_vals, b_vals);
        }
    }
}
