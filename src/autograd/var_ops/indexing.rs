//! Indexing operations (gather, embedding lookup)

use super::ops::*;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::IndexingOps;
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Gather along a dimension: z = gather(a, dim, index)
pub fn var_gather<R, C>(
    a: &Var<R>,
    dim: usize,
    index: &crate::tensor::Tensor<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R>,
    R::Client: IndexingOps<R>,
{
    let output = client.gather(a.tensor(), dim, index)?;

    if a.requires_grad() {
        let grad_fn =
            GatherBackward::<R>::new(a.id(), a.shape(), dim, index.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Embedding lookup: z = embedding_lookup(weight, indices)
///
/// `indices` are integer token IDs and are not differentiable. Gradients flow
/// only to `weight` and repeated token IDs accumulate additively.
pub fn var_embedding_lookup<R, C>(
    weight: &Var<R>,
    indices: &crate::tensor::Tensor<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R>,
    R::Client: IndexingOps<R>,
{
    let output = client.embedding_lookup(weight.tensor(), indices)?;

    if weight.requires_grad() {
        let grad_fn = EmbeddingLookupBackward::<R>::new(
            weight.id(),
            weight.shape(),
            indices.clone(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_gather_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Input: 2x3 matrix
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
                .unwrap(),
            true,
        );

        // Gather along dim=1 with indices [[0, 2], [1, 0]]
        let index = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 0], &[2, 2], &device).unwrap();
        let z = var_gather(&x, 1, &index, &client).unwrap();

        // z = [[1, 3], [5, 4]]
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![1.0, 3.0, 5.0, 4.0]);

        let loss = crate::autograd::var_ops::var_sum(&z, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        // Grad scatters 1s back: x[0,0] += 1, x[0,2] += 1, x[1,1] += 1, x[1,0] += 1
        // So grad = [[1, 0, 1], [1, 1, 0]]
        assert_eq!(grad_x, vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_var_embedding_lookup_backward_gradient_flows() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[
                    0.5f32, -1.0, 2.0, 3.5, 4.0, -2.5, 1.25, 0.75, -0.5, 2.25, 5.0, -3.0,
                ],
                &[4, 3],
                &device,
            )
            .unwrap(),
            true,
        );
        let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1], &[3], &device).unwrap();

        let out = var_embedding_lookup(&weight, &indices, &client).unwrap();
        let loss = crate::autograd::var_ops::var_sum(&out, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_weight = grads.get(weight.id()).expect("weight gradient missing");
        let grad_data = grad_weight.contiguous().unwrap().to_vec::<f32>();
        assert!(grad_data.iter().any(|&g| g != 0.0));
    }

    #[test]
    fn test_var_embedding_lookup_repeated_ids_accumulate() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let single_weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
                .unwrap(),
            true,
        );
        let single_indices = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).unwrap();
        let single_out = var_embedding_lookup(&single_weight, &single_indices, &client).unwrap();
        let single_loss =
            crate::autograd::var_ops::var_sum(&single_out, &[0, 1], false, &client).unwrap();
        let single_grads = backward(&single_loss, &client).unwrap();
        let single_grad = single_grads
            .get(single_weight.id())
            .expect("single weight gradient missing")
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        let repeated_weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
                .unwrap(),
            true,
        );
        let repeated_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 0], &[2], &device).unwrap();
        let repeated_out =
            var_embedding_lookup(&repeated_weight, &repeated_indices, &client).unwrap();
        let repeated_loss =
            crate::autograd::var_ops::var_sum(&repeated_out, &[0, 1], false, &client).unwrap();
        let repeated_grads = backward(&repeated_loss, &client).unwrap();
        let repeated_grad = repeated_grads
            .get(repeated_weight.id())
            .expect("repeated weight gradient missing")
            .contiguous()
            .unwrap()
            .to_vec::<f32>();

        assert_eq!(repeated_grad[0], single_grad[0] * 2.0);
        assert_eq!(repeated_grad[1], single_grad[1] * 2.0);
    }

    #[test]
    fn test_var_embedding_lookup_unindexed_rows_have_zero_grad() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0],
                &[3, 2],
                &device,
            )
            .unwrap(),
            true,
        );
        let indices = Tensor::<CpuRuntime>::from_slice(&[1i64], &[1], &device).unwrap();

        let out = var_embedding_lookup(&weight, &indices, &client).unwrap();
        let loss = crate::autograd::var_ops::var_sum(&out, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_data = grads
            .get(weight.id())
            .expect("weight gradient missing")
            .contiguous()
            .unwrap()
            .to_vec::<f32>();
        assert_eq!(grad_data[0], 0.0);
        assert_eq!(grad_data[1], 0.0);
        assert_eq!(grad_data[4], 0.0);
        assert_eq!(grad_data[5], 0.0);
    }
}
