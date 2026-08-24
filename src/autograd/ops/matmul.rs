//! Backward implementation for matrix multiplication
//!
//! Implements gradient computation for matmul: C = A @ B

use super::broadcast::{reduce_grad_for_broadcast, reduce_var_for_broadcast};
use crate::autograd::var_ops::var_matmul;
use crate::autograd::{GradFn, Var};
use crate::error::Result;
use crate::ops::{MatmulOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// MatmulBackward
// ============================================================================

/// Backward for matrix multiplication: C = A @ B
///
/// For C = A @ B where A is (m, k) and B is (k, n):
/// - dL/dA = dL/dC @ B^T  (shape: m x k)
/// - dL/dB = A^T @ dL/dC  (shape: k x n)
///
/// Supports batched matmul as well.
pub struct MatmulBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [a, b]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> MatmulBackward<R> {
    /// Create a new MatmulBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, b],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for MatmulBackward<R>
where
    R::Client: MatmulOps<R> + TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC
        //
        // The two are separate matmuls sharing nothing but `grad_output`, so
        // each is guarded on its own. A frozen `Linear` wants only dL/dA; the
        // dL/dB matmul below is the full-size wasted GEMM this skips.
        let grad_a = if needed[0] {
            // Transpose B: swap last two dimensions
            let b_t = saved_b.t()?;
            let grad_a_full = client.matmul(grad_output, &b_t)?;
            // Batch dims may have been broadcast during forward: sum the
            // gradient back over the dims where A had extent 1.
            Some(reduce_grad_for_broadcast::<R>(
                &grad_a_full,
                saved_a.shape(),
            )?)
        } else {
            None
        };

        let grad_b = if needed[1] {
            // Transpose A: swap last two dimensions
            let a_t = saved_a.t()?;
            let grad_b_full = client.matmul(&a_t, grad_output)?;
            Some(reduce_grad_for_broadcast::<R>(
                &grad_b_full,
                saved_b.shape(),
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
    {
        use super::shape::var_transpose;

        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC

        // Wrap saved tensors as Vars with original IDs AND grad_fns
        // This is essential for second-order derivatives: if A or B themselves
        // came from computations (e.g., A = X + Y), we need to continue the
        // gradient chain through them.
        let a_var = Var::with_id_and_grad_fn(
            saved_a.clone(),
            self.input_ids[0],
            self.input_grad_fns[0].clone(),
        );
        let b_var = Var::with_id_and_grad_fn(
            saved_b.clone(),
            self.input_ids[1],
            self.input_grad_fns[1].clone(),
        );

        // Transpose B using var_transpose to maintain gradient chain
        let b_t_var = var_transpose(&b_var)?;

        // dL/dA = dL/dC @ B^T
        let grad_a_full = var_matmul(grad_output, &b_t_var, &client)?;

        // Transpose A using var_transpose to maintain gradient chain
        let a_t_var = var_transpose(&a_var)?;

        // dL/dB = A^T @ dL/dC
        let grad_b_full = var_matmul(&a_t_var, grad_output, &client)?;

        // Batch dims may have been broadcast during forward: sum each operand's
        // gradient back over the dims where that operand had extent 1.
        let grad_a = reduce_var_for_broadcast(&grad_a_full, saved_a.shape(), &client)?;
        let grad_b = reduce_var_for_broadcast(&grad_b_full, saved_b.shape(), &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Raw bit pattern of every element — no tolerance, no rounding.
    fn bits(tensor: &Tensor<CpuRuntime>) -> Vec<u32> {
        tensor
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// The frozen-`Linear` case: only operand A is wanted.
    ///
    /// The dead slot must come back `None` — in the guarded impl the `Some`
    /// value has no producer outside the branch, so `None` means the
    /// `A^T @ dL/dC` matmul never ran. The wanted gradient must be bit-identical
    /// to the all-true run, since guarding must not perturb the kept path.
    #[test]
    fn test_matmul_backward_skips_unneeded_operand() {
        let device = CpuDevice::new();

        // A: [2, 3], B: [3, 4], C: [2, 4]
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.5f32, -2.25, 0.75, 3.0, -0.5, 4.25],
            &[2, 3],
            &device,
        )
        .unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(
            &[
                0.5f32, 1.25, -2.0, 3.5, -1.75, 0.25, 2.75, -0.125, 4.0, -3.25, 1.0, 0.625,
            ],
            &[3, 4],
            &device,
        )
        .unwrap();
        let grad_out = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, -2.0, 0.5, 3.25, -1.5, 2.5, -0.75, 0.125],
            &[2, 4],
            &device,
        )
        .unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);

        let all = backward.backward_all(&grad_out).unwrap();
        let only_a = backward.backward(&grad_out, &[true, false]).unwrap();

        assert_eq!(only_a.len(), 2, "one slot per input");
        assert!(
            only_a[1].is_none(),
            "the unwanted operand's gradient must not be produced"
        );

        let want = all[0].as_ref().expect("all-true grad for A");
        let got = only_a[0].as_ref().expect("masked grad for A");
        assert_eq!(bits(want), bits(got), "wanted gradient must not shift");
    }

    /// The mirror case: only operand B is wanted, so `dL/dC @ B^T` is skipped.
    #[test]
    fn test_matmul_backward_skips_unneeded_activation() {
        let device = CpuDevice::new();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);

        let all = backward.backward_all(&grad_out).unwrap();
        let only_b = backward.backward(&grad_out, &[false, true]).unwrap();

        assert!(only_b[0].is_none(), "A's gradient must not be produced");

        let want = all[1].as_ref().expect("all-true grad for B");
        let got = only_b[1].as_ref().expect("masked grad for B");
        assert_eq!(bits(want), bits(got));
    }

    /// An all-false mask produces nothing and touches no operand.
    #[test]
    fn test_matmul_backward_all_false_produces_nothing() {
        let device = CpuDevice::new();

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);

        let grads = backward.backward(&grad_out, &[false, false]).unwrap();
        assert_eq!(grads.len(), 2);
        assert!(grads.iter().all(|g| g.is_none()));
    }

    #[test]
    fn test_matmul_backward_2x2() {
        let device = CpuDevice::new();

        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        // Assume dL/dC = [[1, 1], [1, 1]] (ones)
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = dL/dC @ B^T
        // B^T = [[5, 7], [6, 8]]
        // dL/dA = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![11.0, 15.0, 11.0, 15.0]);

        // dL/dB = A^T @ dL/dC
        // A^T = [[1, 3], [2, 4]]
        // dL/dB = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        let grad_b: Vec<f32> = grads[1].as_ref().unwrap().to_vec();
        assert_eq!(grad_b, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_matmul_backward_non_square() {
        let device = CpuDevice::new();

        // A = [[1, 2, 3]] (1x3)
        // B = [[4], [5], [6]] (3x1)
        // C = A @ B = [[32]] (1x1)
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3, 1], &device).unwrap();

        // dL/dC = [[1]]
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 1], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = dL/dC @ B^T = [[1]] @ [[4, 5, 6]] = [[4, 5, 6]]
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[1, 3]);
        let grad_a_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_a_data, vec![4.0, 5.0, 6.0]);

        // dL/dB = A^T @ dL/dC = [[1], [2], [3]] @ [[1]] = [[1], [2], [3]]
        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[3, 1]);
        let grad_b_data: Vec<f32> = grad_b.to_vec();
        assert_eq!(grad_b_data, vec![1.0, 2.0, 3.0]);
    }

    /// A: [2,3,4,5] @ B: [2,1,5,6] -> [2,3,4,6].
    /// B is broadcast over the MIDDLE batch dim, so dL/dB must be summed back
    /// to [2,1,5,6] rather than left at the output's [2,3,5,6].
    #[test]
    fn test_matmul_backward_broadcast_middle_batch_dim_shapes() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::ones(&[2, 3, 4, 5], DType::F32, &device).unwrap();
        let b = Tensor::<CpuRuntime>::ones(&[2, 1, 5, 6], DType::F32, &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 3, 4, 6], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = dL/dC @ B^T: each entry sums over N=6 ones.
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3, 4, 5]);
        let grad_a_data: Vec<f32> = grad_a.contiguous().unwrap().to_vec();
        assert!(grad_a_data.iter().all(|&v| v == 6.0));

        // dL/dB = A^T @ dL/dC, reduced over the broadcast dim 1 (extent 3).
        // Each entry sums over M=4 ones, then over the 3 broadcast slices: 4*3 = 12.
        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[2, 1, 5, 6]);
        let grad_b_data: Vec<f32> = grad_b.contiguous().unwrap().to_vec();
        assert!(grad_b_data.iter().all(|&v| v == 12.0));
    }

    /// Small hand-computed middle-batch-dim broadcast case.
    ///
    /// A: [2,2,1,2] with rows
    ///   A[0,0]=[1,2]  A[0,1]=[3,4]  A[1,0]=[5,6]  A[1,1]=[7,8]
    /// B: [2,1,2,1] with columns B[0]=[10,20]^T, B[1]=[30,40]^T
    /// grad_out: ones [2,2,1,1]
    ///
    /// dL/dA[i,j] = grad_out * B[i]^T = B[i]^T (no reduction; A is not broadcast)
    /// dL/dB[i]   = sum_j A[i,j]^T * grad_out = A[i,0]^T + A[i,1]^T
    ///            = [1+3, 2+4] = [4,6] and [5+7, 6+8] = [12,14]
    #[test]
    fn test_matmul_backward_broadcast_middle_batch_dim_values() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 1, 2],
            &device,
        )
        .unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 1, 2, 1], &device)
                .unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2, 1, 1], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 2, 1, 2]);
        let grad_a_data: Vec<f32> = grad_a.contiguous().unwrap().to_vec();
        assert_eq!(
            grad_a_data,
            vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
        );

        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[2, 1, 2, 1]);
        let grad_b_data: Vec<f32> = grad_b.contiguous().unwrap().to_vec();
        assert_eq!(grad_b_data, vec![4.0, 6.0, 12.0, 14.0]);
    }

    /// Same broadcast case through the Var path, which must mirror `backward`.
    #[test]
    fn test_matmul_backward_var_broadcast_middle_batch_dim() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 1, 2],
            &device,
        )
        .unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 1, 2, 1], &device)
                .unwrap();
        let grad_out = Var::new(
            Tensor::<CpuRuntime>::ones(&[2, 2, 1, 1], DType::F32, &device).unwrap(),
            true,
        );

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_var(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 2, 1, 2]);
        let grad_a_data: Vec<f32> = grad_a.tensor().contiguous().unwrap().to_vec();
        assert_eq!(
            grad_a_data,
            vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
        );

        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[2, 1, 2, 1]);
        let grad_b_data: Vec<f32> = grad_b.tensor().contiguous().unwrap().to_vec();
        assert_eq!(grad_b_data, vec![4.0, 6.0, 12.0, 14.0]);
    }

    /// Leading-batch-dim broadcast: A: [2,1,3] @ B: [1,3,2] -> [2,1,2].
    #[test]
    fn test_matmul_backward_broadcast_leading_batch_dim() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 1, 3],
            &device,
        )
        .unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 3, 2],
            &device,
        )
        .unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = ones[1,2] @ B^T[2,3]; row sums of B: [3, 7, 11], same for both batches.
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 1, 3]);
        let grad_a_data: Vec<f32> = grad_a.contiguous().unwrap().to_vec();
        assert_eq!(grad_a_data, vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);

        // dL/dB = A^T @ ones, summed over the broadcast batch dim:
        // (A[0] + A[1]) broadcast across N=2 -> [[5,5],[7,7],[9,9]]
        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[1, 3, 2]);
        let grad_b_data: Vec<f32> = grad_b.contiguous().unwrap().to_vec();
        assert_eq!(grad_b_data, vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }
}
