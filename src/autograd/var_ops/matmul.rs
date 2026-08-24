//! Matrix multiplication operations

use super::macros::*;
use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};

impl_var_binary_op_tensors!(
    /// Matrix multiplication: z = a @ b
    ///
    /// Creates MatmulBackward for gradient computation.
    var_matmul, matmul, MatmulBackward
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_matmul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2, 1], &device).unwrap(),
            true,
        );

        let c = var_matmul(&a, &b, &client).unwrap();

        let c_data: Vec<f32> = c.tensor().to_vec();
        assert_eq!(c_data, vec![11.0]);

        let grads = backward(&c, &client).unwrap();

        let grad_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        assert_eq!(grad_a, vec![3.0, 4.0]);

        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert_eq!(grad_b, vec![1.0, 2.0]);
    }
}
