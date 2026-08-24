//! Linear algebra operations (trace, inverse, det, solve, cholesky)

use super::ops::*;
use crate::algorithm::LinearAlgebraAlgorithms;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Matrix trace: scalar = tr(A)
///
/// Creates TraceBackward for gradient computation.
pub fn var_trace<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.trace(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = TraceBackward::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Matrix inverse: B = A^{-1}
///
/// Creates InverseBackward for gradient computation.
pub fn var_inverse<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R>,
{
    let output = client.inverse(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = InverseBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Matrix determinant: scalar = det(A)
///
/// Creates DetBackward for gradient computation.
pub fn var_det<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.det(a.tensor())?;

    if a.requires_grad() {
        // Save the output tensor for the backward pass (preserves dtype)
        let grad_fn = DetBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(), // Save output tensor instead of f64 value
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Linear system solve: x = solve(A, b) where Ax = b
///
/// Creates SolveBackward for gradient computation.
pub fn var_solve<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + LinearAlgebraAlgorithms<R>,
{
    let output = client.solve(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = SolveBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Cholesky decomposition: L = cholesky(A) where A = L @ L^T
///
/// Creates CholeskyBackward for gradient computation.
pub fn var_cholesky<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + LinearAlgebraAlgorithms<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    let decomp = client.cholesky_decompose(a.tensor())?;
    let output = decomp.l;

    if a.requires_grad() {
        let grad_fn = CholeskyBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
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
    fn test_var_trace_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[1, 2], [3, 4]], tr(A) = 5
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
            true,
        );

        let trace = var_trace(&a, &client).unwrap();
        let grads = backward(&trace, &client).unwrap();

        // dL/dA = I (identity matrix)
        let grad_a: Vec<f64> = grads.get(a.id()).unwrap().to_vec();
        assert!((grad_a[0] - 1.0).abs() < 1e-6); // (0,0)
        assert!((grad_a[1] - 0.0).abs() < 1e-6); // (0,1)
        assert!((grad_a[2] - 0.0).abs() < 1e-6); // (1,0)
        assert!((grad_a[3] - 1.0).abs() < 1e-6); // (1,1)
    }

    #[test]
    fn test_var_inverse_backward() {
        use super::super::reduce;
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]]
        // A^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap(),
            true,
        );

        let inv_a = var_inverse(&a, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let sum_inv = reduce::var_sum(&inv_a, &[0, 1], false, &client).unwrap();
        let grads = backward(&sum_inv, &client).unwrap();

        // Verify gradient has correct shape
        let grad_a = grads.get(a.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // Analytical gradient: dL/dA = -B^T @ dL/dB @ B^T where B = A^{-1}
        // With dL/dB = ones [2,2]:
        // B^T @ ones = [[1/3, 1/3], [1/3, 1/3]]
        // [[1/3, 1/3], [1/3, 1/3]] @ B^T = [[1/9, 1/9], [1/9, 1/9]]
        // After negation: dL/dA = [[-1/9, -1/9], [-1/9, -1/9]]
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        let expected = -1.0 / 9.0;
        assert!(
            (grad_a_data[0] - expected).abs() < 1e-6,
            "grad[0,0] = {}, expected {}",
            grad_a_data[0],
            expected
        );
        assert!(
            (grad_a_data[1] - expected).abs() < 1e-6,
            "grad[0,1] = {}, expected {}",
            grad_a_data[1],
            expected
        );
        assert!(
            (grad_a_data[2] - expected).abs() < 1e-6,
            "grad[1,0] = {}, expected {}",
            grad_a_data[2],
            expected
        );
        assert!(
            (grad_a_data[3] - expected).abs() < 1e-6,
            "grad[1,1] = {}, expected {}",
            grad_a_data[3],
            expected
        );
    }

    #[test]
    fn test_var_det_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]], det(A) = 3
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap(),
            true,
        );

        let det = var_det(&a, &client).unwrap();
        let grads = backward(&det, &client).unwrap();

        // dL/dA = det(A) * A^{-T} = 3 * [[2/3, -1/3], [-1/3, 2/3]] = [[2, -1], [-1, 2]]
        let grad_a: Vec<f64> = grads.get(a.id()).unwrap().to_vec();
        assert!((grad_a[0] - 2.0).abs() < 1e-6);
        assert!((grad_a[1] - (-1.0)).abs() < 1e-6);
        assert!((grad_a[2] - (-1.0)).abs() < 1e-6);
        assert!((grad_a[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_solve_backward() {
        use super::super::reduce;
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]], b = [[3], [3]]
        // x = solve(A, b) = [[1], [1]]
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f64, 3.0], &[2, 1], &device).unwrap(),
            true,
        );

        let x = var_solve(&a, &b, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let loss = reduce::var_sum(&x, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // Verify gradients have correct shapes
        let grad_a = grads.get(a.id()).unwrap();
        let grad_b = grads.get(b.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_b.shape(), &[2, 1]);

        // dL/db = solve(A^T, dL/dx)
        // For symmetric A and dL/dx = [[1], [1]]: v = A^{-1} @ [[1], [1]] = [[1/3], [1/3]]
        let grad_b_data: Vec<f64> = grad_b.to_vec();
        assert!(
            (grad_b_data[0] - 1.0 / 3.0).abs() < 1e-6,
            "grad_b[0] = {}, expected {}",
            grad_b_data[0],
            1.0 / 3.0
        );
        assert!(
            (grad_b_data[1] - 1.0 / 3.0).abs() < 1e-6,
            "grad_b[1] = {}, expected {}",
            grad_b_data[1],
            1.0 / 3.0
        );

        // dL/dA = -v @ x^T where x = [[1], [1]]
        // = -[[1/3], [1/3]] @ [[1, 1]] = -[[1/3, 1/3], [1/3, 1/3]]
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        let expected_a = -1.0 / 3.0;
        assert!(
            (grad_a_data[0] - expected_a).abs() < 1e-6,
            "grad_a[0,0] = {}, expected {}",
            grad_a_data[0],
            expected_a
        );
        assert!(
            (grad_a_data[1] - expected_a).abs() < 1e-6,
            "grad_a[0,1] = {}, expected {}",
            grad_a_data[1],
            expected_a
        );
        assert!(
            (grad_a_data[2] - expected_a).abs() < 1e-6,
            "grad_a[1,0] = {}, expected {}",
            grad_a_data[2],
            expected_a
        );
        assert!(
            (grad_a_data[3] - expected_a).abs() < 1e-6,
            "grad_a[1,1] = {}, expected {}",
            grad_a_data[3],
            expected_a
        );
    }

    #[test]
    fn test_var_cholesky_backward() {
        use super::super::reduce;
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[4, 2], [2, 5]] (positive definite)
        let a_data = [4.0f64, 2.0, 2.0, 5.0];
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device).unwrap(),
            true,
        );

        let l = var_cholesky(&a, &client).unwrap();
        // Reduce to scalar for backward - sum over all dimensions
        let loss = reduce::var_sum(&l, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // Verify gradient has correct shape
        let grad_a = grads.get(a.id()).unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // Gradient should be symmetric (since A is symmetric)
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        assert!(
            (grad_a_data[1] - grad_a_data[2]).abs() < 1e-10,
            "grad_a[0,1] = {}, grad_a[1,0] = {}",
            grad_a_data[1],
            grad_a_data[2]
        );

        // Verify gradient using finite differences
        let eps = 1e-5;
        for idx in 0..4 {
            // Perturb A[idx] by +eps and -eps (symmetric perturbation)
            let mut a_plus = a_data;
            let mut a_minus = a_data;
            a_plus[idx] += eps;
            a_minus[idx] -= eps;

            // For symmetric matrix, also perturb transpose element
            let i = idx / 2;
            let j = idx % 2;
            if i != j {
                let t_idx = j * 2 + i;
                a_plus[t_idx] += eps;
                a_minus[t_idx] -= eps;
            }

            // Compute loss for perturbed matrices
            let a_plus_t = Tensor::<CpuRuntime>::from_slice(&a_plus, &[2, 2], &device).unwrap();
            let a_minus_t = Tensor::<CpuRuntime>::from_slice(&a_minus, &[2, 2], &device).unwrap();

            let l_plus = client.cholesky_decompose(&a_plus_t).unwrap().l;
            let l_minus = client.cholesky_decompose(&a_minus_t).unwrap().l;

            let loss_plus: f64 = l_plus.to_vec::<f64>().iter().sum();
            let loss_minus: f64 = l_minus.to_vec::<f64>().iter().sum();

            // Finite difference gradient
            let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

            // For off-diagonal elements, gradient applies to both (i,j) and (j,i)
            let expected_grad = if i != j {
                grad_a_data[idx] * 2.0 // Both elements contribute
            } else {
                grad_a_data[idx]
            };

            assert!(
                (fd_grad - expected_grad).abs() < 1e-4,
                "Finite diff gradient at ({},{}) = {}, analytical = {}",
                i,
                j,
                fd_grad,
                expected_grad
            );
        }
    }
}
