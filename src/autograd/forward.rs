//! Forward-mode automatic differentiation (Jacobian-Vector Products)
//!
//! This module provides forward-mode AD for computing Jacobian-vector products (JVP)
//! in a single forward pass. Given a function f: ℝⁿ → ℝᵐ, input x, and tangent vector v,
//! forward-mode computes (f(x), J(x) @ v) where J is the Jacobian matrix.
//!
//! # When to Use Forward vs Reverse Mode
//!
//! - **Forward-mode (JVP)**: Efficient when output dimension m >> input dimension n
//!   - Computing one column of the Jacobian at a time
//!   - Directional derivatives
//!   - Newton-Krylov methods (need J @ v without forming J)
//!   - Sensitivity analysis with few inputs
//!
//! - **Reverse-mode (VJP/backprop)**: Efficient when input dimension n >> output dimension m
//!   - Training neural networks (scalar loss, many parameters)
//!   - Computing gradients of scalar functions
//!   - Computing one row of the Jacobian at a time
//!
//! # Computing Full Jacobians
//!
//! For a function f: ℝⁿ → ℝᵐ, the full Jacobian can be computed:
//! - With forward-mode: n calls to `jvp` (one per input dimension)
//! - With reverse-mode: m calls to `vjp` (one per output dimension)
//!
//! Choose based on which is smaller: n or m.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{DualTensor, jvp};
//! # use numr::autograd::dual_ops::*;
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // f(x) = sum(x²), compute directional derivative in direction v
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device)?;
//! let v = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3], &device)?;
//!
//! let (y, dy) = jvp(
//!     |inputs, c| {
//!         let x = &inputs[0];
//!         let x_sq = dual_mul(x, x, c)?;
//!         dual_sum(&x_sq, &[], false, c)
//!     },
//!     &[&x],
//!     &[&v],
//!     &client,
//! )?;
//! // y = 14.0 (1 + 4 + 9)
//! // dy = 2*1*1 + 2*2*1 + 2*3*1 = 12.0
//! # Ok::<(), numr::error::Error>(())
//! ```

use super::DualTensor;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Compute Jacobian-vector product (JVP) via forward-mode AD
///
/// Given a function f: ℝⁿ → ℝᵐ, input primals x, and tangent vectors v,
/// computes (f(x), J(x) @ v) in a single forward pass.
///
/// This is the forward-mode analog of reverse-mode's vector-Jacobian product (VJP).
/// Instead of propagating gradients backward from outputs, it propagates tangents
/// forward from inputs.
///
/// # Arguments
///
/// * `f` - Function that takes `DualTensor` inputs and returns `DualTensor` output.
///   This function should use the `dual_*` operations from `dual_ops` module.
/// * `primals` - Input tensor values (the point at which to evaluate)
/// * `tangents` - Tangent vectors (directions to differentiate along)
/// * `client` - Runtime client for tensor operations
///
/// # Returns
///
/// A tuple of:
/// * Output tensor (f(x))
/// * Output tangent (J(x) @ v)
///
/// # Panics
///
/// Panics if `primals` and `tangents` have different lengths.
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::{DualTensor, jvp};
/// # use numr::autograd::dual_ops::dual_mul;
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // Compute directional derivative of f(x) = x² at x=3 in direction v=1
/// let x = Tensor::from_slice(&[3.0f32], &[1], &device)?;
/// let v = Tensor::from_slice(&[1.0f32], &[1], &device)?;
///
/// let (y, dy) = jvp(
///     |inputs, c| {
///         let x = &inputs[0];
///         Ok(dual_mul(x, x, c)?)
///     },
///     &[&x],
///     &[&v],
///     &client,
/// )?;
/// // y = 9.0, dy = 2*3*1 = 6.0
/// # Ok::<(), numr::error::Error>(())
/// ```
pub fn jvp<R, C, F>(
    f: F,
    primals: &[&Tensor<R>],
    tangents: &[&Tensor<R>],
    client: &C,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> Result<DualTensor<R>>,
{
    assert_eq!(
        primals.len(),
        tangents.len(),
        "Number of primals ({}) must match number of tangents ({})",
        primals.len(),
        tangents.len()
    );

    // Create dual tensors from primals and tangents
    let dual_inputs: Vec<DualTensor<R>> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(p, t)| DualTensor::with_tangent((*p).clone(), (*t).clone()))
        .collect();

    // Apply the function
    let dual_output = f(&dual_inputs, client)?;

    // Extract primal and tangent from output
    let output_primal = dual_output.primal().clone();
    let output_tangent = match dual_output.tangent() {
        Some(t) => t.clone(),
        None => Tensor::zeros(
            output_primal.shape(),
            output_primal.dtype(),
            output_primal.device(),
        )?,
    };

    Ok((output_primal, output_tangent))
}

/// Compute JVP for a function with multiple outputs
///
/// Similar to `jvp`, but handles functions that return multiple tensors.
///
/// # Arguments
///
/// * `f` - Function taking `DualTensor` inputs and returning multiple `DualTensor` outputs
/// * `primals` - Input tensor values
/// * `tangents` - Tangent vectors
/// * `client` - Runtime client
///
/// # Returns
///
/// A tuple of:
/// * Vector of output tensors (f(x))
/// * Vector of output tangents (J(x) @ v for each output)
pub fn jvp_multi<R, C, F>(
    f: F,
    primals: &[&Tensor<R>],
    tangents: &[&Tensor<R>],
    client: &C,
) -> Result<(Vec<Tensor<R>>, Vec<Tensor<R>>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> Result<Vec<DualTensor<R>>>,
{
    assert_eq!(
        primals.len(),
        tangents.len(),
        "Number of primals ({}) must match number of tangents ({})",
        primals.len(),
        tangents.len()
    );

    // Create dual tensors
    let dual_inputs: Vec<DualTensor<R>> = primals
        .iter()
        .zip(tangents.iter())
        .map(|(p, t)| DualTensor::with_tangent((*p).clone(), (*t).clone()))
        .collect();

    // Apply the function
    let dual_outputs = f(&dual_inputs, client)?;

    // Extract primals and tangents
    let mut output_primals = Vec::with_capacity(dual_outputs.len());
    let mut output_tangents = Vec::with_capacity(dual_outputs.len());

    for dual in dual_outputs {
        let primal = dual.primal().clone();
        let tangent = match dual.tangent() {
            Some(t) => t.clone(),
            None => Tensor::zeros(primal.shape(), primal.dtype(), primal.device())?,
        };
        output_primals.push(primal);
        output_tangents.push(tangent);
    }

    Ok((output_primals, output_tangents))
}

/// Compute the full Jacobian matrix using forward-mode AD
///
/// For a function f: ℝⁿ → ℝᵐ, computes the full Jacobian matrix J where `J[i,j] = ∂fᵢ/∂xⱼ`.
/// This is done by calling `jvp` n times, once for each standard basis vector.
///
/// # Arguments
///
/// * `f` - Function taking a single `DualTensor` input and returning a single `DualTensor` output
/// * `x` - Input tensor (must be 1D for now)
/// * `client` - Runtime client
///
/// # Returns
///
/// Jacobian matrix of shape `[m, n]` where m is output size and n is input size.
///
/// # Example
///
/// ```ignore
/// // f(x) = `[x[0]², x[0]*x[1], x[1]²]`
/// // J = `[[2*x[0], 0], [x[1], x[0]], [0, 2*x[1]]]`
/// let x = Tensor::from_slice(&[3.0f32, 2.0], &[2], &device).unwrap();
/// let f = |x: &DualTensor<_>, c: &_| -> Result<DualTensor<_>> {
///     let x0_sq = dual_mul(&x.index(&[0..1]), &x.index(&[0..1]), c)?;
///     let x01 = dual_mul(&x.index(&[0..1]), &x.index(&[1..2]), c)?;
///     let x1_sq = dual_mul(&x.index(&[1..2]), &x.index(&[1..2]), c)?;
///     dual_cat(&[&x0_sq, &x01, &x1_sq], 0, c)
/// };
/// let jacobian = jacobian_forward(&f, &x, &client)?;
/// // jacobian.shape() = [3, 2]
/// ```
pub fn jacobian_forward<R, C, F>(f: F, x: &Tensor<R>, client: &C) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    F: Fn(&DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let n = x.numel();
    let device = x.device();
    let dtype = x.dtype();

    // We'll collect the Jacobian columns
    let mut columns: Vec<Tensor<R>> = Vec::with_capacity(n);

    for j in 0..n {
        // Create unit vector in j-th direction
        let mut v_data = vec![0.0f64; n];
        v_data[j] = 1.0;

        // Convert to appropriate dtype
        let v = match dtype {
            crate::dtype::DType::F32 => {
                let v_f32: Vec<f32> = v_data.iter().map(|&x| x as f32).collect();
                Tensor::<R>::from_slice(&v_f32, x.shape(), device)?
            }
            crate::dtype::DType::F64 => Tensor::<R>::from_slice(&v_data, x.shape(), device)?,
            _ => {
                // For other dtypes, create as f64 and rely on operations to handle
                Tensor::<R>::from_slice(&v_data, x.shape(), device)?
            }
        };

        // Create dual input
        let dual_x = DualTensor::with_tangent(x.clone(), v);

        // Compute f and its tangent
        let dual_y = f(&dual_x, client)?;

        // The tangent is the j-th column of the Jacobian
        let col = match dual_y.tangent() {
            Some(t) => t.clone(),
            None => Tensor::zeros(dual_y.shape(), dtype, device)?,
        };

        columns.push(col);
    }

    // Stack columns into Jacobian matrix [m, n]
    // Each column has shape [m], we want [m, n]
    let col_refs: Vec<&Tensor<R>> = columns.iter().collect();
    client.stack(&col_refs, 1)
}

/// Compute Hessian-vector product using forward-over-reverse mode
///
/// For a scalar function f: ℝⁿ → ℝ, computes H @ v where H is the Hessian
/// and v is the direction vector. This is done by:
/// 1. Forward-mode through the reverse-mode computation
/// 2. Or equivalently, computing d/dt[∇f(x + t*v)] at t=0
///
/// This requires second-order differentiation and is more complex than
/// simple JVP. This function provides a convenience wrapper.
///
/// # Arguments
///
/// * `grad_f` - Function that computes the gradient ∇f(x) (from reverse-mode)
/// * `x` - Point at which to compute Hessian-vector product
/// * `v` - Direction vector
/// * `client` - Runtime client
///
/// # Returns
///
/// Hessian-vector product H(x) @ v
///
/// # Note
///
/// For most uses, prefer `backward_with_graph` + `backward` for computing
/// second-order derivatives through the existing reverse-mode infrastructure.
pub fn hvp<R, C, F>(grad_f: F, x: &Tensor<R>, v: &Tensor<R>, client: &C) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    F: Fn(&DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    // Forward-mode through the gradient computation
    // If grad_f computes ∇f(x), then jvp of grad_f gives H @ v
    let (_, hvp_result) = jvp(
        |inputs, c| {
            assert_eq!(inputs.len(), 1);
            grad_f(&inputs[0], c)
        },
        &[x],
        &[v],
        client,
    )?;

    Ok(hvp_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::dual_ops::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_jvp_square() {
        let (device, client) = setup();

        // f(x) = x² at x=3, tangent v=1 → f(x)=9, df=6
        let x = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();

        let (y, dy) = jvp(
            |inputs, c| {
                let x = &inputs[0];
                dual_mul(x, x, c)
            },
            &[&x],
            &[&v],
            &client,
        )
        .unwrap();

        assert_eq!(y.to_vec::<f32>(), [9.0]);
        assert_eq!(dy.to_vec::<f32>(), [6.0]);
    }

    #[test]
    fn test_jvp_sum_squares() {
        let (device, client) = setup();

        // f(x) = sum(x²), x = [1, 2, 3], v = [1, 1, 1]
        // f(x) = 1 + 4 + 9 = 14
        // df = 2*1*1 + 2*2*1 + 2*3*1 = 2 + 4 + 6 = 12
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap();

        let (y, dy) = jvp(
            |inputs, c| {
                let x = &inputs[0];
                let x_sq = dual_mul(x, x, c)?;
                dual_sum(&x_sq, &[0], false, c)
            },
            &[&x],
            &[&v],
            &client,
        )
        .unwrap();

        assert_eq!(y.to_vec::<f32>(), [14.0]);
        assert_eq!(dy.to_vec::<f32>(), [12.0]);
    }

    #[test]
    fn test_jvp_chain_rule() {
        let (device, client) = setup();

        // f(x) = exp(x²) at x=1, v=1
        // f'(x) = exp(x²) * 2x
        // f(1) = e, f'(1) = 2e
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();

        let (y, dy) = jvp(
            |inputs, c| {
                let x = &inputs[0];
                let x_sq = dual_mul(x, x, c)?;
                dual_exp(&x_sq, c)
            },
            &[&x],
            &[&v],
            &client,
        )
        .unwrap();

        let e = std::f32::consts::E;
        assert!((y.to_vec::<f32>()[0] - e).abs() < 1e-5);
        assert!((dy.to_vec::<f32>()[0] - 2.0 * e).abs() < 1e-4);
    }

    #[test]
    fn test_jvp_two_inputs() {
        let (device, client) = setup();

        // f(x, y) = x * y at (2, 3), tangents (1, 0)
        // ∂f/∂x = y = 3, ∂f/∂y = x = 2
        // df = 3*1 + 2*0 = 3
        let x = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap();
        let vx = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let vy = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();

        let (f, df) = jvp(
            |inputs, c| {
                let x = &inputs[0];
                let y = &inputs[1];
                dual_mul(x, y, c)
            },
            &[&x, &y],
            &[&vx, &vy],
            &client,
        )
        .unwrap();

        assert_eq!(f.to_vec::<f32>(), [6.0]);
        assert_eq!(df.to_vec::<f32>(), [3.0]);
    }

    #[test]
    fn test_jvp_matmul() {
        let (device, client) = setup();

        // f(A) = A @ B where B is constant
        // A = [[1, 2]], B = [[1], [1]]
        // A @ B = [[3]]
        // dA = [[1, 0]] (tangent), dA @ B = [[1]]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2, 1], &device).unwrap();
        let da = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[1, 2], &device).unwrap();
        let db = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2, 1], &device).unwrap();

        let (y, dy) = jvp(
            |inputs, c| {
                let a = &inputs[0];
                let b = &inputs[1];
                dual_matmul(a, b, c)
            },
            &[&a, &b],
            &[&da, &db],
            &client,
        )
        .unwrap();

        assert_eq!(y.to_vec::<f32>(), [3.0]);
        assert_eq!(dy.to_vec::<f32>(), [1.0]);
    }

    #[test]
    fn test_jacobian_forward_linear() {
        let (device, client) = setup();

        // f(x) = 2*x (linear function)
        // Jacobian = diag([2, 2, 2])
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        let jacobian =
            jacobian_forward(|dual_x, c| dual_mul_scalar(dual_x, 2.0, c), &x, &client).unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f32> = jacobian.to_vec();
        // Should be diagonal with 2s
        assert!((j[0] - 2.0).abs() < 1e-6); // [0,0]
        assert!((j[1] - 0.0).abs() < 1e-6); // [0,1]
        assert!((j[2] - 0.0).abs() < 1e-6); // [0,2]
        assert!((j[3] - 0.0).abs() < 1e-6); // [1,0]
        assert!((j[4] - 2.0).abs() < 1e-6); // [1,1]
        assert!((j[5] - 0.0).abs() < 1e-6); // [1,2]
        assert!((j[6] - 0.0).abs() < 1e-6); // [2,0]
        assert!((j[7] - 0.0).abs() < 1e-6); // [2,1]
        assert!((j[8] - 2.0).abs() < 1e-6); // [2,2]
    }

    #[test]
    fn test_jvp_multi() {
        let (device, client) = setup();

        // f(x) = (x², x³)
        // At x=2: f(2) = (4, 8), df = (4, 12) for v=1
        let x = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();

        let (ys, dys) = jvp_multi(
            |inputs, c| {
                let x = &inputs[0];
                let x_sq = dual_mul(x, x, c)?;
                let x_cube = dual_mul(&x_sq, x, c)?;
                Ok(vec![x_sq, x_cube])
            },
            &[&x],
            &[&v],
            &client,
        )
        .unwrap();

        assert_eq!(ys.len(), 2);
        assert_eq!(dys.len(), 2);
        assert_eq!(ys[0].to_vec::<f32>(), [4.0]);
        assert_eq!(ys[1].to_vec::<f32>(), [8.0]);
        assert_eq!(dys[0].to_vec::<f32>(), [4.0]); // d(x²)/dx * v = 2x = 4
        assert_eq!(dys[1].to_vec::<f32>(), [12.0]); // d(x³)/dx * v = 3x² = 12
    }
}
