//! Backward implementations for linear algebra operations
//!
//! Implements gradient computation for:
//! - trace: tr(A)
//! - inverse: A^{-1}
//! - det: |A|
//! - solve: solve(A, b) where Ax = b
//! - cholesky: L where A = LL^T
//!
//! ## Performance Notes
//!
//! | Operation   | Forward    | Backward   | Notes                           |
//! |-------------|------------|------------|---------------------------------|
//! | trace       | O(N)       | O(N²)      | Creates identity matrix         |
//! | inverse     | O(N³)      | O(N³)      | Two matmuls                     |
//! | determinant | O(N³)      | O(N³)      | Requires inverse                |
//! | solve       | O(N³)      | O(N³)      | One solve + one matmul          |
//! | cholesky    | O(N³)      | O(N³)      | Two triangular solves + matmul  |

use crate::algorithm::LinearAlgebraAlgorithms;
use crate::autograd::var_ops::{var_matmul, var_mul, var_neg};
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    BinaryOps, LinalgOps, MatmulOps, ScalarOps, TensorOps, TypeConversionOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract lower triangular part with diagonal halved: Φ(X) = tril(X) with diag/2
///
/// This operation preserves the input tensor's dtype by using tensor operations
/// rather than extracting to host memory.
///
/// # Arguments
/// * `x` - Input square matrix (must be contiguous)
/// * `client` - Runtime client for tensor operations
///
/// # Returns
/// A tensor where upper triangular elements are zero, lower triangular elements
/// are unchanged, and diagonal elements are halved.
fn tril_with_halved_diagonal<R: Runtime<DType = DType>>(
    x: &Tensor<R>,
    client: &R::Client,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let n = x.shape()[0];
    debug_assert_eq!(x.shape().len(), 2);
    debug_assert_eq!(x.shape()[0], x.shape()[1]);

    // Create combined mask: 1.0 for strict lower triangle, 0.5 for diagonal, 0.0 for upper
    // This is more efficient than applying two separate masks (1 allocation + 1 pass vs 2 + 2)
    let mut mask_data = vec![0.0f64; n * n];
    for i in 0..n {
        // Strict lower triangle: j < i
        for j in 0..i {
            mask_data[i * n + j] = 1.0;
        }
        // Diagonal: halved
        mask_data[i * n + i] = 0.5;
        // Upper triangle (j > i): stays 0.0
    }

    // Create mask in F64 first, then cast to input dtype
    let mask_f64 = Tensor::<R>::from_slice(&mask_data, &[n, n], x.device())?;
    let mask = client.cast(&mask_f64, x.dtype())?;

    // Apply mask: result = x * mask (single multiplication)
    let phi = client.mul(x, &mask)?;

    Ok(phi)
}

// ============================================================================
// TraceBackward
// ============================================================================

/// Backward for trace: scalar = tr(A)
///
/// For scalar s = tr(A):
/// - dL/dA = dL/ds * I (identity matrix scaled by upstream gradient)
pub struct TraceBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    saved_tensors: Vec<Tensor<R>>, // [A] - need shape and device
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 1],
}

impl<R: Runtime> TraceBackward<R> {
    /// Create a new TraceBackward.
    pub fn new(a_id: TensorId, a: Tensor<R>, a_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_ids: [a_id],
            saved_tensors: vec![a],
            input_grad_fns: [a_grad_fn],
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for TraceBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let saved_a = &self.saved_tensors[0];
        let n = saved_a.shape()[0];
        let client = R::default_client(saved_a.device());

        // grad_output is a scalar tensor (shape [] or [1])
        // dL/dA = dL/ds * I
        //
        // We avoid GPU→CPU→GPU data movement by keeping all operations on device:
        // 1. Create ones vector [1, 1, ..., 1] of length n on device
        // 2. Multiply by grad_output scalar (broadcasts to all elements)
        // 3. Create diagonal matrix from the scaled vector

        // Create ones vector with same dtype as input
        let ones_vec = Tensor::<R>::ones(&[n], saved_a.dtype(), saved_a.device())?;

        // Scale by grad_output via broadcasting (stays on device)
        let scaled_diag = client.mul(&ones_vec, grad_output)?;

        // Create diagonal matrix (identity scaled by grad_output)
        let eye = LinalgOps::diagflat(&client, &scaled_diag)?;

        Ok(vec![Some(eye)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
    {
        let saved_a = &self.saved_tensors[0];
        let n = saved_a.shape()[0];
        let client = R::default_client(saved_a.device());

        // dL/dA = dL/ds * I
        // Create ones vector (constant, no gradient tracking)
        let ones_vec = Tensor::<R>::ones(&[n], saved_a.dtype(), saved_a.device())?;
        let ones_var = Var::new(ones_vec, false);

        // Scale by grad_output via var_mul to track gradients through grad_output
        let scaled_diag = var_mul(&ones_var, grad_output, &client)?;

        // Create diagonal matrix (non-differentiable operation on the result)
        let eye = LinalgOps::diagflat(&client, scaled_diag.tensor())?;

        Ok(vec![Some(Var::new(eye, false))])
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
        "TraceBackward"
    }
}

// ============================================================================
// InverseBackward
// ============================================================================

/// Backward for matrix inverse: B = A^{-1}
///
/// For B = A^{-1}:
/// - dL/dA = -B^T @ dL/dB @ B^T
pub struct InverseBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    saved_tensors: Vec<Tensor<R>>, // [inv_A] - save the inverse output
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 1],
}

impl<R: Runtime> InverseBackward<R> {
    /// Create a new InverseBackward.
    pub fn new(a_id: TensorId, inv_a: Tensor<R>, a_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_ids: [a_id],
            saved_tensors: vec![inv_a],
            input_grad_fns: [a_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for InverseBackward<R>
where
    R::Client: MatmulOps<R> + TensorOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let inv_a = &self.saved_tensors[0];

        // dL/dA = -B^T @ dL/dB @ B^T where B = A^{-1}
        let inv_a_t = inv_a.t()?;
        let temp = client.matmul(&inv_a_t, grad_output)?;
        let grad_a = client.matmul(&temp, &inv_a_t)?;

        // Negate
        let grad_a = client.neg(&grad_a)?;

        Ok(vec![Some(grad_a)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let inv_a = &self.saved_tensors[0];

        // dL/dA = -B^T @ dL/dB @ B^T where B = A^{-1}
        // The inverse B is saved from forward pass, treat as constant
        let inv_a_t = inv_a.t()?;
        let inv_a_t_var = Var::new(inv_a_t, false);

        // temp = B^T @ dL/dB (track gradients through grad_output)
        let temp = var_matmul(&inv_a_t_var, grad_output, &client)?;

        // grad_a = temp @ B^T (track gradients through temp)
        let inv_a_t2 = inv_a.t()?;
        let inv_a_t_var2 = Var::new(inv_a_t2, false);
        let grad_a = var_matmul(&temp, &inv_a_t_var2, &client)?;

        // Negate (track gradients through neg)
        let grad_a = var_neg(&grad_a, &client)?;

        Ok(vec![Some(grad_a)])
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
        "InverseBackward"
    }
}

// ============================================================================
// DetBackward
// ============================================================================

/// Backward for determinant: scalar = det(A)
///
/// For scalar d = det(A):
/// - dL/dA = dL/dd * det(A) * A^{-T}
pub struct DetBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    saved_tensors: Vec<Tensor<R>>, // [A, det_output] - need A for inverse, det for scaling
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 1],
}

impl<R: Runtime> DetBackward<R> {
    /// Create a new DetBackward.
    ///
    /// # Arguments
    /// * `a_id` - TensorId of the input matrix
    /// * `a` - The input matrix A (saved for computing inverse)
    /// * `det_output` - The determinant output tensor (saved for scaling)
    /// * `a_grad_fn` - Gradient function of the input
    pub fn new(
        a_id: TensorId,
        a: Tensor<R>,
        det_output: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id],
            saved_tensors: vec![a, det_output],
            input_grad_fns: [a_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for DetBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        use crate::error::Error;

        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let det_output = &self.saved_tensors[1];

        // dL/dA = dL/dd * det(A) * A^{-T}
        // = grad_output * det_output * (A^{-1})^T
        //
        // Note: If det(A) ≈ 0, the matrix is singular and inverse will fail.
        // This is expected behavior - determinant gradient is undefined for singular matrices.
        let inv_a = LinalgOps::inverse(&client, saved_a).map_err(|e| {
            Error::Internal(format!(
                "DetBackward: failed to compute inverse for gradient \
                 (matrix may be singular or nearly singular): {}",
                e
            ))
        })?;
        let inv_a_t = inv_a.t()?.contiguous()?;

        // Scale: det_output * inv_a_t (element-wise broadcast)
        // det_output is scalar [1] or [], inv_a_t is [n, n]
        let det_scaled = client.mul(&inv_a_t, det_output)?;

        // Scale by grad_output (upstream gradient)
        let grad_a = client.mul(&det_scaled, grad_output)?;

        Ok(vec![Some(grad_a)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
    {
        use crate::error::Error;

        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let det_output = &self.saved_tensors[1];

        // dL/dA = dL/dd * det(A) * A^{-T}
        // Recompute inverse (same as in backward)
        let inv_a = LinalgOps::inverse(&client, saved_a).map_err(|e| {
            Error::Internal(format!(
                "DetBackward: failed to compute inverse for gradient \
                 (matrix may be singular or nearly singular): {}",
                e
            ))
        })?;
        let inv_a_t = inv_a.t()?.contiguous()?;

        // det_output * inv_a_t is constant (not dependent on grad_output)
        let det_scaled = client.mul(&inv_a_t, det_output)?;
        let det_scaled_var = Var::new(det_scaled, false);

        // Scale by grad_output using var_mul to track gradients
        let grad_a = var_mul(&det_scaled_var, grad_output, &client)?;

        Ok(vec![Some(grad_a)])
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
        "DetBackward"
    }
}

// ============================================================================
// SolveBackward
// ============================================================================

/// Backward for solve: x = solve(A, b) where Ax = b
///
/// For x = A^{-1}b:
/// - dL/dA = -solve(A^T, dL/dx) @ x^T
/// - dL/db = solve(A^T, dL/dx)
pub struct SolveBackward<R: Runtime> {
    input_ids: [TensorId; 2],      // [A_id, b_id]
    saved_tensors: Vec<Tensor<R>>, // [A, x] - need A for solve, x for gradient
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> SolveBackward<R> {
    /// Create a new SolveBackward.
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        x: Tensor<R>, // The solution
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, x],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for SolveBackward<R>
where
    R::Client: MatmulOps<R> + TensorOps<R> + LinearAlgebraAlgorithms<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // Both slots share the triangular-ish solve for `v`, but dL/dA costs a
        // further matmul and negation on top, so that slot is guarded.
        if !needed.iter().any(|&n| n) {
            return Ok(vec![None, None]);
        }

        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_x = &self.saved_tensors[1];

        // grad_output = dL/dx
        // Solve A^T @ v = dL/dx for v
        let a_t = saved_a.t()?.contiguous()?;
        let v = LinalgOps::solve(&client, &a_t, grad_output)?;

        // dL/dA = -v @ x^T
        let grad_a = if needed[0] {
            let x_t = saved_x.t()?;
            let grad_a = client.matmul(&v, &x_t)?;
            Some(client.neg(&grad_a)?)
        } else {
            None
        };

        // dL/db = v = solve(A^T, dL/dx)
        let grad_b = if needed[1] { Some(v) } else { None };

        Ok(vec![grad_a, grad_b])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + MatmulOps<R> + TensorOps<R> + LinearAlgebraAlgorithms<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let saved_x = &self.saved_tensors[1];

        // grad_output = dL/dx
        // Solve A^T @ v = dL/dx for v
        // The solve operation is not differentiable in the var sense here,
        // so we compute v as a constant tensor
        let a_t = saved_a.t()?.contiguous()?;
        let v = LinalgOps::solve(&client, &a_t, grad_output.tensor())?;

        // dL/db = v = solve(A^T, dL/dx)
        // v depends on grad_output, but solve is not tracked, so grad_b is constant
        let grad_b = Var::new(v.clone(), false);

        // dL/dA = -v @ x^T
        // v is constant (result of solve), x^T is constant (saved tensor)
        let v_var = Var::new(v, false);
        let x_t = saved_x.t()?;
        let x_t_var = Var::new(x_t, false);

        // Use var_matmul (though both inputs are constant here)
        let grad_a = var_matmul(&v_var, &x_t_var, &client)?;

        // Negate using var_neg
        let grad_a = var_neg(&grad_a, &client)?;

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
        "SolveBackward"
    }
}

// ============================================================================
// CholeskyBackward
// ============================================================================

/// Backward for Cholesky decomposition: L = cholesky(A) where A = L @ L^T
///
/// The gradient is computed using the formula from:
/// "Differentiation of the Cholesky decomposition" - Murray 2016
///
/// For symmetric positive definite A with Cholesky factor L:
///
/// ```text
/// grad_A = symmetrize(L^{-T} @ Φ(L^T @ grad_L) @ L^{-1})
/// ```
///
/// Implementation steps:
/// 1. Compute S = L^T @ grad_L (matmul)
/// 2. Apply Φ(S) = tril(S) with diagonal halved
/// 3. Compute Z = Φ @ L^{-1} via solve: L^T @ W = Φ^T, then Z = W^T
/// 4. Compute Y = L^{-T} @ Z via solve: L^T @ Y = Z
/// 5. Symmetrize: grad_A = (Y + Y^T) / 2
///
/// Where Φ extracts the lower triangular part and halves the diagonal.
pub struct CholeskyBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    saved_tensors: Vec<Tensor<R>>, // [L] - the Cholesky factor
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 1],
}

impl<R: Runtime> CholeskyBackward<R> {
    /// Create a new CholeskyBackward.
    pub fn new(a_id: TensorId, l: Tensor<R>, a_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_ids: [a_id],
            saved_tensors: vec![l],
            input_grad_fns: [a_grad_fn],
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for CholeskyBackward<R>
where
    R::Client: MatmulOps<R> + TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        use crate::error::Error;

        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let l = &self.saved_tensors[0];

        // Step 1: Compute S = L^T @ grad_L
        // Ensure L^T is contiguous for matmul
        let l_t = l.t()?.contiguous()?;
        let s = client.matmul(&l_t, grad_output)?;

        // Step 2: Φ(S) = tril(S) with diagonal halved
        // Use tensor operations to preserve dtype
        let phi = tril_with_halved_diagonal::<R>(&s.contiguous()?, &client)?;

        // Step 3: Compute Z = Φ @ L^{-1}
        // Solve L^T @ W = Φ^T for W, then Z = W^T
        // (because L^T @ W = Φ^T implies W = L^{-T} @ Φ^T, so W^T = Φ @ L^{-1})
        let phi_t = phi.t()?.contiguous()?;
        let w = client.solve_triangular_upper(&l_t, &phi_t).map_err(|e| {
            Error::Internal(format!(
                "CholeskyBackward: triangular solve failed in step 3 \
                 (L may have zero diagonal elements): {}",
                e
            ))
        })?;
        let z = w.t()?.contiguous()?;

        // Step 4: Compute grad_A = L^{-T} @ Z
        // Solve L^T @ Y = Z for Y (gives Y = L^{-T} @ Z = L^{-T} @ Φ @ L^{-1})
        let grad_a_raw = client.solve_triangular_upper(&l_t, &z).map_err(|e| {
            Error::Internal(format!(
                "CholeskyBackward: triangular solve failed in step 4 \
                 (L may have zero diagonal elements): {}",
                e
            ))
        })?;

        // Step 5: Symmetrize the result: grad_A = (Y + Y^T) / 2
        // Use tensor operations to preserve dtype
        let y_contiguous = grad_a_raw.contiguous()?;
        let y_t = y_contiguous.t()?.contiguous()?;
        let sum = client.add(&y_contiguous, &y_t)?;
        let grad_a = client.div_scalar(&sum, 2.0)?;

        Ok(vec![Some(grad_a)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + MatmulOps<R>
            + TensorOps<R>
            + ScalarOps<R>
            + LinearAlgebraAlgorithms<R>,
    {
        use crate::error::Error;

        let client = R::default_client(grad_output.tensor().device());
        let l = &self.saved_tensors[0];

        // Step 1: Compute S = L^T @ grad_L
        // L is saved (constant), grad_output needs gradient tracking
        let l_t = l.t()?.contiguous()?;
        let l_t_var = Var::new(l_t.clone(), false);
        let s = var_matmul(&l_t_var, grad_output, &client)?;

        // Step 2: Φ(S) = tril(S) with diagonal halved
        // This is a non-differentiable mask operation
        let phi = tril_with_halved_diagonal::<R>(&s.tensor().contiguous()?, &client)?;

        // Step 3-5: The triangular solves and symmetrization
        // These don't have var_ops equivalents, so we compute them as constants
        let phi_t = phi.t()?.contiguous()?;
        let w = client.solve_triangular_upper(&l_t, &phi_t).map_err(|e| {
            Error::Internal(format!(
                "CholeskyBackward: triangular solve failed in step 3: {}",
                e
            ))
        })?;
        let z = w.t()?.contiguous()?;

        let grad_a_raw = client.solve_triangular_upper(&l_t, &z).map_err(|e| {
            Error::Internal(format!(
                "CholeskyBackward: triangular solve failed in step 4: {}",
                e
            ))
        })?;

        // Step 5: Symmetrize: grad_A = (Y + Y^T) / 2
        let y_contiguous = grad_a_raw.contiguous()?;
        let y_t = y_contiguous.t()?.contiguous()?;
        let sum = client.add(&y_contiguous, &y_t)?;
        let grad_a = client.div_scalar(&sum, 2.0)?;

        // The result depends on grad_output through the chain,
        // but triangular solves are not tracked, so wrap as constant
        Ok(vec![Some(Var::new(grad_a, false))])
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
        "CholeskyBackward"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_eq_vec(a: &[f64], b: &[f64], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn test_trace_backward() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        // A = [[1, 2], [3, 4]]
        // tr(A) = 5
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        // dL/dtr = 1 (scalar gradient, 0-dim tensor)
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[], &device).unwrap();

        let backward = TraceBackward::<CpuRuntime>::new(a.id(), a.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA should be identity matrix
        let grad_a: Vec<f64> = grads[0].as_ref().unwrap().to_vec();
        assert!(approx_eq_vec(&grad_a, &[1.0, 0.0, 0.0, 1.0], 1e-10));
    }

    #[test]
    fn test_inverse_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]]
        // A^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap();
        let inv_a = LinalgOps::inverse(&client, &a).unwrap();

        // dL/dB = ones (gradient w.r.t. inverse)
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F64, &device).unwrap();

        let backward = InverseBackward::<CpuRuntime>::new(a.id(), inv_a.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = -B^T @ dL/dB @ B^T
        // Verify shape is correct
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // Numerical check: gradient should be -B^T @ ones @ B^T
        // B^T @ ones = sum of rows of B^T = sum of columns of B
        // For our B: column sums are [1/3, 1/3]
        // B^T @ [1/3, 1/3]^T @ B^T = ...
        // The gradient should be negative definite
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        // Just verify it's non-zero and has correct sign pattern
        assert!(grad_a_data[0] < 0.0); // Diagonal should be negative
    }

    #[test]
    fn test_det_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]]
        // det(A) = 3
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap();
        let det_output = LinalgOps::det(&client, &a).unwrap(); // Returns tensor with value 3.0

        // dL/ddet = 1 (scalar gradient, 0-dim tensor)
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[], &device).unwrap();

        let backward = DetBackward::<CpuRuntime>::new(a.id(), a.clone(), det_output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // dL/dA = det(A) * A^{-T}
        // A^{-1} = [[2/3, -1/3], [-1/3, 2/3]]
        // det(A) * A^{-T} = 3 * [[2/3, -1/3], [-1/3, 2/3]] = [[2, -1], [-1, 2]]
        let grad_a: Vec<f64> = grads[0].as_ref().unwrap().to_vec();
        assert!(approx_eq_vec(&grad_a, &[2.0, -1.0, -1.0, 2.0], 1e-10));
    }

    #[test]
    fn test_solve_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[2, 1], [1, 2]], b = [[3], [3]]
        // x = solve(A, b) = [[1], [1]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 3.0], &[2, 1], &device).unwrap();
        let x = LinalgOps::solve(&client, &a, &b).unwrap();

        // dL/dx = [[1], [1]]
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2, 1], &device).unwrap();

        let backward =
            SolveBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), x.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // Verify shapes
        let grad_a = grads[0].as_ref().unwrap();
        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);
        assert_eq!(grad_b.shape(), &[2, 1]);

        // dL/db = solve(A^T, dL/dx)
        // A^T @ v = dL/dx, v = dL/db
        // For symmetric A: v = A^{-1} @ [1, 1]^T = [[1/3], [1/3]]
        let grad_b_data: Vec<f64> = grad_b.to_vec();
        assert!(approx_eq_vec(&grad_b_data, &[1.0 / 3.0, 1.0 / 3.0], 1e-10));
    }

    #[test]
    fn test_cholesky_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A = [[4, 2], [2, 5]] (positive definite)
        // L = cholesky(A) = [[2, 0], [1, 2]]
        let a =
            Tensor::<CpuRuntime>::from_slice(&[4.0f64, 2.0, 2.0, 5.0], &[2, 2], &device).unwrap();
        let l = client.cholesky_decompose(&a).unwrap().l;

        // dL/dL = ones
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F64, &device).unwrap();

        let backward = CholeskyBackward::<CpuRuntime>::new(a.id(), l.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        // Verify shape
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 2]);

        // The gradient should be symmetric (since A is symmetric)
        let grad_a_data: Vec<f64> = grad_a.to_vec();
        // Check symmetry: grad_a[0,1] should equal grad_a[1,0]
        assert!(
            approx_eq(grad_a_data[1], grad_a_data[2], 1e-10),
            "grad_a[0,1] = {}, grad_a[1,0] = {}, diff = {}",
            grad_a_data[1],
            grad_a_data[2],
            (grad_a_data[1] - grad_a_data[2]).abs()
        );
    }
}
