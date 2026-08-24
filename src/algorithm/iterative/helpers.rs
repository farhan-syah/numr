//! Shared helper functions for iterative solvers
//!
//! These functions are used by both standard GMRES and adaptive GMRES
//! implementations to avoid code duplication.

use crate::algorithm::sparse_linalg::{
    IluDecomposition, IlukDecomposition, SparseLinAlgAlgorithms,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Near-zero threshold for detecting breakdown (division by near-zero scalar).
/// Used in CG, MINRES, CGS for inner product denominators.
pub const BREAKDOWN_TOL: f64 = 1e-40;

/// Threshold for detecting invariant subspace (lucky breakdown) in Krylov methods.
/// When a new basis vector has norm below this, the Krylov subspace is exhausted.
pub const INVARIANT_SUBSPACE_TOL: f64 = 1e-14;

/// Tolerance for Gram-Schmidt reorthogonalization corrections.
/// Projections with magnitude below this are skipped to avoid amplifying noise.
pub const REORTH_TOL: f64 = 1e-15;

/// Compute vector L2 norm: ||v|| = sqrt(sum(v^2))
///
/// Uses optimized `item()` for scalar extraction (single element copy, no Vec allocation).
pub fn vector_norm<R, C>(client: &C, v: &Tensor<R>) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + UnaryOps<R> + ReduceOps<R>,
{
    // v^2
    let v_sq = client.mul(v, v)?;
    // sum(v^2) over all dimensions → scalar
    let ndim = v_sq.ndim();
    let dims: Vec<usize> = (0..ndim).collect();
    let sum_sq = client.sum(&v_sq, &dims, false)?;
    // sqrt(sum(v^2))
    let norm_tensor = client.sqrt(&sum_sq)?;

    // Extract scalar using optimized item() (no Vec allocation)
    match norm_tensor.dtype() {
        DType::F32 => Ok(norm_tensor.item::<f32>()? as f64),
        DType::F64 => Ok(norm_tensor.item::<f64>()?),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "vector_norm",
        }),
    }
}

/// Compute vector dot product: <u, v> = sum(u * v)
///
/// Uses optimized `item()` for scalar extraction (single element copy, no Vec allocation).
pub fn vector_dot<R, C>(client: &C, u: &Tensor<R>, v: &Tensor<R>) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ReduceOps<R>,
{
    // u * v
    let uv = client.mul(u, v)?;
    // sum(u * v) over all dimensions → scalar
    let ndim = uv.ndim();
    let dims: Vec<usize> = (0..ndim).collect();
    let dot_tensor = client.sum(&uv, &dims, false)?;

    // Extract scalar using optimized item() (no Vec allocation)
    match dot_tensor.dtype() {
        DType::F32 => Ok(dot_tensor.item::<f32>()? as f64),
        DType::F64 => Ok(dot_tensor.item::<f64>()?),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "vector_dot",
        }),
    }
}

/// Apply ILU(0) preconditioner: z = M^-1 @ v = U^-1 @ (L^-1 @ v)
pub fn apply_ilu0_preconditioner<R, C>(
    client: &C,
    precond: &Option<IluDecomposition<R>>,
    v: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SparseLinAlgAlgorithms<R>,
{
    match precond {
        None => Ok(v.clone()),
        Some(ilu) => {
            // L^-1 @ v (forward substitution, unit diagonal)
            let y = client.sparse_solve_triangular(&ilu.l, v, true, true)?;
            // U^-1 @ y (backward substitution)
            client.sparse_solve_triangular(&ilu.u, &y, false, false)
        }
    }
}

/// Apply ILU(k) preconditioner: z = M^-1 @ v = U^-1 @ (L^-1 @ v)
pub fn apply_iluk_preconditioner<R, C>(
    client: &C,
    ilu: &IlukDecomposition<R>,
    v: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SparseLinAlgAlgorithms<R>,
{
    // L^-1 @ v (forward substitution, unit diagonal)
    let y = client.sparse_solve_triangular(&ilu.l, v, true, true)?;
    // U^-1 @ y (backward substitution)
    client.sparse_solve_triangular(&ilu.u, &y, false, false)
}

/// Compute Givens rotation coefficients
///
/// Given a and b, computes c, s, r such that:
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
///
/// where c^2 + s^2 = 1 and r = sqrt(a^2 + b^2) >= 0
///
/// This implementation ensures r >= 0 for numerical robustness,
/// following the convention used in LAPACK's DLARTG.
#[inline]
pub fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-15 {
        // b ≈ 0, no rotation needed
        // Ensure r >= 0 by adjusting sign
        if a >= 0.0 {
            (1.0, 0.0, a)
        } else {
            (-1.0, 0.0, -a)
        }
    } else if a.abs() < 1e-15 {
        // a ≈ 0, 90-degree rotation
        // r = |b| >= 0
        (0.0, b.signum(), b.abs())
    } else {
        // General case: use hypot for numerical stability
        let r = a.hypot(b); // Always >= 0
        let c = a / r;
        let s = b / r;
        (c, s, r)
    }
}

/// Solve upper triangular system R @ y = g via back substitution
///
/// h_matrix[j] contains column j of R (upper triangular)
pub fn solve_upper_triangular(h_matrix: &[Vec<f64>], g: &[f64]) -> Vec<f64> {
    let m = g.len();
    let mut y = vec![0.0; m];

    for i in (0..m).rev() {
        let mut sum = g[i];
        for j in (i + 1)..m {
            sum -= h_matrix[j][i] * y[j];
        }
        if h_matrix[i][i].abs() > 1e-15 {
            y[i] = sum / h_matrix[i][i];
        }
    }

    y
}

/// Update solution: x = x + Z @ y where Z contains preconditioned basis vectors
pub fn update_solution<R, C>(
    client: &C,
    x: &Tensor<R>,
    z_basis: &[Tensor<R>],
    y: &[f64],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ScalarOps<R>,
{
    let m = y.len();
    let mut delta = Tensor::<R>::zeros(x.shape(), x.dtype(), x.device())?;

    // delta = sum_j y[j] * z[j] (z vectors already preconditioned)
    for j in 0..m {
        if y[j].abs() > 1e-15 {
            let scaled_z = client.mul_scalar(&z_basis[j], y[j])?;
            delta = client.add(&delta, &scaled_z)?;
        }
    }

    // x = x + delta
    client.add(x, &delta)
}

/// Detect stagnation: residual not decreasing by reduction_factor over window_size iterations
pub fn detect_stagnation(
    residual_history: &[f64],
    params: &super::types::StagnationParams,
) -> bool {
    let len = residual_history.len();
    if len < params.min_iterations + params.window_size {
        return false;
    }

    let current = residual_history[len - 1];
    let past = residual_history[len - 1 - params.window_size];

    // Stagnation if current residual is greater than reduction_factor * past residual
    // i.e., we haven't reduced the residual by the required factor
    current > params.reduction_factor * past
}

/// Compute a linear combination of basis vectors: result = sum_j coefficients[j] * basis[j]
///
/// Accumulates entirely on-device (no GPU↔CPU transfers). Both basis vectors and
/// the result remain on the compute device throughout.
///
/// Coefficients with magnitude below [`REORTH_TOL`] are skipped.
pub fn accumulate_basis_combination<R, C>(
    client: &C,
    basis: &[Tensor<R>],
    coefficients: &[f64],
    n: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ScalarOps<R>,
{
    let mut result = Tensor::<R>::zeros(&[n], dtype, device)?;
    let len = basis.len().min(coefficients.len());
    for (vj, &coeff) in basis.iter().zip(coefficients.iter()).take(len) {
        if coeff.abs() > REORTH_TOL {
            let scaled = client.mul_scalar(vj, coeff)?;
            result = client.add(&result, &scaled)?;
        }
    }
    Ok(result)
}

/// Extract the inverse diagonal of a CSR matrix: D_inv[i] = 1 / A[i,i]
///
/// Uses the on-device `extract_diagonal_csr` kernel — no GPU↔CPU transfers.
/// Called once at solver init (not in any iteration loop).
///
/// Used by Jacobi, SOR, and AMG V-cycle smoothing.
pub fn extract_diagonal_inv<R, C>(client: &C, a: &crate::sparse::CsrData<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: UnaryOps<R> + BinaryOps<R> + ScalarOps<R> + crate::sparse::SparseOps<R>,
{
    let n = a.shape[0];
    let device = a.values().device();
    let dtype = a.values().dtype();

    // Extract diagonal on-device via SparseOps kernel
    let diag = a.diagonal_with_client(client)?;

    // Compute 1/diag on-device: create ones tensor, divide
    let ones = Tensor::<R>::ones(&[n], dtype, device)?;
    client.div(&ones, &diag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_givens_rotation() {
        // Test case 1: a = 3, b = 4 -> r = 5
        let (c, s, r) = givens_rotation(3.0, 4.0);
        assert!((c * c + s * s - 1.0).abs() < 1e-10, "c^2 + s^2 = 1");
        assert!((r - 5.0).abs() < 1e-10, "r = 5");
        assert!(r >= 0.0, "r must be non-negative");
        assert!((c * 3.0 + s * 4.0 - 5.0).abs() < 1e-10, "rotation works");
        assert!((-s * 3.0 + c * 4.0).abs() < 1e-10, "zeroes out b");

        // Test case 2: b = 0, a > 0
        let (c, s, r) = givens_rotation(5.0, 0.0);
        assert_eq!(c, 1.0);
        assert_eq!(s, 0.0);
        assert_eq!(r, 5.0);
        assert!(r >= 0.0);

        // Test case 3: b = 0, a < 0 (must ensure r >= 0)
        let (c, s, r) = givens_rotation(-5.0, 0.0);
        assert_eq!(c, -1.0);
        assert_eq!(s, 0.0);
        assert_eq!(r, 5.0);
        assert!(r >= 0.0, "r must be non-negative even for negative a");
        // Verify: c*a + s*b = -1*(-5) + 0*0 = 5 = r ✓
        assert!((c * (-5.0) + s * 0.0 - r).abs() < 1e-10);

        // Test case 4: a = 0, b > 0
        let (c, s, r) = givens_rotation(0.0, 3.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(r, 3.0);
        assert!(r >= 0.0);

        // Test case 5: a = 0, b < 0
        let (c, s, r) = givens_rotation(0.0, -3.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, -1.0);
        assert_eq!(r, 3.0);
        assert!(r >= 0.0, "r must be non-negative for negative b");

        // Test case 6: negative a and b
        let (c, s, r) = givens_rotation(-3.0, -4.0);
        assert!((c * c + s * s - 1.0).abs() < 1e-10, "c^2 + s^2 = 1");
        assert!((r - 5.0).abs() < 1e-10, "r = 5");
        assert!(r >= 0.0, "r must be non-negative");
        // Verify rotation: c*a + s*b = r, -s*a + c*b = 0
        assert!(
            (c * (-3.0) + s * (-4.0) - r).abs() < 1e-10,
            "rotation gives r"
        );
        assert!((-s * (-3.0) + c * (-4.0)).abs() < 1e-10, "zeroes out b");
    }

    #[test]
    fn test_solve_upper_triangular() {
        // R = [[2, 1], [0, 3]], g = [3, 6]
        // Solution: y[1] = 6/3 = 2, y[0] = (3 - 1*2)/2 = 0.5
        let h_matrix = vec![vec![2.0, 0.0], vec![1.0, 3.0]];
        let g = vec![3.0, 6.0];
        let y = solve_upper_triangular(&h_matrix, &g);
        assert!((y[0] - 0.5).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_stagnation_detection() {
        let params = super::super::types::StagnationParams {
            reduction_factor: 0.5,
            window_size: 3,
            min_iterations: 2,
        };

        // Not enough iterations
        let history = vec![1.0, 0.9];
        assert!(!detect_stagnation(&history, &params));

        // Enough iterations, but still improving
        let history = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        assert!(!detect_stagnation(&history, &params));

        // Stagnation: not improving enough
        let history = vec![1.0, 0.9, 0.85, 0.8, 0.75, 0.72];
        assert!(detect_stagnation(&history, &params));
    }
}
