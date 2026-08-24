//! Linear system solvers (Ax=b, triangular, least squares)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::decompositions::{lu_decompose_impl, qr_decompose_impl};
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Solve Ax = b using LU decomposition
pub fn solve_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => solve_typed::<f32>(client, &a, &b, n),
        DType::F64 => solve_typed::<f64>(client, &a, &b, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();
    let pivots_data: Vec<i64> = lu_decomp.pivots.to_vec();

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (b_rows, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    if b_rows != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_rows],
        });
    }

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        // Apply permutation to b
        let mut pb: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            pb[i] = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };
        }
        for (i, &pivot_idx) in pivots_data.iter().enumerate() {
            let pivot_row = pivot_idx as usize;
            if pivot_row != i {
                pb.swap(i, pivot_row);
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        let mut y: Vec<T> = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + lu_data[i * n + j] * y[j];
            }
            y[i] = pb[i] - sum;
        }

        // Backward substitution: Ux = y
        let mut x_col: Vec<T> = vec![T::zero(); n];
        for ii in (0..n).rev() {
            let mut s = T::zero();
            for jj in (ii + 1)..n {
                s = s + lu_data[ii * n + jj] * x_col[jj];
            }
            x_col[ii] = (y[ii] - s) / lu_data[ii * n + ii];
        }
        // Copy result
        for ii in 0..n {
            if num_rhs == 1 {
                x[ii] = x_col[ii];
            } else {
                x[ii * num_rhs + rhs] = x_col[ii];
            }
        }
    }

    // Create output tensor
    if b_shape.len() == 1 {
        Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device)
    }
}

/// Forward substitution for lower triangular system
pub fn solve_triangular_lower_impl(
    client: &CpuClient,
    l: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(l.dtype())?;
    if l.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: l.dtype(),
            rhs: b.dtype(),
        });
    }
    let (l, original_dtype) = linalg_promote(client, l)?;
    let (b, _) = linalg_promote(client, b)?;
    let n = validate_square_matrix(l.shape())?;

    let result = match l.dtype() {
        DType::F32 => solve_triangular_lower_typed::<f32>(client, &l, &b, n, unit_diagonal),
        DType::F64 => solve_triangular_lower_typed::<f64>(client, &l, &b, n, unit_diagonal),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_triangular_lower_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    l: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
    unit_diagonal: bool,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let l_data: Vec<T> = l.to_vec();
    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + l_data[i * n + j] * x_val;
            }

            let b_val = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };

            let result = b_val - sum;
            let x_val = if unit_diagonal {
                result
            } else {
                result / l_data[i * n + i]
            };

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
    }

    if b_shape.len() == 1 {
        Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device)
    }
}

/// Backward substitution for upper triangular system
pub fn solve_triangular_upper_impl(
    client: &CpuClient,
    u: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(u.dtype())?;
    if u.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: u.dtype(),
            rhs: b.dtype(),
        });
    }
    let (u, original_dtype) = linalg_promote(client, u)?;
    let (b, _) = linalg_promote(client, b)?;
    let n = validate_square_matrix(u.shape())?;

    let result = match u.dtype() {
        DType::F32 => solve_triangular_upper_typed::<f32>(client, &u, &b, n),
        DType::F64 => solve_triangular_upper_typed::<f64>(client, &u, &b, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_triangular_upper_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    u: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let u_data: Vec<T> = u.to_vec();
    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        for i in (0..n).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..n {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + u_data[i * n + j] * x_val;
            }

            let b_val = if num_rhs == 1 {
                b_data[i]
            } else {
                b_data[i * num_rhs + rhs]
            };

            let x_val = (b_val - sum) / u_data[i * n + i];

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
    }

    if b_shape.len() == 1 {
        Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device)
    }
}

/// Least squares via QR decomposition
pub fn lstsq_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => lstsq_typed::<f32>(client, &a, &b, m, n),
        DType::F64 => lstsq_typed::<f64>(client, &a, &b, m, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn lstsq_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute thin QR
    let qr = qr_decompose_impl(client, a, true)?;
    let q_data: Vec<T> = qr.q.to_vec();
    let r_data: Vec<T> = qr.r.to_vec();

    let b_shape = b.shape();
    let (_, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else {
        (b_shape[0], b_shape[1])
    };

    let b_data: Vec<T> = b.to_vec();
    let k = m.min(n);

    // Compute Q^T @ b
    let mut qtb: Vec<T> = vec![T::zero(); k * num_rhs];
    for rhs in 0..num_rhs {
        for i in 0..k {
            let mut sum = T::zero();
            for j in 0..m {
                let b_val = if num_rhs == 1 {
                    b_data[j]
                } else {
                    b_data[j * num_rhs + rhs]
                };
                sum = sum + q_data[j * k + i] * b_val;
            }
            if num_rhs == 1 {
                qtb[i] = sum;
            } else {
                qtb[i * num_rhs + rhs] = sum;
            }
        }
    }

    // Solve R @ x = Q^T @ b via back substitution
    // R is k x n, but only the first k columns are used
    let mut x: Vec<T> = vec![T::zero(); n * num_rhs];

    for rhs in 0..num_rhs {
        // Solve k x k upper triangular system
        for i in (0..k).rev() {
            let mut sum = T::zero();
            for j in (i + 1)..k {
                let x_val = if num_rhs == 1 {
                    x[j]
                } else {
                    x[j * num_rhs + rhs]
                };
                sum = sum + r_data[i * n + j] * x_val;
            }

            let qtb_val = if num_rhs == 1 {
                qtb[i]
            } else {
                qtb[i * num_rhs + rhs]
            };

            let x_val = (qtb_val - sum) / r_data[i * n + i];

            if num_rhs == 1 {
                x[i] = x_val;
            } else {
                x[i * num_rhs + rhs] = x_val;
            }
        }
        // Remaining x[k..n] are zeros (already initialized)
    }

    if b_shape.len() == 1 {
        Tensor::<CpuRuntime>::from_slice(&x[..n], &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&x, &[n, num_rhs], device)
    }
}
