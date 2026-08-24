//! Banded linear system solver (Thomas algorithm + general banded LU)

use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::runtime::cpu::jacobi::LinalgElement;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Validate banded system inputs
fn validate_banded(
    ab_shape: &[usize],
    b_shape: &[usize],
    kl: usize,
    ku: usize,
) -> Result<(usize, usize)> {
    let (ab_rows, n) = validate_matrix_2d(ab_shape)?;
    let expected_rows = kl + ku + 1;
    if ab_rows != expected_rows {
        return Err(Error::ShapeMismatch {
            expected: vec![expected_rows, n],
            got: ab_shape.to_vec(),
        });
    }
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "ab",
            reason: "banded system size n must be > 0".to_string(),
        });
    }

    // b can be [n] or [n, nrhs]
    let nrhs = match b_shape.len() {
        1 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n],
                    got: b_shape.to_vec(),
                });
            }
            1
        }
        2 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n, b_shape[1]],
                    got: b_shape.to_vec(),
                });
            }
            b_shape[1]
        }
        _ => {
            return Err(Error::InvalidArgument {
                arg: "b",
                reason: format!("expected 1D or 2D tensor, got {}D", b_shape.len()),
            });
        }
    };

    Ok((n, nrhs))
}

pub fn solve_banded_impl(
    client: &CpuClient,
    ab: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    kl: usize,
    ku: usize,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(ab.dtype())?;
    if ab.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: ab.dtype(),
            rhs: b.dtype(),
        });
    }
    let (ab, original_dtype) = linalg_promote(client, ab)?;
    let (b, _) = linalg_promote(client, b)?;

    let (n, nrhs) = validate_banded(ab.shape(), b.shape(), kl, ku)?;

    let result = match ab.dtype() {
        DType::F32 => solve_banded_typed::<f32>(client, &ab, &b, kl, ku, n, nrhs),
        DType::F64 => solve_banded_typed::<f64>(client, &ab, &b, kl, ku, n, nrhs),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn solve_banded_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    ab: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    kl: usize,
    ku: usize,
    n: usize,
    nrhs: usize,
) -> Result<Tensor<CpuRuntime>> {
    let ab_contig = ab.contiguous()?;
    let b_contig = b.contiguous()?;

    let ab_data: Vec<T> = ab_contig.to_vec();
    let b_data: Vec<T> = b_contig.to_vec();

    let band_rows = kl + ku + 1;

    // Solve for each rhs column
    let mut result = vec![T::from_f64(0.0); n * nrhs];

    if kl == 1 && ku == 1 {
        // Tridiagonal: Thomas algorithm
        for rhs in 0..nrhs {
            let rhs_col: Vec<T> = (0..n).map(|i| b_data[i * nrhs + rhs]).collect();
            let x = thomas_solve::<T>(&ab_data, &rhs_col, n, ku, band_rows)?;
            for i in 0..n {
                result[i * nrhs + rhs] = x[i];
            }
        }
    } else {
        // General banded LU
        for rhs in 0..nrhs {
            let rhs_col: Vec<T> = if nrhs == 1 {
                b_data.clone()
            } else {
                (0..n).map(|i| b_data[i * nrhs + rhs]).collect()
            };
            let x = banded_lu_solve::<T>(&ab_data, &rhs_col, n, kl, ku, band_rows)?;
            for i in 0..n {
                result[i * nrhs + rhs] = x[i];
            }
        }
    }

    let device = client.device();
    let b_is_1d = b.ndim() == 1;

    if b_is_1d {
        Tensor::<CpuRuntime>::from_slice(&result, &[n], device)
    } else {
        Tensor::<CpuRuntime>::from_slice(&result, &[n, nrhs], device)
    }
}

/// Thomas algorithm for tridiagonal systems O(n)
///
/// Band storage: ab[ku-1, j] = upper diagonal, ab[ku, j] = main diagonal, ab[ku+1, j] = lower diagonal
/// With ku=1: ab[0,j] = upper, ab[1,j] = main, ab[2,j] = lower
fn thomas_solve<T: Element + LinalgElement>(
    ab_data: &[T],
    b: &[T],
    n: usize,
    ku: usize,
    _band_rows: usize,
) -> Result<Vec<T>> {
    let zero = T::from_f64(0.0);

    // Extract diagonals from band storage
    // ab is stored row-major: ab[row, col] = ab_data[row * n + col]
    // upper diagonal: ab[ku-1, j] for j=1..n (i.e., row ku-1)
    // main diagonal:  ab[ku, j]   for j=0..n (i.e., row ku)
    // lower diagonal: ab[ku+1, j] for j=0..n-1 (i.e., row ku+1)

    let mut c = vec![zero; n]; // modified upper diagonal
    let mut d = vec![zero; n]; // modified rhs

    // Read main diagonal
    let main_diag = |j: usize| -> T { ab_data[ku * n + j] };
    // Read upper diagonal (element at column j, valid for j < n-1)
    let upper_diag = |j: usize| -> T {
        if ku == 0 {
            zero
        } else {
            ab_data[(ku - 1) * n + j + 1]
        }
    };
    // Read lower diagonal (element at column j, valid for j > 0)
    let lower_diag = |j: usize| -> T { ab_data[(ku + 1) * n + j - 1] };

    // Forward elimination
    let m0 = main_diag(0);
    if m0.to_f64() == 0.0 {
        return Err(Error::Internal("Singular tridiagonal matrix".to_string()));
    }
    let m0_inv = T::from_f64(1.0 / m0.to_f64());
    if n > 1 {
        c[0] = T::from_f64(upper_diag(0).to_f64() * m0_inv.to_f64());
    }
    d[0] = T::from_f64(b[0].to_f64() * m0_inv.to_f64());

    for i in 1..n {
        let a_i = lower_diag(i); // sub-diagonal at row i
        let b_i = main_diag(i); // main diagonal at row i
        let denom = b_i.to_f64() - a_i.to_f64() * c[i - 1].to_f64();
        if denom == 0.0 {
            return Err(Error::Internal("Singular tridiagonal matrix".to_string()));
        }
        let denom_inv = 1.0 / denom;
        if i < n - 1 {
            c[i] = T::from_f64(upper_diag(i).to_f64() * denom_inv);
        }
        d[i] = T::from_f64((b[i].to_f64() - a_i.to_f64() * d[i - 1].to_f64()) * denom_inv);
    }

    // Back substitution
    let mut x = vec![zero; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = T::from_f64(d[i].to_f64() - c[i].to_f64() * x[i + 1].to_f64());
    }

    Ok(x)
}

/// General banded LU solve with partial pivoting
///
/// Band storage: ab[ku + i - j, j] = A[i, j]
fn banded_lu_solve<T: Element + LinalgElement>(
    ab_data: &[T],
    b: &[T],
    n: usize,
    kl: usize,
    ku: usize,
    band_rows: usize,
) -> Result<Vec<T>> {
    // Working copy with extra kl rows for fill-in during pivoting
    let work_rows = 2 * kl + ku + 1;
    let mut work = vec![T::from_f64(0.0); work_rows * n];
    let mut _pivots = vec![0usize; n];

    // Copy band data into working storage
    // Original ab[r, j] -> work[kl + r, j]
    for r in 0..band_rows {
        for j in 0..n {
            work[(kl + r) * n + j] = ab_data[r * n + j];
        }
    }

    let mut rhs = b.to_vec();

    // LU factorization with partial pivoting
    for k in 0..n {
        // Find pivot in column k (rows k..min(k+kl+1, n))
        let max_row = std::cmp::min(k + kl + 1, n);
        let mut pivot_row = k;
        let mut pivot_val = 0.0f64;

        for i in k..max_row {
            // In work storage, element A[i, k] is at work[(kl + ku + i - k), k]
            let row_idx = kl + ku + i - k;
            let val = work[row_idx * n + k].to_f64().abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = i;
            }
        }

        _pivots[k] = pivot_row;

        if pivot_val == 0.0 {
            return Err(Error::Internal("Singular banded matrix".to_string()));
        }

        // Swap rows if needed
        if pivot_row != k {
            // Swap row k and pivot_row in the band
            // For column j, row i maps to work[(kl + ku + i - j), j]
            let j_start = k.saturating_sub(ku);
            let j_end = std::cmp::min(k + kl + ku + 1, n);
            for j in j_start..j_end {
                let idx_k = (kl + ku + k - j) as isize;
                let idx_p = (kl + ku + pivot_row - j) as isize;
                if idx_k >= 0
                    && (idx_k as usize) < work_rows
                    && idx_p >= 0
                    && (idx_p as usize) < work_rows
                {
                    let a = (idx_k as usize) * n + j;
                    let b_idx = (idx_p as usize) * n + j;
                    work.swap(a, b_idx);
                }
            }
            rhs.swap(k, pivot_row);
        }

        // Eliminate below pivot
        let diag_idx = (kl + ku) * n + k; // work[(kl+ku), k] = A[k,k]
        let diag_val = work[diag_idx].to_f64();

        for i in (k + 1)..max_row {
            let sub_row = kl + ku + i - k;
            let sub_idx = sub_row * n + k;
            let factor = work[sub_idx].to_f64() / diag_val;
            work[sub_idx] = T::from_f64(factor); // Store L factor

            // Update remaining elements in row i
            let col_end = std::cmp::min(k + ku + 1, n);
            for j in (k + 1)..col_end {
                let row_i_j = kl + ku + i - j;
                let row_k_j = kl + ku + k - j;
                if row_i_j < work_rows && row_k_j < work_rows {
                    let val_i = work[row_i_j * n + j].to_f64();
                    let val_k = work[row_k_j * n + j].to_f64();
                    work[row_i_j * n + j] = T::from_f64(val_i - factor * val_k);
                }
            }

            // Update rhs
            let rhs_k = rhs[k].to_f64();
            rhs[i] = T::from_f64(rhs[i].to_f64() - factor * rhs_k);
        }
    }

    // Back substitution
    let mut x = rhs;
    for k in (0..n).rev() {
        let diag_idx = (kl + ku) * n + k;
        let diag_val = work[diag_idx].to_f64();

        let col_end = std::cmp::min(k + ku + 1, n);
        for j in (k + 1)..col_end {
            let row_idx = kl + ku + k - j;
            if row_idx < work_rows {
                let u_val = work[row_idx * n + j].to_f64();
                x[k] = T::from_f64(x[k].to_f64() - u_val * x[j].to_f64());
            }
        }
        x[k] = T::from_f64(x[k].to_f64() / diag_val);
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuDevice;

    fn create_client() -> CpuClient {
        let device = CpuDevice::default();
        CpuClient::new(device)
    }

    #[test]
    fn test_tridiagonal_solve() {
        // Tridiagonal system:
        // [2 -1  0  0] [x0]   [1]
        // [-1 2 -1  0] [x1] = [0]
        // [0 -1  2 -1] [x2]   [0]
        // [0  0 -1  2] [x3]   [1]
        //
        // Band storage (kl=1, ku=1): ab shape [3, 4]
        // Row 0 (upper): [0, -1, -1, -1]  (ab[0,j] = A[j-1,j])
        // Row 1 (main):  [2,  2,  2,  2]  (ab[1,j] = A[j,j])
        // Row 2 (lower): [-1,-1, -1,  0]  (ab[2,j] = A[j+1,j])
        let device = CpuDevice::default();
        let ab = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0f64, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0,
            ],
            &[3, 4],
            &device,
        )
        .unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[4], &device).unwrap();
        let client = create_client();
        let x = solve_banded_impl(&client, &ab, &b, 1, 1).unwrap();

        let x_vec: Vec<f64> = x.to_vec();
        // Known solution: x = [1, 1, 1, 1]
        for val in &x_vec {
            assert!((*val - 1.0).abs() < 1e-10, "Expected ~1.0, got {}", val);
        }
    }

    #[test]
    fn test_general_banded_solve() {
        // 3x3 system with kl=1, ku=2:
        // A = [1 2 3]   band storage shape [4, 3] (kl+ku+1=4)
        //     [4 5 0]   ab[ku+i-j, j] = A[i,j]
        //     [0 6 7]
        //
        // ab[2+0-0, 0] = A[0,0] = 1 -> row 2, col 0
        // ab[2+0-1, 1] = A[0,1] = 2 -> row 1, col 1
        // ab[2+0-2, 2] = A[0,2] = 3 -> row 0, col 2
        // ab[2+1-0, 0] = A[1,0] = 4 -> row 3, col 0
        // ab[2+1-1, 1] = A[1,1] = 5 -> row 2, col 1
        // ab[2+2-1, 1] = A[2,1] = 6 -> row 3, col 1
        // ab[2+2-2, 2] = A[2,2] = 7 -> row 2, col 2
        //
        // ab = [[0, 0, 3],   row 0
        //       [0, 2, 0],   row 1
        //       [1, 5, 7],   row 2
        //       [4, 6, 0]]   row 3
        let device = CpuDevice::default();
        let ab = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0f64, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 5.0, 7.0, 4.0, 6.0, 0.0,
            ],
            &[4, 3],
            &device,
        )
        .unwrap();
        // b = [1, 2, 3], solve Ax = b
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();
        let client = create_client();
        let x = solve_banded_impl(&client, &ab, &b, 1, 2).unwrap();

        // Verify Ax ≈ b
        let x_vec: Vec<f64> = x.to_vec();
        let x0 = x_vec[0];
        let x1 = x_vec[1];
        let x2 = x_vec[2];

        // A*x:
        let b0 = 1.0 * x0 + 2.0 * x1 + 3.0 * x2;
        let b1 = 4.0 * x0 + 5.0 * x1;
        let b2 = 6.0 * x1 + 7.0 * x2;

        assert!((b0 - 1.0).abs() < 1e-10, "b0: expected 1.0, got {}", b0);
        assert!((b1 - 2.0).abs() < 1e-10, "b1: expected 2.0, got {}", b1);
        assert!((b2 - 3.0).abs() < 1e-10, "b2: expected 3.0, got {}", b2);
    }

    #[test]
    fn test_tridiagonal_f32() {
        let device = CpuDevice::default();
        let ab = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32, -1.0, -1.0, 2.0, 2.0, 2.0, -1.0, -1.0, 0.0],
            &[3, 3],
            &device,
        )
        .unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0], &[3], &device).unwrap();
        let client = create_client();
        let x = solve_banded_impl(&client, &ab, &b, 1, 1).unwrap();

        let x_vec: Vec<f32> = x.to_vec();
        // Known solution for this system: x = [1, 1, 1]
        for val in &x_vec {
            assert!((*val - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", val);
        }
    }
}
