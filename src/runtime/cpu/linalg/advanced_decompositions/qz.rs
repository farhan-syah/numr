//! QZ Decomposition (Generalized Schur decomposition)
//!
//! Implements the implicit double-shift QZ algorithm.
//! Uses Givens-pair approach: each left Givens (which damages R's triangularity)
//! is immediately followed by a right Givens to restore R upper triangular.
//! This ensures T remains upper triangular throughout.

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    GeneralizedSchurDecomposition, linalg_demote, linalg_promote, validate_linalg_dtype,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// QZ decomposition for matrix pencil (A, B)
pub fn qz_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<GeneralizedSchurDecomposition<CpuRuntime>> {
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
    let n_b = validate_square_matrix(b.shape())?;
    if n != n_b {
        return Err(Error::ShapeMismatch {
            expected: vec![n, n],
            got: vec![n_b, n_b],
        });
    }

    let result = match a.dtype() {
        DType::F32 => qz_decompose_typed::<f32>(client, &a, &b, n),
        DType::F64 => qz_decompose_typed::<f64>(client, &a, &b, n),
        _ => unreachable!(),
    }?;

    Ok(GeneralizedSchurDecomposition {
        q: linalg_demote(client, result.q, original_dtype)?,
        z: linalg_demote(client, result.z, original_dtype)?,
        s: linalg_demote(client, result.s, original_dtype)?,
        t: linalg_demote(client, result.t, original_dtype)?,
        eigenvalues_real: linalg_demote(client, result.eigenvalues_real, original_dtype)?,
        eigenvalues_imag: linalg_demote(client, result.eigenvalues_imag, original_dtype)?,
    })
}

fn qz_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<GeneralizedSchurDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            z: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            s: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            t: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0], device)?,
            eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0], device)?,
        });
    }

    if n == 1 {
        let s_val: T = a.to_vec::<T>()[0];
        let t_val: T = b.to_vec::<T>()[0];
        let lambda = if t_val.to_f64().abs() > T::epsilon_val() {
            s_val.to_f64() / t_val.to_f64()
        } else if s_val.to_f64() >= 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<CpuRuntime>::try_from_slice(&[T::one()], &[1, 1], device)?,
            z: Tensor::<CpuRuntime>::try_from_slice(&[T::one()], &[1, 1], device)?,
            s: Tensor::<CpuRuntime>::try_from_slice(&[s_val], &[1, 1], device)?,
            t: Tensor::<CpuRuntime>::try_from_slice(&[t_val], &[1, 1], device)?,
            eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(
                &[T::from_f64(lambda)],
                &[1],
                device,
            )?,
            eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&[T::zero()], &[1], device)?,
        });
    }

    // Copy inputs to working matrices
    let mut s_data: Vec<T> = a.to_vec();
    let mut t_data: Vec<T> = b.to_vec();

    // Initialize Q and Z as identity matrices
    let mut q_data: Vec<T> = vec![T::zero(); n * n];
    let mut z_data: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        q_data[i * n + i] = T::one();
        z_data[i * n + i] = T::one();
    }

    // Step 1: Reduce to Hessenberg-triangular form
    hessenberg_triangular_reduction::<T>(&mut s_data, &mut t_data, &mut q_data, &mut z_data, n);

    // Step 2: QZ iteration
    qz_iteration::<T>(&mut s_data, &mut t_data, &mut q_data, &mut z_data, n);

    // Step 3: Final cleanup — zero small sub-diagonals in both H and R
    let eps = T::epsilon_val();
    for i in 1..n {
        // H sub-diagonal
        let h_ii = s_data[(i - 1) * n + (i - 1)].to_f64().abs();
        let h_ip1 = s_data[i * n + i].to_f64().abs();
        let threshold = eps * (h_ii + h_ip1).max(1.0);
        if s_data[i * n + (i - 1)].to_f64().abs() <= threshold {
            s_data[i * n + (i - 1)] = T::zero();
        }
        // R below-diagonal elements (should already be zero, but clean up numerical noise)
        for j in 0..i {
            if t_data[i * n + j].to_f64().abs() <= eps * t_data[j * n + j].to_f64().abs().max(1.0) {
                t_data[i * n + j] = T::zero();
            }
        }
    }

    // Step 4: Extract generalized eigenvalues
    let (eigenvalues_real, eigenvalues_imag) =
        extract_generalized_eigenvalues::<T>(&s_data, &t_data, n);

    Ok(GeneralizedSchurDecomposition {
        q: Tensor::<CpuRuntime>::try_from_slice(&q_data, &[n, n], device)?,
        z: Tensor::<CpuRuntime>::try_from_slice(&z_data, &[n, n], device)?,
        s: Tensor::<CpuRuntime>::try_from_slice(&s_data, &[n, n], device)?,
        t: Tensor::<CpuRuntime>::try_from_slice(&t_data, &[n, n], device)?,
        eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(&eigenvalues_real, &[n], device)?,
        eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&eigenvalues_imag, &[n], device)?,
    })
}

// =========================================================================
// Givens rotation helpers
// =========================================================================

/// Compute Givens rotation to zero b:
/// | c  s| |a|   |r|
/// |-s  c| |b| = |0|
/// Returns (c, s) such that c*a + s*b = r, -s*a + c*b = 0
#[inline]
fn givens_params(a: f64, b: f64) -> (f64, f64) {
    if b.abs() == 0.0 {
        return (1.0, 0.0);
    }
    if a.abs() == 0.0 {
        return (0.0, if b >= 0.0 { 1.0 } else { -1.0 });
    }
    let r = a.hypot(b);
    (a / r, b / r)
}

/// Apply Givens rotation from the left to rows i1, i2:
/// new[i1, j] =  c * mat[i1, j] + s * mat[i2, j]
/// new[i2, j] = -s * mat[i1, j] + c * mat[i2, j]
#[inline]
fn left_givens<T: Element + LinalgElement>(
    mat: &mut [T],
    n: usize,
    i1: usize,
    i2: usize,
    c: f64,
    s: f64,
    col_lo: usize,
    col_hi: usize,
) {
    for j in col_lo..col_hi {
        let a = mat[i1 * n + j].to_f64();
        let b = mat[i2 * n + j].to_f64();
        mat[i1 * n + j] = T::from_f64(c * a + s * b);
        mat[i2 * n + j] = T::from_f64(-s * a + c * b);
    }
}

/// Apply Givens rotation from the right to cols j1, j2:
/// new[i, j1] =  c * mat[i, j1] + s * mat[i, j2]
/// new[i, j2] = -s * mat[i, j1] + c * mat[i, j2]
#[inline]
fn right_givens<T: Element + LinalgElement>(
    mat: &mut [T],
    n: usize,
    j1: usize,
    j2: usize,
    c: f64,
    s: f64,
    row_lo: usize,
    row_hi: usize,
) {
    for i in row_lo..row_hi {
        let a = mat[i * n + j1].to_f64();
        let b = mat[i * n + j2].to_f64();
        mat[i * n + j1] = T::from_f64(c * a + s * b);
        mat[i * n + j2] = T::from_f64(-s * a + c * b);
    }
}

/// Apply a left Givens to H and R (rows i1, i2), accumulate into Q.
/// This zeros mat[i2, target_col] in H (or introduces a transformation).
/// Then immediately restore R upper triangular by a right Givens.
///
/// Left Givens on rows (i1, i2) of R creates fill-in at R[i2, i1]
/// (since i2 > i1 and R was upper triangular, mixing rows creates R[i2, i1] != 0).
/// We restore with right Givens on cols (i1, i2) of R.
#[allow(clippy::too_many_arguments)]
fn left_givens_with_r_restore<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
    i1: usize,
    i2: usize,
    c: f64,
    s: f64,
    ihi: usize,
) {
    // Left Givens on rows i1, i2 of H and R
    left_givens(h, n, i1, i2, c, s, 0, n);
    left_givens(r, n, i1, i2, c, s, 0, n);
    // Accumulate into Q (apply from right to Q columns)
    right_givens(q, n, i1, i2, c, s, 0, n);

    // R now has fill-in at R[i2, i1] (one element below diagonal).
    // Also, due to accumulated numerical noise, other below-diagonal entries
    // in rows i1 and i2 may have grown. Clean them all up.

    // First, zero any below-diagonal entries in row i2 from col 0 to col i2-1
    // using right Givens with the diagonal element.
    for col in (0..i2).rev() {
        let ri2_col = r[i2 * n + col].to_f64();
        if ri2_col.abs() > f64::MIN_POSITIVE {
            let ri2_diag = r[i2 * n + i2].to_f64();
            let rr = ri2_diag.hypot(ri2_col);
            let cr = ri2_diag / rr;
            let sr = -ri2_col / rr;
            right_givens(h, n, col, i2, cr, sr, 0, ihi);
            right_givens(r, n, col, i2, cr, sr, 0, ihi);
            right_givens(z, n, col, i2, cr, sr, 0, n);
        }
    }
}

// =========================================================================
// Hessenberg-triangular reduction
// =========================================================================

/// Reduce (A, B) to Hessenberg-triangular form using Givens rotations.
/// After this: H is upper Hessenberg, R is upper triangular.
fn hessenberg_triangular_reduction<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
) {
    let eps = T::epsilon_val();

    // First, reduce B to upper triangular using QR (left Givens on R, applied to H and Q)
    for k in 0..(n - 1) {
        for i in (k + 1)..n {
            let a_val = r[k * n + k].to_f64();
            let b_val = r[i * n + k].to_f64();

            if b_val.abs() < eps {
                continue;
            }

            let (c, s) = givens_params(a_val, b_val);
            left_givens(r, n, k, i, c, s, 0, n);
            left_givens(h, n, k, i, c, s, 0, n);
            right_givens(q, n, k, i, c, s, 0, n);
        }
    }

    // Then reduce A to upper Hessenberg while keeping B upper triangular.
    // For each column k, zero H[i, k] for i > k+1 using left Givens on rows (k+1, i).
    // Each left Givens on R creates fill-in R[i, k+1] which we fix with right Givens.
    for k in 0..(n - 2) {
        for i in (k + 2)..n {
            let a_val = h[(k + 1) * n + k].to_f64();
            let b_val = h[i * n + k].to_f64();

            if b_val.abs() < eps {
                continue;
            }

            let (c, s) = givens_params(a_val, b_val);

            // Left Givens on rows (k+1, i) of H and R
            left_givens(h, n, k + 1, i, c, s, 0, n);
            left_givens(r, n, k + 1, i, c, s, 0, n);
            right_givens(q, n, k + 1, i, c, s, 0, n);

            // R now has fill-in at R[i, k+1]. Zero it with right Givens on cols (k+1, i).
            let b1 = r[i * n + i].to_f64();
            let b2 = r[i * n + (k + 1)].to_f64();

            if b2.abs() > eps {
                let rr2 = b1.hypot(b2);
                let c2 = b1 / rr2;
                let s2 = -b2 / rr2;

                right_givens(r, n, k + 1, i, c2, s2, 0, n);
                right_givens(h, n, k + 1, i, c2, s2, 0, n);
                right_givens(z, n, k + 1, i, c2, s2, 0, n);
            }
        }
    }
}

// =========================================================================
// QZ iteration (Francis double-shift)
// =========================================================================

/// Main QZ iteration loop with deflation.
fn qz_iteration<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
) {
    if n < 2 {
        return;
    }

    // Standard QZ iteration bound (LAPACK uses 30*n for QR; doubled for QZ pencil)
    let max_iter = 60 * n;
    let eps = T::epsilon_val();
    let mut ihi = n;

    for _iter in 0..max_iter {
        // Deflation: check for converged eigenvalues at the bottom
        while ihi > 1 {
            let i = ihi - 1;
            let h_ii = h[(i - 1) * n + (i - 1)].to_f64().abs();
            let h_ip1 = h[i * n + i].to_f64().abs();
            let threshold = eps * (h_ii + h_ip1).max(1.0);

            if h[i * n + (i - 1)].to_f64().abs() <= threshold {
                h[i * n + (i - 1)] = T::zero();
                ihi -= 1;
            } else {
                break;
            }
        }

        if ihi <= 1 {
            break;
        }

        // Find ilo: start of active unreduced block
        let mut ilo = 0;
        for i in (1..ihi).rev() {
            let h_ii = h[(i - 1) * n + (i - 1)].to_f64().abs();
            let h_ip1 = h[i * n + i].to_f64().abs();
            let threshold = eps * (h_ii + h_ip1).max(1.0);

            if h[i * n + (i - 1)].to_f64().abs() <= threshold {
                h[i * n + (i - 1)] = T::zero();
                ilo = i;
                break;
            }
        }

        let block_size = ihi - ilo;

        if block_size <= 1 {
            ihi = ilo;
            continue;
        }

        if block_size == 2 {
            // 2×2 block — handle with a single-step 2×2 QZ
            qz_step_2x2::<T>(h, r, q, z, n, ilo);
            ihi = ilo;
            continue;
        }

        // Block size >= 3: perform implicit double-shift QZ step
        implicit_double_shift_qz_step::<T>(h, r, q, z, n, ilo, ihi);

        // Clean up any numerical noise below diagonal of R in the active block
        for ii in (ilo + 1)..ihi {
            for jj in ilo..ii {
                if r[ii * n + jj].to_f64().abs() <= eps * r[jj * n + jj].to_f64().abs().max(1.0) {
                    r[ii * n + jj] = T::zero();
                }
            }
        }
    }
}

/// Handle a converged 2×2 block at position [ilo, ilo+2).
/// Reduces to real Schur form: either diagonal (two real eigenvalues)
/// or 2×2 block with sub-diagonal entry (complex conjugate pair).
fn qz_step_2x2<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
    ilo: usize,
) {
    let eps = T::epsilon_val();
    let i = ilo;
    let j = ilo + 1;

    // Check if T[j,i] is non-zero (it shouldn't be if Hessenberg-triangular is correct,
    // but clean it up just in case)
    if r[j * n + i].to_f64().abs() > eps * r[i * n + i].to_f64().abs().max(1.0) {
        // Zero R[j,i] with right Givens on cols (i, j)
        let a = r[j * n + j].to_f64();
        let b = r[j * n + i].to_f64();
        let rr = a.hypot(b);
        let c = a / rr;
        let s = -b / rr;
        right_givens(r, n, i, j, c, s, 0, n);
        right_givens(h, n, i, j, c, s, 0, n);
        right_givens(z, n, i, j, c, s, 0, n);
    }

    // Try to diagonalize the 2×2 block in H
    let a00 = h[i * n + i].to_f64();
    let a10 = h[j * n + i].to_f64();
    let a11 = h[j * n + j].to_f64();

    let trace = a00 + a11;
    let det = a00 * a11 - h[i * n + j].to_f64() * a10;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 && a10.abs() > eps * (a00.abs() + a11.abs()).max(1.0) {
        // Real eigenvalues — diagonalize with a left Givens to zero H[j, i]
        let (c, s) = givens_params(h[i * n + i].to_f64(), h[j * n + i].to_f64());
        left_givens_with_r_restore(h, r, q, z, n, i, j, c, s, n);
    }
    // If disc < 0, leave as 2×2 block (complex conjugate pair)
}

/// Implicit double-shift QZ step on the active block [ilo, ihi).
///
/// Uses Givens-pair approach: every left Givens (which introduces one sub-diagonal
/// element in R) is immediately followed by a right Givens to restore R.
fn implicit_double_shift_qz_step<T: Element + LinalgElement>(
    h: &mut [T],
    r: &mut [T],
    q: &mut [T],
    z: &mut [T],
    n: usize,
    ilo: usize,
    ihi: usize,
) {
    let eps = T::epsilon_val();

    // --- Compute shifts from trailing 2×2 of M = H·R⁻¹ ---
    let m = ihi - 1;
    let h_pp = h[(m - 1) * n + (m - 1)].to_f64();
    let h_qp = h[m * n + (m - 1)].to_f64();
    let h_pq = h[(m - 1) * n + m].to_f64();
    let h_qq = h[m * n + m].to_f64();

    let r_pp = r[(m - 1) * n + (m - 1)].to_f64();
    let r_pq = r[(m - 1) * n + m].to_f64();
    let r_qq = r[m * n + m].to_f64();

    let (s_tr, p_det) = if r_pp.abs() > eps && r_qq.abs() > eps {
        let inv_rpp = 1.0 / r_pp;
        let inv_rqq = 1.0 / r_qq;
        let m00 = h_pp * inv_rpp;
        let m01 = (h_pq - h_pp * r_pq * inv_rpp) * inv_rqq;
        let m10 = h_qp * inv_rpp;
        let m11 = (h_qq - h_qp * r_pq * inv_rpp) * inv_rqq;
        (m00 + m11, m00 * m11 - m01 * m10)
    } else {
        (h_pp + h_qq, h_pp * h_qq - h_pq * h_qp)
    };

    // --- Compute first column of shift polynomial ---
    // v = first column of (M² - s·M + p·I) where M = H·R⁻¹
    // M·e_ilo = H·(R⁻¹·e_ilo) = H·[1/r00, 0, ...]' = [h00/r00, h10/r00, 0, ...]'
    // So only M[ilo,ilo] and M[ilo+1,ilo] are nonzero in first column.
    // M²·e_ilo uses first two columns of M.

    let a00 = h[ilo * n + ilo].to_f64();
    let a10 = h[(ilo + 1) * n + ilo].to_f64();
    let a01 = h[ilo * n + (ilo + 1)].to_f64();
    let a11 = h[(ilo + 1) * n + (ilo + 1)].to_f64();
    let a21 = if ilo + 2 < ihi {
        h[(ilo + 2) * n + (ilo + 1)].to_f64()
    } else {
        0.0
    };

    let rr00 = r[ilo * n + ilo].to_f64();
    let rr01 = r[ilo * n + (ilo + 1)].to_f64();
    let rr11 = r[(ilo + 1) * n + (ilo + 1)].to_f64();

    if rr00.abs() < eps || rr11.abs() < eps {
        return;
    }

    let inv_r00 = 1.0 / rr00;
    let inv_r11 = 1.0 / rr11;

    // M elements:
    // M[ilo, ilo]     = a00 / r00
    // M[ilo+1, ilo]   = a10 / r00
    // M[ilo, ilo+1]   = (a01 - a00 * r01 / r00) / r11
    // M[ilo+1, ilo+1] = (a11 - a10 * r01 / r00) / r11
    // M[ilo+2, ilo+1] = a21 / r11
    let mm00 = a00 * inv_r00;
    let mm10 = a10 * inv_r00;
    let mm01 = (a01 - a00 * rr01 * inv_r00) * inv_r11;
    let mm11 = (a11 - a10 * rr01 * inv_r00) * inv_r11;
    let mm21 = a21 * inv_r11;

    // v = (M² - s_tr·M + p_det·I) · e_ilo
    // v[0] = mm00² + mm01·mm10 - s_tr·mm00 + p_det
    // v[1] = mm10·(mm00 + mm11 - s_tr)
    // v[2] = mm21·mm10
    let v0 = mm00 * mm00 + mm01 * mm10 - s_tr * mm00 + p_det;
    let v1 = mm10 * (mm00 + mm11 - s_tr);
    let v2 = mm21 * mm10;

    if v0.abs() < eps && v1.abs() < eps && v2.abs() < eps {
        return;
    }

    // --- Create initial bulge using two left Givens ---
    // We want to implicitly apply a transformation whose first column is proportional to v.
    // Decompose into two Givens: first zero v[2], then zero v[1].

    // Step 1: Left Givens G1 on rows (ilo+1, ilo+2) to zero v[2]
    let (c1, s1) = givens_params(v1, v2);
    left_givens_with_r_restore(h, r, q, z, n, ilo + 1, ilo + 2, c1, s1, ihi);

    // Step 2: Left Givens G2 on rows (ilo, ilo+1) to zero v[1]
    // After G1, v becomes [v0, c1*v1 + s1*v2, 0]. Use (v0, c1*v1 + s1*v2).
    let v1_new = c1 * v1 + s1 * v2;
    let (c2, s2) = givens_params(v0, v1_new);
    left_givens_with_r_restore(h, r, q, z, n, ilo, ilo + 1, c2, s2, ihi);

    // --- Chase the bulge ---
    // After the initial step, H has extra nonzero entries (the "bulge") below the
    // subdiagonal. The right Givens from R-restoration create fill-in in H that
    // we need to chase down to the bottom of the active block.
    //
    // The bulge pattern: H[k+2, k] and H[k+3, k+1] may be non-zero.
    // We zero them with left Givens, each immediately followed by right Givens
    // to maintain R upper triangular.

    for k in ilo..(ihi - 2) {
        // Zero H[k+2, k] if it's below the subdiagonal
        if h[(k + 2) * n + k].to_f64().abs()
            > eps
                * (h[(k + 1) * n + k].to_f64().abs() + h[(k + 2) * n + (k + 2)].to_f64().abs())
                    .max(1.0)
        {
            let (c, s) = givens_params(h[(k + 1) * n + k].to_f64(), h[(k + 2) * n + k].to_f64());
            left_givens_with_r_restore(h, r, q, z, n, k + 1, k + 2, c, s, ihi);
        }

        // Zero H[k+3, k+1] if it exists and is below the subdiagonal
        if k + 3 < ihi
            && h[(k + 3) * n + (k + 1)].to_f64().abs()
                > eps
                    * (h[(k + 2) * n + (k + 1)].to_f64().abs()
                        + h[(k + 3) * n + (k + 3)].to_f64().abs())
                    .max(1.0)
        {
            let (c, s) = givens_params(
                h[(k + 2) * n + (k + 1)].to_f64(),
                h[(k + 3) * n + (k + 1)].to_f64(),
            );
            left_givens_with_r_restore(h, r, q, z, n, k + 2, k + 3, c, s, ihi);
        }
    }
}

// =========================================================================
// Eigenvalue extraction
// =========================================================================

/// Push eigenvalue pair from a 2×2 matrix with given trace and determinant.
fn push_eigenvalues_from_2x2<T: Element + LinalgElement>(
    trace: f64,
    det: f64,
    real_parts: &mut Vec<T>,
    imag_parts: &mut Vec<T>,
) {
    let disc = trace * trace - 4.0 * det;
    if disc < 0.0 {
        let re = trace / 2.0;
        let im = (-disc).sqrt() / 2.0;
        real_parts.push(T::from_f64(re));
        imag_parts.push(T::from_f64(im));
        real_parts.push(T::from_f64(re));
        imag_parts.push(T::from_f64(-im));
    } else {
        let sqrt_disc = disc.sqrt();
        real_parts.push(T::from_f64((trace + sqrt_disc) / 2.0));
        imag_parts.push(T::zero());
        real_parts.push(T::from_f64((trace - sqrt_disc) / 2.0));
        imag_parts.push(T::zero());
    }
}

/// Extract generalized eigenvalues from the quasi-triangular pencil (S, T).
/// S is quasi-upper-triangular (1×1 and 2×2 blocks on diagonal).
/// T is upper triangular.
fn extract_generalized_eigenvalues<T: Element + LinalgElement>(
    s: &[T],
    t: &[T],
    n: usize,
) -> (Vec<T>, Vec<T>) {
    let mut real_parts = Vec::with_capacity(n);
    let mut imag_parts = Vec::with_capacity(n);
    let eps = T::epsilon_val();

    let mut i = 0;
    while i < n {
        // Check for 2×2 block (complex eigenvalues)
        if i + 1 < n && s[(i + 1) * n + i].to_f64().abs() > eps {
            let a = s[i * n + i].to_f64();
            let b = s[i * n + (i + 1)].to_f64();
            let c = s[(i + 1) * n + i].to_f64();
            let d = s[(i + 1) * n + (i + 1)].to_f64();

            let t_ii = t[i * n + i].to_f64();
            let t_i1i1 = t[(i + 1) * n + (i + 1)].to_f64();
            let t_ii1 = t[i * n + (i + 1)].to_f64();

            // Solve generalized 2×2 eigenvalue problem: det(S_2x2 - λ T_2x2) = 0
            let (trace, det) = if t_ii.abs() > eps && t_i1i1.abs() > eps {
                // Compute M = S * T⁻¹ for the 2×2 block
                let inv_tii = 1.0 / t_ii;
                let inv_ti1 = 1.0 / t_i1i1;
                let m00 = a * inv_tii;
                let m01 = (b - a * t_ii1 * inv_tii) * inv_ti1;
                let m10 = c * inv_tii;
                let m11 = (d - c * t_ii1 * inv_tii) * inv_ti1;
                (m00 + m11, m00 * m11 - m01 * m10)
            } else {
                // Degenerate T block — use S directly
                (a + d, a * d - b * c)
            };

            push_eigenvalues_from_2x2(trace, det, &mut real_parts, &mut imag_parts);
            i += 2;
        } else {
            // 1×1 block — real eigenvalue λ = s_ii / t_ii
            let s_ii = s[i * n + i].to_f64();
            let t_ii = t[i * n + i].to_f64();

            let lambda = if t_ii.abs() > eps {
                s_ii / t_ii
            } else if s_ii >= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            real_parts.push(T::from_f64(lambda));
            imag_parts.push(T::zero());
            i += 1;
        }
    }

    (real_parts, imag_parts)
}
