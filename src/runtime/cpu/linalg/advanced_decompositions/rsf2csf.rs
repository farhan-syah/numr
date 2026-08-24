//! Real Schur Form to Complex Schur Form conversion

use super::super::super::jacobi::LinalgElement;
use super::super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    ComplexSchurDecomposition, SchurDecomposition, linalg_demote, linalg_promote,
    validate_linalg_dtype,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Convert real Schur form to complex Schur form
///
/// Transforms 2×2 blocks (complex conjugate eigenvalue pairs) into
/// 1×1 complex diagonal entries.
pub fn rsf2csf_impl(
    client: &CpuClient,
    schur: &SchurDecomposition<CpuRuntime>,
) -> Result<ComplexSchurDecomposition<CpuRuntime>> {
    validate_linalg_dtype(schur.t.dtype())?;
    let (t, original_dtype) = linalg_promote(client, &schur.t)?;
    let (z, _) = linalg_promote(client, &schur.z)?;
    let schur = SchurDecomposition {
        t: t.into_owned(),
        z: z.into_owned(),
    };

    let shape = schur.t.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(Error::Internal(
            "rsf2csf: Schur form T must be square".to_string(),
        ));
    }
    let n = shape[0];

    let result = match schur.t.dtype() {
        DType::F32 => rsf2csf_typed::<f32>(client, &schur, n),
        DType::F64 => rsf2csf_typed::<f64>(client, &schur, n),
        _ => unreachable!(),
    }?;

    Ok(ComplexSchurDecomposition {
        z_real: linalg_demote(client, result.z_real, original_dtype)?,
        z_imag: linalg_demote(client, result.z_imag, original_dtype)?,
        t_real: linalg_demote(client, result.t_real, original_dtype)?,
        t_imag: linalg_demote(client, result.t_imag, original_dtype)?,
    })
}

fn rsf2csf_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    schur: &SchurDecomposition<CpuRuntime>,
    n: usize,
) -> Result<ComplexSchurDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(ComplexSchurDecomposition {
            z_real: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            z_imag: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            t_real: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            t_imag: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
        });
    }

    // Copy input to working matrices
    let mut t_real: Vec<T> = schur.t.to_vec();
    let mut t_imag: Vec<T> = vec![T::zero(); n * n];
    let mut z_real: Vec<T> = schur.z.to_vec();
    let mut z_imag: Vec<T> = vec![T::zero(); n * n];

    // Process 2×2 blocks on the diagonal
    let mut i = 0;
    while i < n {
        if i + 1 < n {
            let subdiag = t_real[(i + 1) * n + i].to_f64();
            if subdiag.abs() > T::epsilon_val() {
                // Found a 2×2 block - convert to complex diagonal entries
                convert_2x2_block::<T>(&mut t_real, &mut t_imag, &mut z_real, &mut z_imag, n, i);
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    Ok(ComplexSchurDecomposition {
        z_real: Tensor::<CpuRuntime>::try_from_slice(&z_real, &[n, n], device)?,
        z_imag: Tensor::<CpuRuntime>::try_from_slice(&z_imag, &[n, n], device)?,
        t_real: Tensor::<CpuRuntime>::try_from_slice(&t_real, &[n, n], device)?,
        t_imag: Tensor::<CpuRuntime>::try_from_slice(&t_imag, &[n, n], device)?,
    })
}

/// Convert a 2×2 block at position (i, i) to complex diagonal form
///
/// For a 2×2 block [a, b; c, d] with complex eigenvalues λ = μ ± iω,
/// we construct a unitary transformation that diagonalizes it.
/// The eigenvector for λ = μ + iω is proportional to [b, λ - a].
fn convert_2x2_block<T: Element + LinalgElement>(
    t_real: &mut [T],
    t_imag: &mut [T],
    z_real: &mut [T],
    z_imag: &mut [T],
    n: usize,
    i: usize,
) {
    // Extract 2×2 block: [a, b; c, d]
    let a = t_real[i * n + i].to_f64();
    let b = t_real[i * n + (i + 1)].to_f64();
    let c = t_real[(i + 1) * n + i].to_f64();
    let d = t_real[(i + 1) * n + (i + 1)].to_f64();

    // Compute complex eigenvalues: λ = μ ± iω
    let mu = (a + d) / 2.0; // Real part
    let _det = a * d - b * c; // Used for reference; eigenvalues computed via discriminant
    let disc = (a - d) * (a - d) / 4.0 + b * c;

    // omega^2 = det - mu^2 = -(discriminant of characteristic polynomial)/4
    // For complex eigenvalues, disc < 0, so omega = sqrt(-disc)
    let omega = if disc < 0.0 { (-disc).sqrt() } else { 0.0 };

    if omega.abs() < T::epsilon_val() {
        // Real eigenvalues - no transformation needed, just set values
        t_imag[i * n + i] = T::zero();
        t_imag[(i + 1) * n + (i + 1)] = T::zero();
        t_imag[i * n + (i + 1)] = T::zero();
        t_imag[(i + 1) * n + i] = T::zero();
        return;
    }

    // Eigenvector v = [b, (μ-a) + iω] for eigenvalue μ + iω
    let v_re_0 = b;
    let v_re_1 = mu - a;
    let v_im_1 = omega;

    // Normalize: |v|^2 = b^2 + (mu-a)^2 + omega^2
    let v_norm_sq = v_re_0 * v_re_0 + v_re_1 * v_re_1 + v_im_1 * v_im_1;
    let v_norm = v_norm_sq.sqrt();

    if v_norm < T::epsilon_val() {
        // Degenerate case - use identity
        t_real[i * n + i] = T::from_f64(mu);
        t_imag[i * n + i] = T::from_f64(omega);
        t_real[(i + 1) * n + (i + 1)] = T::from_f64(mu);
        t_imag[(i + 1) * n + (i + 1)] = T::from_f64(-omega);
        t_real[(i + 1) * n + i] = T::zero();
        t_imag[(i + 1) * n + i] = T::zero();
        return;
    }

    // Normalized eigenvector components
    let u0_re = v_re_0 / v_norm;
    let u1_re = v_re_1 / v_norm;
    let u1_im = v_im_1 / v_norm;

    // Compute T[i,i+1] after transformation using the original values
    let t12_new = b * (u0_re * u0_re - u1_re * u1_re - u1_im * u1_im)
        + (a - d) * u0_re * u1_re
        + 2.0 * u0_re * u1_im * omega;

    // Set the transformed 2×2 block
    t_real[i * n + i] = T::from_f64(mu);
    t_imag[i * n + i] = T::from_f64(omega);
    t_real[(i + 1) * n + (i + 1)] = T::from_f64(mu);
    t_imag[(i + 1) * n + (i + 1)] = T::from_f64(-omega);
    t_real[i * n + (i + 1)] = T::from_f64(t12_new.abs()); // Upper triangular form
    t_imag[i * n + (i + 1)] = T::zero();
    t_real[(i + 1) * n + i] = T::zero();
    t_imag[(i + 1) * n + i] = T::zero();

    // Apply the unitary transformation to Z columns: Z_new = Z * Q
    for row in 0..n {
        let z1_re = z_real[row * n + i].to_f64();
        let z2_re = z_real[row * n + (i + 1)].to_f64();
        let z1_im = z_imag[row * n + i].to_f64();
        let z2_im = z_imag[row * n + (i + 1)].to_f64();

        // Column 1: z1 * u0_re + z2 * u1
        z_real[row * n + i] = T::from_f64(z1_re * u0_re + z2_re * u1_re - z2_im * u1_im);
        z_imag[row * n + i] = T::from_f64(z1_im * u0_re + z2_im * u1_re + z2_re * u1_im);

        // Column 2: z1 * u0_re + z2 * conj(u1)
        z_real[row * n + (i + 1)] = T::from_f64(z1_re * u0_re + z2_re * u1_re + z2_im * u1_im);
        z_imag[row * n + (i + 1)] = T::from_f64(z1_im * u0_re + z2_im * u1_re - z2_re * u1_im);
    }

    // Also need to update T entries above the 2×2 block (rows 0..i)
    for row in 0..i {
        let t1_re = t_real[row * n + i].to_f64();
        let t2_re = t_real[row * n + (i + 1)].to_f64();
        let t1_im = t_imag[row * n + i].to_f64();
        let t2_im = t_imag[row * n + (i + 1)].to_f64();

        // Transform: T_new = T * Q
        t_real[row * n + i] = T::from_f64(t1_re * u0_re + t2_re * u1_re - t2_im * u1_im);
        t_imag[row * n + i] = T::from_f64(t1_im * u0_re + t2_im * u1_re + t2_re * u1_im);
        t_real[row * n + (i + 1)] = T::from_f64(t1_re * u0_re + t2_re * u1_re + t2_im * u1_im);
        t_imag[row * n + (i + 1)] = T::from_f64(t1_im * u0_re + t2_im * u1_re - t2_re * u1_im);
    }
}
