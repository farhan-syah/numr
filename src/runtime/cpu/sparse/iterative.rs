//! CPU implementation of iterative solvers.
//!
//! This module provides CPU implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative::impl_generic`.

use super::{CpuClient, CpuRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions, CgsResult, GmresOptions,
    GmresResult, IterativeSolvers, JacobiOptions, JacobiResult, LgmresOptions, LgmresResult,
    MinresOptions, MinresResult, QmrOptions, QmrResult, SorOptions, SorResult,
    SparseEigComplexResult, SparseEigOptions, SparseEigResult, SparseSvdResult, SvdsOptions,
};
use crate::algorithm::iterative::{
    arnoldi_eig_impl, bicgstab_impl, cg_impl, cgs_impl, gmres_impl, jacobi_impl, lanczos_eig_impl,
    lgmres_impl, minres_impl, qmr_impl, sor_impl, svds_impl,
};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl IterativeSolvers<CpuRuntime> for CpuClient {
    fn gmres(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: GmresOptions,
    ) -> Result<GmresResult<CpuRuntime>> {
        gmres_impl(self, a, b, x0, options)
    }

    fn bicgstab(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<CpuRuntime>> {
        bicgstab_impl(self, a, b, x0, options)
    }

    fn cg(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: CgOptions,
    ) -> Result<CgResult<CpuRuntime>> {
        cg_impl(self, a, b, x0, options)
    }

    fn minres(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: MinresOptions,
    ) -> Result<MinresResult<CpuRuntime>> {
        minres_impl(self, a, b, x0, options)
    }

    fn cgs(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: CgsOptions,
    ) -> Result<CgsResult<CpuRuntime>> {
        cgs_impl(self, a, b, x0, options)
    }

    fn lgmres(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: LgmresOptions,
    ) -> Result<LgmresResult<CpuRuntime>> {
        lgmres_impl(self, a, b, x0, options)
    }

    fn qmr(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: QmrOptions,
    ) -> Result<QmrResult<CpuRuntime>> {
        qmr_impl(self, a, b, x0, options)
    }

    fn jacobi(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: JacobiOptions,
    ) -> Result<JacobiResult<CpuRuntime>> {
        jacobi_impl(self, a, b, x0, options)
    }

    fn sor(
        &self,
        a: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        options: SorOptions,
    ) -> Result<SorResult<CpuRuntime>> {
        sor_impl(self, a, b, x0, options)
    }

    fn sparse_eig_symmetric(
        &self,
        a: &CsrData<CpuRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigResult<CpuRuntime>> {
        lanczos_eig_impl(self, a, k, options)
    }

    fn sparse_eig(
        &self,
        a: &CsrData<CpuRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigComplexResult<CpuRuntime>> {
        arnoldi_eig_impl(self, a, k, options)
    }

    fn svds(
        &self,
        a: &CsrData<CpuRuntime>,
        k: usize,
        options: SvdsOptions,
    ) -> Result<SparseSvdResult<CpuRuntime>> {
        svds_impl(self, a, k, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::iterative::PreconditionerType;
    use crate::runtime::Runtime;

    fn get_client() -> CpuClient {
        let device = CpuRuntime::default_device();
        CpuRuntime::default_client(&device)
    }

    /// Create 1D Laplacian (tridiagonal) matrix:
    /// [ 2, -1,  0,  0, ...]
    /// [-1,  2, -1,  0, ...]
    /// [ 0, -1,  2, -1, ...]
    /// ...
    fn create_1d_laplacian(
        n: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CsrData<CpuRuntime> {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(0i64);
        for i in 0..n {
            if i > 0 {
                col_indices.push((i - 1) as i64);
                values.push(-1.0f64);
            }
            col_indices.push(i as i64);
            values.push(2.0f64);
            if i < n - 1 {
                col_indices.push((i + 1) as i64);
                values.push(-1.0f64);
            }
            row_ptrs.push(col_indices.len() as i64);
        }

        let row_ptrs_tensor =
            Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
        let col_indices_tensor =
            Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device).unwrap();
        let values_tensor =
            Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device).unwrap();

        CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
            .expect("CSR creation should succeed")
    }

    #[test]
    fn test_gmres_tridiagonal() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device).unwrap();

        let options = GmresOptions {
            max_iter: 100,
            restart: 10,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
            ..Default::default()
        };

        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        assert!(result.converged, "GMRES should converge");
        assert!(
            result.iterations <= 10,
            "GMRES should converge quickly on tridiagonal"
        );
        assert!(
            result.residual_norm < 1e-8,
            "Residual should be small: {}",
            result.residual_norm
        );

        let ax = a.spmv(&result.solution).expect("spmv should work");
        use crate::ops::BinaryOps;
        let residual = client.sub(&b, &ax).expect("sub should work");
        let res_data: Vec<f64> = residual.to_vec();
        let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
    }

    #[test]
    fn test_gmres_with_ilu0() {
        let client = get_client();
        let device = &client.device;

        let n = 20;
        let a = create_1d_laplacian(n, device);

        let b_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

        let options_no_precond = GmresOptions {
            max_iter: 100,
            restart: 30,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
            ..Default::default()
        };

        let options_ilu = GmresOptions {
            max_iter: 100,
            restart: 10,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::Ilu0,
            ..Default::default()
        };

        let result_no_precond = client
            .gmres(&a, &b, None, options_no_precond)
            .expect("GMRES should succeed");
        let result_ilu = client
            .gmres(&a, &b, None, options_ilu)
            .expect("GMRES with ILU should succeed");

        assert!(result_no_precond.converged, "GMRES should converge");
        assert!(result_ilu.converged, "GMRES+ILU should converge");

        assert!(
            result_ilu.iterations <= result_no_precond.iterations,
            "ILU should help convergence: {} vs {}",
            result_ilu.iterations,
            result_no_precond.iterations
        );
    }

    #[test]
    fn test_gmres_identity_matrix() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let row_ptrs =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4, 5], &[n + 1], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4], &[n], device).unwrap();
        let values = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 5], &[n], device).unwrap();
        let a = CsrData::new(row_ptrs, col_indices, values, [n, n])
            .expect("CSR creation should succeed");

        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[n], device).unwrap();

        let options = GmresOptions::default();
        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        assert!(result.converged);
        assert_eq!(
            result.iterations, 1,
            "Identity should converge in 1 iteration"
        );

        let x_data: Vec<f64> = result.solution.to_vec();
        let b_data: Vec<f64> = b.to_vec();
        for i in 0..n {
            assert!(
                (x_data[i] - b_data[i]).abs() < 1e-10,
                "x[{}] = {} != b[{}] = {}",
                i,
                x_data[i],
                i,
                b_data[i]
            );
        }
    }

    #[test]
    fn test_bicgstab_tridiagonal() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device).unwrap();

        let options = crate::algorithm::iterative::BiCgStabOptions {
            max_iter: 100,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        };

        let result = client
            .bicgstab(&a, &b, None, options)
            .expect("BiCGSTAB should succeed");

        assert!(result.converged, "BiCGSTAB should converge");
        assert!(
            result.residual_norm < 1e-8,
            "Residual: {}",
            result.residual_norm
        );

        let ax = a.spmv(&result.solution).expect("spmv should work");
        use crate::ops::BinaryOps;
        let residual = client.sub(&b, &ax).expect("sub should work");
        let res_data: Vec<f64> = residual.to_vec();
        let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
    }

    #[test]
    fn test_gmres_zero_rhs() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b = Tensor::<CpuRuntime>::zeros(&[n], crate::dtype::DType::F64, device).unwrap();

        let options = GmresOptions::default();
        let result = client
            .gmres(&a, &b, None, options)
            .expect("GMRES should succeed");

        assert!(result.converged);
        assert_eq!(result.iterations, 0);

        let x_data: Vec<f64> = result.solution.to_vec();
        for (i, &xi) in x_data.iter().enumerate() {
            assert!(xi.abs() < 1e-14, "x[{}] = {} should be zero", i, xi);
        }
    }

    #[test]
    fn test_gmres_with_initial_guess() {
        let client = get_client();
        let device = &client.device;

        let n = 5;
        let a = create_1d_laplacian(n, device);
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0], &[n], device).unwrap();

        let result1 = client
            .gmres(&a, &b, None, GmresOptions::default())
            .expect("GMRES should succeed");
        assert!(result1.converged);

        let result2 = client
            .gmres(&a, &b, Some(&result1.solution), GmresOptions::default())
            .expect("GMRES should succeed");
        assert!(result2.converged);
        assert!(
            result2.iterations <= 1,
            "With good initial guess, should converge in 0-1 iterations"
        );
    }
}
