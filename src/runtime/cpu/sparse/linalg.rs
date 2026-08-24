//! CPU implementation of sparse linear algebra algorithms.
//!
//! This module provides CPU implementations of sparse linear algebra algorithms
//! using the CPU-optimized algorithms in `algorithm::sparse_linalg::cpu`.

use super::{CpuClient, CpuRuntime};
use crate::algorithm::sparse_linalg::{
    IcDecomposition,
    IcOptions,
    IluDecomposition,
    IluFillLevel,
    IluOptions,
    IlukDecomposition,
    IlukOptions,
    IlukSymbolic,
    SparseLinAlgAlgorithms,
    SymbolicIlu0,
    // CPU implementations
    ic0_cpu,
    ilu0_cpu,
    ilu0_numeric_cpu,
    ilu0_symbolic_cpu,
    iluk_cpu,
    iluk_numeric_cpu,
    iluk_symbolic_cpu,
    sparse_solve_triangular_cpu,
};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl SparseLinAlgAlgorithms<CpuRuntime> for CpuClient {
    fn ilu0(
        &self,
        a: &CsrData<CpuRuntime>,
        options: IluOptions,
    ) -> Result<IluDecomposition<CpuRuntime>> {
        ilu0_cpu(a, options)
    }

    fn ic0(
        &self,
        a: &CsrData<CpuRuntime>,
        options: IcOptions,
    ) -> Result<IcDecomposition<CpuRuntime>> {
        ic0_cpu(a, options)
    }

    fn sparse_solve_triangular(
        &self,
        l_or_u: &CsrData<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_solve_triangular_cpu(l_or_u, b, lower, unit_diagonal)
    }

    fn iluk_symbolic(&self, a: &CsrData<CpuRuntime>, level: IluFillLevel) -> Result<IlukSymbolic> {
        iluk_symbolic_cpu(a, level)
    }

    fn iluk_numeric(
        &self,
        a: &CsrData<CpuRuntime>,
        symbolic: &IlukSymbolic,
        opts: &IlukOptions,
    ) -> Result<IlukDecomposition<CpuRuntime>> {
        iluk_numeric_cpu(a, symbolic, opts)
    }

    fn iluk(
        &self,
        a: &CsrData<CpuRuntime>,
        opts: IlukOptions,
    ) -> Result<IlukDecomposition<CpuRuntime>> {
        iluk_cpu(a, opts)
    }

    fn ilu0_symbolic(&self, pattern: &CsrData<CpuRuntime>) -> Result<SymbolicIlu0> {
        ilu0_symbolic_cpu(pattern)
    }

    fn ilu0_numeric(
        &self,
        a: &CsrData<CpuRuntime>,
        symbolic: &SymbolicIlu0,
        options: IluOptions,
    ) -> Result<IluDecomposition<CpuRuntime>> {
        ilu0_numeric_cpu(a, symbolic, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    fn get_client() -> CpuClient {
        let device = CpuRuntime::default_device();
        CpuRuntime::default_client(&device)
    }

    #[test]
    fn test_ilu0_basic() {
        let client = get_client();
        let device = &client.device;

        // Create a simple 3x3 sparse matrix in CSR format:
        // A = [ 4 -1  0]
        //     [-1  4 -1]
        //     [ 0 -1  4]
        // This is a tridiagonal matrix (positive definite)
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device).unwrap();
        let values = Tensor::<CpuRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ilu0(&a, IluOptions::default())
            .expect("ILU0 should succeed");

        // L should be lower triangular (unit diagonal stored implicitly)
        // U should be upper triangular
        assert_eq!(decomp.l.shape, [3, 3]);
        assert_eq!(decomp.u.shape, [3, 3]);

        // Check that L has entries only below diagonal
        let l_col_indices: Vec<i64> = decomp.l.col_indices().to_vec();
        let l_row_ptrs: Vec<i64> = decomp.l.row_ptrs().to_vec();
        for row in 0..3 {
            let start = l_row_ptrs[row] as usize;
            let end = l_row_ptrs[row + 1] as usize;
            for idx in start..end {
                assert!(
                    l_col_indices[idx] < row as i64,
                    "L should be strictly lower triangular"
                );
            }
        }

        // Check that U has diagonal
        let u_col_indices: Vec<i64> = decomp.u.col_indices().to_vec();
        let u_row_ptrs: Vec<i64> = decomp.u.row_ptrs().to_vec();
        for row in 0..3 {
            let start = u_row_ptrs[row] as usize;
            let end = u_row_ptrs[row + 1] as usize;
            // Should have at least the diagonal
            assert!(end > start, "U should have diagonal for row {}", row);
            // First entry in each row should be diagonal or above
            assert!(
                u_col_indices[start] >= row as i64,
                "U should be upper triangular"
            );
        }
    }

    #[test]
    fn test_ic0_basic() {
        let client = get_client();
        let device = &client.device;

        // Create a symmetric positive definite 3x3 sparse matrix:
        // A = [ 4 -1  0]
        //     [-1  4 -1]
        //     [ 0 -1  4]
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device).unwrap();
        let values = Tensor::<CpuRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ic0(&a, IcOptions::default())
            .expect("IC0 should succeed");

        // L should be lower triangular
        assert_eq!(decomp.l.shape, [3, 3]);

        // Check L is lower triangular
        let l_col_indices: Vec<i64> = decomp.l.col_indices().to_vec();
        let l_row_ptrs: Vec<i64> = decomp.l.row_ptrs().to_vec();
        for row in 0..3 {
            let start = l_row_ptrs[row] as usize;
            let end = l_row_ptrs[row + 1] as usize;
            for idx in start..end {
                assert!(
                    l_col_indices[idx] <= row as i64,
                    "L should be lower triangular"
                );
            }
        }

        // Check diagonal is positive (since we're taking sqrt)
        let l_values: Vec<f32> = decomp.l.values().to_vec();
        for row in 0..3usize {
            let start = l_row_ptrs[row] as usize;
            let end = l_row_ptrs[row + 1] as usize;
            // Find diagonal entry
            for idx in start..end {
                if l_col_indices[idx] == row as i64 {
                    assert!(l_values[idx] > 0.0, "L diagonal should be positive");
                }
            }
        }
    }

    #[test]
    fn test_sparse_solve_triangular_lower() {
        let client = get_client();
        let device = &client.device;

        // Create a simple lower triangular matrix:
        // L = [2 0 0]
        //     [1 3 0]
        //     [0 2 4]
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 3, 5], &[4], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1, 1, 2], &[5], device).unwrap();
        let values =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device).unwrap();

        let l = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // Solve L*x = b where b = [2, 4, 8]
        // x[0] = 2/2 = 1
        // x[1] = (4 - 1*1)/3 = 1
        // x[2] = (8 - 2*1)/4 = 1.5
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 8.0], &[3], device).unwrap();

        let x = client
            .sparse_solve_triangular(&l, &b, true, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
        assert!((x_data[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_solve_triangular_upper() {
        let client = get_client();
        let device = &client.device;

        // Create a simple upper triangular matrix:
        // U = [2 1 0]
        //     [0 3 2]
        //     [0 0 4]
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 4, 5], &[4], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 1, 2, 2], &[5], device).unwrap();
        let values =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 1.0, 3.0, 2.0, 4.0], &[5], device).unwrap();

        let u = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // Solve U*x = b where b = [5, 7, 8]
        // x[2] = 8/4 = 2
        // x[1] = (7 - 2*2)/3 = 1
        // x[0] = (5 - 1*1)/2 = 2
        let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 7.0, 8.0], &[3], device).unwrap();

        let x = client
            .sparse_solve_triangular(&u, &b, false, false)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 2.0).abs() < 1e-5);
        assert!((x_data[1] - 1.0).abs() < 1e-5);
        assert!((x_data[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_solve_triangular_unit_diagonal() {
        let client = get_client();
        let device = &client.device;

        // Create a unit lower triangular matrix (no diagonal stored):
        // L = [1 0 0]
        //     [2 1 0]
        //     [0 3 1]
        // We only store: (1,0)=2, (2,1)=3
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1, 2], &[4], device).unwrap();
        let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], device).unwrap();
        let values = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], device).unwrap();

        let l = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // Solve L*x = b where b = [1, 4, 6]
        // x[0] = 1
        // x[1] = 4 - 2*1 = 2
        // x[2] = 6 - 3*2 = 0
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 4.0, 6.0], &[3], device).unwrap();

        let x = client
            .sparse_solve_triangular(&l, &b, true, true)
            .expect("Triangular solve should succeed");

        let x_data: Vec<f32> = x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 2.0).abs() < 1e-5);
        assert!((x_data[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_ilu0_with_diagonal_shift() {
        let client = get_client();
        let device = &client.device;

        // Matrix with small diagonal that could cause numerical issues
        // A = [ 0.001 -1    0  ]
        //     [-1     0.001 -1 ]
        //     [ 0    -1     0.001]
        let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device).unwrap();
        let col_indices =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device).unwrap();
        let values = Tensor::<CpuRuntime>::from_slice(
            &[0.001f32, -1.0, -1.0, 0.001, -1.0, -1.0, 0.001],
            &[7],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        // With diagonal shift, it should succeed even with small pivots
        let options = IluOptions {
            drop_tolerance: 0.0,
            diagonal_shift: 1.0,
        };
        let result = client.ilu0(&a, options);
        // Should succeed with the shift
        assert!(result.is_ok(), "ILU0 with diagonal shift should succeed");
    }
}
