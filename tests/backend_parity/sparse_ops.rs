//! Integration tests for sparse operations
//!
//! Tests backend parity between CPU and CUDA implementations,
//! edge cases, and dtype support.

// Backend parity tests moved to tests/backend_parity/sparse.rs.

#[cfg(all(feature = "sparse", feature = "cuda"))]
mod edge_cases {
    use crate::backend_parity::helpers::with_cuda_backend;
    use numr::error::Error;
    use numr::sparse::{CooData, CsrData, SparseOps, SparseStorage, SparseTensor};
    use numr::tensor::Tensor;

    #[test]
    fn test_empty_matrix() {
        with_cuda_backend(|client, device| {
            // Empty sparse matrix (0 non-zeros)
            let row_ptrs = vec![0i64, 0, 0, 0];
            let col_indices: Vec<i64> = vec![];
            let values: Vec<f32> = vec![];

            let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[4], &device).unwrap();
            let col_indices_tensor = Tensor::from_slice(&col_indices, &[0], &device).unwrap();
            let values_tensor = Tensor::from_slice(&values, &[0], &device).unwrap();

            let csr_a = CsrData::new(
                row_ptrs_tensor.clone(),
                col_indices_tensor.clone(),
                values_tensor.clone(),
                [3, 3],
            )
            .expect("csr a");
            let csr_b = CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [3, 3])
                .expect("csr b");

            let a = SparseTensor::Csr(csr_a);
            let b = SparseTensor::Csr(csr_b);

            // Should work - empty × empty = empty
            let result = client.sparse_matmul(&a, &b).expect("sparse_matmul");

            match result {
                SparseTensor::Csr(data) => {
                    assert_eq!(data.shape(), [3, 3]);
                    let nnz = data.values().shape()[0];
                    assert_eq!(nnz, 0, "Expected 0 non-zeros");
                }
                _ => panic!("Expected CSR format"),
            }
        });
    }

    #[test]
    fn test_single_element() {
        with_cuda_backend(|client, device| {
            // 1×1 matrix with single element
            let row_ptrs = vec![0i64, 1];
            let col_indices = vec![0i64];
            let values = vec![5.0f32];

            let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[2], &device).unwrap();
            let col_indices_tensor = Tensor::from_slice(&col_indices, &[1], &device).unwrap();
            let values_tensor = Tensor::from_slice(&values, &[1], &device).unwrap();

            let csr_a = CsrData::new(
                row_ptrs_tensor.clone(),
                col_indices_tensor.clone(),
                values_tensor.clone(),
                [1, 1],
            )
            .expect("csr a");
            let csr_b = CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [1, 1])
                .expect("csr b");

            let a = SparseTensor::Csr(csr_a);
            let b = SparseTensor::Csr(csr_b);

            let result = client.sparse_matmul(&a, &b).expect("sparse_matmul");

            match result {
                SparseTensor::Csr(data) => {
                    assert_eq!(data.shape(), [1, 1]);
                    let result_values: Vec<f32> = data.values().to_vec();
                    assert_eq!(result_values.len(), 1);
                    assert!((result_values[0] - 25.0).abs() < 1e-5, "5 * 5 should be 25");
                }
                _ => panic!("Expected CSR format"),
            }
        });
    }

    #[test]
    fn test_dimension_mismatch() {
        with_cuda_backend(|client, device| {
            // Create matrices with mismatched dimensions
            let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let a_coo =
                CooData::from_slices(&[0, 0, 1, 1], &[0, 1, 0, 1], &a_data, [2, 2], &device)
                    .expect("coo a");

            let b_data = vec![1.0f32, 2.0, 3.0];
            let b_coo = CooData::from_slices(&[0, 1, 2], &[0, 0, 0], &b_data, [3, 1], &device)
                .expect("coo b");

            let a = SparseTensor::Coo(a_coo);
            let b = SparseTensor::Coo(b_coo);

            // Should fail with shape mismatch
            let result = client.sparse_matmul(&a, &b);
            assert!(result.is_err());

            match result {
                Err(Error::ShapeMismatch { .. }) => {
                    // Expected
                }
                _ => panic!("Expected ShapeMismatch error"),
            }
        });
    }

    #[test]
    fn test_dtype_mismatch() {
        with_cuda_backend(|client, device| {
            // Create matrices with different dtypes
            let a_data_f32 = vec![1.0f32, 2.0];
            let a_coo = CooData::from_slices(&[0, 1], &[0, 1], &a_data_f32, [2, 2], &device)
                .expect("coo a");

            let b_data_f64 = vec![1.0f64, 2.0];
            let b_tensor = Tensor::from_slice(&b_data_f64, &[2], &device).unwrap();
            let b_coo_f64 = CooData::new(
                Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap(),
                Tensor::from_slice(&[0i64, 1], &[2], &device).unwrap(),
                b_tensor,
                [2, 2],
            )
            .expect("coo b f64");

            let a = SparseTensor::Coo(a_coo);
            let b = SparseTensor::Coo(b_coo_f64);

            // Should fail with dtype mismatch
            let result = client.sparse_matmul(&a, &b);
            assert!(result.is_err());

            match result {
                Err(Error::DTypeMismatch { .. }) => {
                    // Expected
                }
                _ => panic!("Expected DTypeMismatch error"),
            }
        });
    }

    #[test]
    fn test_very_sparse_1000x1000() {
        with_cuda_backend(|client, device| {
            // 1000×1000 matrix with only 10 non-zeros (99.99% sparse)
            let size = 1000;
            let nnz = 10;

            let row_indices: Vec<i64> = (0..nnz).map(|i| (i * 100) as i64).collect();
            let col_indices: Vec<i64> = (0..nnz).map(|i| (i * 100) as i64).collect();
            let values: Vec<f32> = (0..nnz).map(|i| (i + 1) as f32).collect();

            let coo_a =
                CooData::from_slices(&row_indices, &col_indices, &values, [size, size], &device)
                    .expect("coo a");
            let coo_b =
                CooData::from_slices(&row_indices, &col_indices, &values, [size, size], &device)
                    .expect("coo b");

            let a = SparseTensor::Coo(coo_a);
            let b = SparseTensor::Coo(coo_b);

            // Should handle very sparse matrices efficiently
            let result = client.sparse_matmul(&a, &b).expect("sparse_matmul");

            match result {
                SparseTensor::Csr(data) => {
                    assert_eq!(data.shape(), [size, size]);
                    // Result should also be very sparse
                    let result_nnz = data.values().shape()[0];
                    assert!(
                        result_nnz <= nnz,
                        "Result should have at most {} non-zeros, got {}",
                        nnz,
                        result_nnz
                    );
                }
                _ => panic!("Expected CSR format"),
            }
        });
    }
}
