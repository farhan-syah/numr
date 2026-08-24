//! Tests for sparse matrix operations

use super::*;

mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::sparse::SparseOps;

    #[test]
    fn test_spmv_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = Tensor::try_from_slice(&[0i64, 2, 3, 5], &[4], &device).unwrap();
        let col_indices = Tensor::try_from_slice(&[0i64, 2, 2, 0, 1], &[5], &device).unwrap();
        let values = Tensor::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();

        // x = [1, 2, 3]
        let x = Tensor::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        // y = A * x
        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*3 = 9
        // y[2] = 4*1 + 5*2 = 14
        let y = client
            .spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &x, [3, 3])
            .unwrap();

        assert_eq!(y.shape(), &[3]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_add_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // A:
        // [1, 0, 2]
        // [0, 3, 0]
        let a_row_ptrs = Tensor::try_from_slice(&[0i64, 2, 3], &[3], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64, 2, 1], &[3], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        // B:
        // [0, 4, 0]
        // [5, 0, 6]
        let b_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 3], &[3], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[1i64, 0, 2], &[3], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).unwrap();

        // C = A + B:
        // [1, 4, 2]
        // [5, 3, 6]
        let (row_ptrs, col_indices, values) = client
            .add_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 3],
            )
            .unwrap();

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<f32> = values.to_vec();

        assert_eq!(row_ptrs_data, vec![0, 3, 6]);
        assert_eq!(col_indices_data, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(values_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // =========================================================================
    // Zero-Elimination Tests
    // =========================================================================
    //
    // These tests explicitly verify that values below the dtype-specific
    // tolerance threshold are eliminated from sparse results.

    #[test]
    fn test_csr_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test via multiplication which reliably produces small values
        // A = [1e-4, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B = [1e-5, 0]
        //     [0, 2.0]
        let b_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The tiny value (1e-9) should be eliminated, only 4.0 remains
        assert_eq!(
            values_data.len(),
            1,
            "Near-zero values should be eliminated"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }

    #[test]
    fn test_csr_f32_preserves_values_above_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create matrices that produce values ABOVE the tolerance
        // A = [1.001, 0]
        //     [0, 5.0]
        let a_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1.001f32, 5.0], &[2], &device).unwrap();

        // B = [1.0, 0]
        //     [0, 5.0]
        let b_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1.0f32, 5.0], &[2], &device).unwrap();

        // C = A - B should keep the first element (1.001 - 1.0 = 0.001)
        // since 0.001 > 1e-7 (F32 tolerance)
        let (_row_ptrs, col_indices, values) = client
            .sub_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The value 0.001 should be preserved (above tolerance)
        assert_eq!(
            values_data.len(),
            1,
            "Values above tolerance should be kept"
        );
        assert!((values_data[0] - 0.001).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0]);
    }

    #[test]
    fn test_csr_f64_higher_precision_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // F64 has much tighter tolerance (1e-15)
        // Create values that would be eliminated in F32 but kept in F64
        // A = [1.0 + 1e-8, 0]
        let a_row_ptrs = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64], &[1], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1.0 + 1e-8], &[1], &device).unwrap();

        // B = [1.0, 0]
        let b_row_ptrs = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[0i64], &[1], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1.0], &[1], &device).unwrap();

        // C = A - B = 1e-8, which is well above F64 tolerance (1e-15)
        let (_, _col_indices, values) = client
            .sub_csr::<f64>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [1, 2],
            )
            .unwrap();

        let values_data: Vec<f64> = values.to_vec();

        // F64 should preserve values down to 1e-15
        assert_eq!(values_data.len(), 1, "F64 preserves higher precision");
        assert!((values_data[0] - 1e-8).abs() < 1e-16);
    }

    #[test]
    fn test_coo_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // COO format test: Use multiplication to create predictable small values
        // A has triplets: (0, 0, 1e-4), (1, 1, 2.0)
        let a_row_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B has triplets: (0, 0, 1e-5), (1, 1, 2.0)
        let b_row_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (_row_indices, col_indices, values) = client
            .mul_coo::<f32>(
                &a_row_indices,
                &a_col_indices,
                &a_values,
                &b_row_indices,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "COO zero elimination should remove near-zero values"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }

    #[test]
    fn test_csc_f32_removes_values_below_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // CSC format test: Use multiplication to create predictable small values
        // A (column-major):
        // Col 0: [1e-4]
        // Col 1: [2.0]
        let a_col_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_row_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        // B (column-major):
        // Col 0: [1e-5]
        // Col 1: [2.0]
        let b_col_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_row_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // Col 0: 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // Col 1: 2.0 * 2.0 = 4.0 (kept)
        let (_, row_indices, values) = client
            .mul_csc::<f32>(
                &a_col_ptrs,
                &a_row_indices,
                &a_values,
                &b_col_ptrs,
                &b_row_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let row_indices_data: Vec<i64> = row_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "CSC zero elimination should work like CSR"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(row_indices_data, vec![1]);
    }

    #[test]
    fn test_mul_intersection_applies_zero_tolerance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Multiplication uses intersection semantics
        // Create small values through multiplication, testing both sides of tolerance
        // A = [1e-3, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a_values = Tensor::try_from_slice(&[1e-3f32, 2.0], &[2], &device).unwrap();

        // B = [1e-3, 0]
        //     [0, 3.0]
        let b_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b_values = Tensor::try_from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-3 * 1e-3 = 1e-6 (> 1e-7, above tolerance, kept)
        // (1,1): 2.0 * 3.0 = 6.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            2,
            "Both values above F32 tolerance (1e-7)"
        );
        assert!(
            (values_data[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            values_data[0]
        );
        assert!((values_data[1] - 6.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0, 1]);
    }

    #[test]
    fn test_f32_boundary_conditions_at_tolerance_threshold() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test values right at the boundary of F32 tolerance (1e-7)
        // Use 2x2 matrices to avoid edge cases with single-element matrices

        // Test 1: Below tolerance (1e-5 * 1e-4 = 1e-9 < 1e-7, eliminated)
        // A = [1e-5, 0]    B = [1e-4, 0]
        //     [0, 2.0]         [0, 2.0]
        let a1_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a1_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a1_values = Tensor::try_from_slice(&[1e-5f32, 2.0], &[2], &device).unwrap();

        let b1_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b1_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b1_values = Tensor::try_from_slice(&[1e-4f32, 2.0], &[2], &device).unwrap();

        let (_, _col_indices1, values1) = client
            .mul_csr::<f32>(
                &a1_row_ptrs,
                &a1_col_indices,
                &a1_values,
                &b1_row_ptrs,
                &b1_col_indices,
                &b1_values,
                [2, 2],
            )
            .unwrap();
        let vals1 = values1.to_vec::<f32>();
        assert_eq!(
            vals1.len(),
            1,
            "1e-9 < 1e-7 should be eliminated, only 4.0 kept"
        );
        assert!((vals1[0] - 4.0).abs() < 1e-6);

        // Test 2: Above tolerance (1e-3 * 1e-3 = 1e-6 > 1e-7, kept)
        // A = [1e-3, 0]    B = [1e-3, 0]
        //     [0, 3.0]         [0, 3.0]
        let a2_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let a2_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let a2_values = Tensor::try_from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        let b2_row_ptrs = Tensor::try_from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let b2_col_indices = Tensor::try_from_slice(&[0i64, 1], &[2], &device).unwrap();
        let b2_values = Tensor::try_from_slice(&[1e-3f32, 3.0], &[2], &device).unwrap();

        let (_, _col_indices2, values2) = client
            .mul_csr::<f32>(
                &a2_row_ptrs,
                &a2_col_indices,
                &a2_values,
                &b2_row_ptrs,
                &b2_col_indices,
                &b2_values,
                [2, 2],
            )
            .unwrap();
        let vals2 = values2.to_vec::<f32>();
        assert_eq!(vals2.len(), 2, "1e-6 > 1e-7 should be kept along with 9.0");
        assert!(
            (vals2[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            vals2[0]
        );
        assert!((vals2[1] - 9.0).abs() < 1e-6);
    }
}
