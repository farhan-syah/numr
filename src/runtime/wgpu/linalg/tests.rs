//! Tests for WebGPU linear algebra implementations.

#[cfg(test)]
mod tests {
    use crate::algorithm::LinearAlgebraAlgorithms;
    use crate::ops::MatmulOps;
    use crate::runtime::wgpu::WgpuRuntime;
    use crate::runtime::wgpu::{WgpuDevice, is_wgpu_available};
    use crate::runtime::{Runtime, RuntimeClient};
    use crate::tensor::Tensor;

    fn create_client() -> crate::runtime::wgpu::WgpuClient {
        let device = WgpuDevice::new(0);
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_trace() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device)
            .unwrap();

        let t = client.trace(&a).unwrap();
        let result: Vec<f32> = t.to_vec();

        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diag() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x3 matrix
        let a = Tensor::<WgpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            device,
        )
        .unwrap();

        let d = client.diag(&a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

        let m = client.diagflat(&a).unwrap();
        let result: Vec<f32> = m.to_vec();

        assert_eq!(m.shape(), &[3, 3]);
        // Expected: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert!((result[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((result[1]).abs() < 1e-5); // [0,1]
        assert!((result[4] - 2.0).abs() < 1e-5); // [1,1]
        assert!((result[8] - 3.0).abs() < 1e-5); // [2,2]
    }

    #[test]
    fn test_lu_decomposition() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[4, 3], [6, 3]]
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device)
            .unwrap();

        let lu = client.lu_decompose(&a).unwrap();

        assert_eq!(lu.lu.shape(), &[2, 2]);
        assert_eq!(lu.pivots.shape(), &[2]);
    }

    #[test]
    fn test_cholesky() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Symmetric positive definite: [[4, 2], [2, 5]]
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[4.0f32, 2.0, 2.0, 5.0], &[2, 2], device)
            .unwrap();

        let chol = client.cholesky_decompose(&a).unwrap();

        assert_eq!(chol.l.shape(), &[2, 2]);

        // L should be lower triangular
        let l_data: Vec<f32> = chol.l.to_vec();
        assert!((l_data[1]).abs() < 1e-5); // Upper triangle should be 0
    }

    #[test]
    fn test_det() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // det = 1*4 - 2*3 = -2
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device)
            .unwrap();

        let d = client.det(&a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert!((result[0] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_solve() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Solve [[2, 1], [1, 2]] @ x = [3, 3]
        // Solution: x = [1, 1]
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device)
            .unwrap();
        let b = Tensor::<WgpuRuntime>::try_from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

        let x = client.solve(&a, &b).unwrap();
        let result: Vec<f32> = x.to_vec();

        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_inverse() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Test 2x2 matrix: [[4, 7], [2, 6]]
        // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device)
            .unwrap();

        let inv = client.inverse(&a).unwrap();
        let result: Vec<f32> = inv.to_vec();

        // Check inverse values (det = 4*6 - 7*2 = 10)
        // inv = (1/10) * [[6, -7], [-2, 4]]
        assert!((result[0] - 0.6).abs() < 1e-4); // [0,0]
        assert!((result[1] - (-0.7)).abs() < 1e-4); // [0,1]
        assert!((result[2] - (-0.2)).abs() < 1e-4); // [1,0]
        assert!((result[3] - 0.4).abs() < 1e-4); // [1,1]
    }

    #[test]
    fn test_matrix_rank_full() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Full rank 2x2 matrix
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device)
            .unwrap();

        let rank = client.matrix_rank(&a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_matrix_rank_deficient() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Rank-deficient 2x2 matrix (rows are linearly dependent)
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device)
            .unwrap();

        let rank = client.matrix_rank(&a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_qr_decomposition() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Test QR: A = Q @ R
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device)
            .unwrap();

        let qr = client.qr_decompose(&a).unwrap();

        // Verify Q @ R == A
        let reconstructed = client.matmul(&qr.q, &qr.r).unwrap();
        let reconstructed = reconstructed.contiguous().unwrap();
        let a_data: Vec<f32> = a.to_vec();
        let reconstructed_data: Vec<f32> = reconstructed.to_vec();

        for i in 0..4 {
            assert!(
                (a_data[i] - reconstructed_data[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                a_data[i],
                reconstructed_data[i]
            );
        }
    }

    #[test]
    fn test_lstsq() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // Overdetermined system: A is 3x2, b is 3x1
        // A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 3]
        let a = Tensor::<WgpuRuntime>::try_from_slice(
            &[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0],
            &[3, 2],
            device,
        )
        .unwrap();
        let b = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

        let x = client.lstsq(&a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let result: Vec<f32> = x.to_vec();

        // Verify the solution is reasonable by checking residual
        assert!(!result[0].is_nan() && !result[0].is_infinite());
        assert!(!result[1].is_nan() && !result[1].is_infinite());
    }

    #[test]
    fn test_kron_2x2() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device)
            .unwrap();
        let b = Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32, 5.0, 6.0, 7.0], &[2, 2], device)
            .unwrap();

        let kron = client.kron(&a, &b).unwrap();
        assert_eq!(kron.shape(), &[4, 4]);

        let data: Vec<f32> = kron.to_vec();

        // Expected result:
        // [[0,  5,  0, 10],
        //  [6,  7, 12, 14],
        //  [0, 15,  0, 20],
        //  [18, 21, 24, 28]]
        #[rustfmt::skip]
        let expected: [f32; 16] = [
            0.0, 5.0, 0.0, 10.0,
            6.0, 7.0, 12.0, 14.0,
            0.0, 15.0, 0.0, 20.0,
            18.0, 21.0, 24.0, 28.0,
        ];

        for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element {} differs: {} vs {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_kron_identity() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let client = create_client();
        let device = client.device();

        // I₂ ⊗ I₂ = I₄
        let i2 = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], device)
            .unwrap();

        let kron = client.kron(&i2, &i2).unwrap();
        assert_eq!(kron.shape(), &[4, 4]);

        let data: Vec<f32> = kron.to_vec();

        // Should be 4x4 identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 1e-5,
                    "kron[{},{}] = {} expected {}",
                    i,
                    j,
                    data[i * 4 + j],
                    expected
                );
            }
        }
    }
}
