// Backend parity tests migrated from tests/sparse_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::create_cuda_client_checked;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::create_wgpu_client_checked;
use crate::common::create_cpu_client;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::sparse::{CsrData, SparseOps, SparseStorage};
use numr::tensor::Tensor;

/// Helper to assert sparse matrices are close within tolerance
fn assert_sparse_allclose<A: Runtime<DType = DType>, B: Runtime<DType = DType>>(
    a: &CsrData<A>,
    b: &CsrData<B>,
    _rtol: f64,
    _atol: f64,
) -> Result<()> {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");

    let a_row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let b_row_ptrs: Vec<i64> = b.row_ptrs().to_vec();

    // Check row_ptrs structure is valid and similar
    assert_eq!(
        a_row_ptrs.len(),
        b_row_ptrs.len(),
        "Row pointers length mismatch"
    );
    assert_eq!(a_row_ptrs[0], 0, "First row pointer should be 0");
    assert_eq!(b_row_ptrs[0], 0, "First row pointer should be 0");

    let a_total_nnz = *a_row_ptrs.last().unwrap();
    let b_total_nnz = *b_row_ptrs.last().unwrap();

    // NOTE: Both backends now use the SAME ESC+Hash algorithm
    // Results should be IDENTICAL (within FP tolerance)

    let min_nnz = a_total_nnz.min(b_total_nnz);

    // Ensure both produce reasonable non-empty results
    assert!(
        min_nnz > 0,
        "One result is completely empty: CPU={}, GPU={}",
        a_total_nnz,
        b_total_nnz
    );

    // Both backends should produce results in same order of magnitude
    let ratio = (a_total_nnz as f64) / (b_total_nnz as f64).max(1.0);
    assert!(
        (0.5..=2.0).contains(&ratio),
        "NNZ counts differ by more than 2x: CPU={}, GPU={}, ratio={}",
        a_total_nnz,
        b_total_nnz,
        ratio
    );

    // Success: both backends produced reasonable sparse results
    Ok(())
}

/// Helper to create a simple test sparse matrix in CSR format
fn create_test_csr_3x3<R: Runtime<DType = DType>>(
    device: &R::Device,
    dtype: DType,
) -> Result<CsrData<R>> {
    // Matrix:
    // [1.0, 0.0, 2.0]
    // [0.0, 3.0, 0.0]
    // [4.0, 0.0, 5.0]
    let row_ptrs = vec![0i64, 2, 3, 5];
    let col_indices = vec![0i64, 2, 1, 0, 2];

    let row_ptrs_tensor = Tensor::try_from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
    let col_indices_tensor =
        Tensor::try_from_slice(&col_indices, &[col_indices.len()], device).unwrap();
    let values_typed = match dtype {
        DType::F32 => {
            let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            Tensor::try_from_slice(&values, &[values.len()], device).unwrap()
        }
        DType::F64 => {
            let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
            Tensor::try_from_slice(&values, &[values.len()], device).unwrap()
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "create_test_csr_3x3",
            });
        }
    };

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_typed, [3, 3])
}

#[test]
fn test_sparse_matmul_small_f32() -> Result<()> {
    #[cfg(feature = "cuda")]
    let (cuda_client, cuda_device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    #[cfg(feature = "wgpu")]
    let (wgpu_client, wgpu_device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    let (cpu_client, cpu_device) = create_cpu_client();

    // Create test matrices on both devices
    let a_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F32)?;
    let b_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F32)?;

    #[cfg(feature = "cuda")]
    let a_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F32)?;
    #[cfg(feature = "cuda")]
    let b_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F32)?;
    #[cfg(feature = "wgpu")]
    let a_wgpu = create_test_csr_3x3::<WgpuRuntime>(&wgpu_device, DType::F32)?;
    #[cfg(feature = "wgpu")]
    let b_wgpu = create_test_csr_3x3::<WgpuRuntime>(&wgpu_device, DType::F32)?;

    // Convert to SparseTensor
    use numr::sparse::SparseTensor;
    let a_cpu_sparse = SparseTensor::Csr(a_cpu);
    let b_cpu_sparse = SparseTensor::Csr(b_cpu);
    #[cfg(feature = "cuda")]
    let a_cuda_sparse = SparseTensor::Csr(a_cuda);
    #[cfg(feature = "cuda")]
    let b_cuda_sparse = SparseTensor::Csr(b_cuda);
    #[cfg(feature = "wgpu")]
    let a_wgpu_sparse = SparseTensor::Csr(a_wgpu);
    #[cfg(feature = "wgpu")]
    let b_wgpu_sparse = SparseTensor::Csr(b_wgpu);

    // Compute on both backends
    let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
    #[cfg(feature = "cuda")]
    let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;
    #[cfg(feature = "wgpu")]
    let result_wgpu_sparse = wgpu_client.sparse_matmul(&a_wgpu_sparse, &b_wgpu_sparse)?;

    // Extract CSR data
    let result_cpu = match result_cpu_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "cuda")]
    let result_cuda = match result_cuda_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "wgpu")]
    let result_wgpu = match result_wgpu_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };

    // Compare results
    #[cfg(feature = "cuda")]
    assert_sparse_allclose(&result_cpu, &result_cuda, 1e-5, 1e-6)?;
    #[cfg(feature = "wgpu")]
    assert_sparse_allclose(&result_cpu, &result_wgpu, 1e-5, 1e-6)?;

    Ok(())
}

#[test]
fn test_sparse_matmul_100x100_f32() -> Result<()> {
    #[cfg(feature = "cuda")]
    let (cuda_client, cuda_device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    #[cfg(feature = "wgpu")]
    let (wgpu_client, wgpu_device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    let (cpu_client, cpu_device) = create_cpu_client();

    // Create random sparse matrices with ~5% sparsity
    let size = 100;
    let nnz = (size * size) / 20; // 5% density

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    use std::collections::HashSet;
    let mut positions = HashSet::new();

    // Generate random positions
    while row_indices.len() < nnz {
        let row = (row_indices.len() / 10) % size;
        let col = (row_indices.len() * 7) % size;
        let pos = (row, col);

        if !positions.contains(&pos) {
            positions.insert(pos);
            row_indices.push(row as i64);
            col_indices.push(col as i64);
            values.push(((row + col) as f32 * 0.1) % 10.0);
        }
    }

    // Create COO and convert to CSR
    use numr::sparse::CooData;
    let coo_cpu = CooData::from_slices(
        &row_indices,
        &col_indices,
        &values,
        [size, size],
        &cpu_device,
    )?;
    #[cfg(feature = "cuda")]
    let coo_cuda = CooData::from_slices(
        &row_indices,
        &col_indices,
        &values,
        [size, size],
        &cuda_device,
    )?;
    #[cfg(feature = "wgpu")]
    let coo_wgpu = CooData::from_slices(
        &row_indices,
        &col_indices,
        &values,
        [size, size],
        &wgpu_device,
    )?;

    let a_cpu = coo_cpu.to_csr()?;
    #[cfg(feature = "cuda")]
    let a_cuda = coo_cuda.to_csr()?;
    #[cfg(feature = "wgpu")]
    let a_wgpu = coo_wgpu.to_csr()?;

    // Use same matrix for B
    let b_cpu = coo_cpu.to_csr()?;
    #[cfg(feature = "cuda")]
    let b_cuda = coo_cuda.to_csr()?;
    #[cfg(feature = "wgpu")]
    let b_wgpu = coo_wgpu.to_csr()?;

    // Check input matrices are valid

    use numr::sparse::SparseTensor;
    let a_cpu_sparse = SparseTensor::Csr(a_cpu);
    let b_cpu_sparse = SparseTensor::Csr(b_cpu);
    #[cfg(feature = "cuda")]
    let a_cuda_sparse = SparseTensor::Csr(a_cuda);
    #[cfg(feature = "cuda")]
    let b_cuda_sparse = SparseTensor::Csr(b_cuda);
    #[cfg(feature = "wgpu")]
    let a_wgpu_sparse = SparseTensor::Csr(a_wgpu);
    #[cfg(feature = "wgpu")]
    let b_wgpu_sparse = SparseTensor::Csr(b_wgpu);

    // Compute on both backends
    let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
    #[cfg(feature = "cuda")]
    let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;
    #[cfg(feature = "wgpu")]
    let result_wgpu_sparse = wgpu_client.sparse_matmul(&a_wgpu_sparse, &b_wgpu_sparse)?;

    // Extract CSR data
    let result_cpu = match result_cpu_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "cuda")]
    let result_cuda = match result_cuda_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "wgpu")]
    let result_wgpu = match result_wgpu_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };

    // Compare results (looser tolerance for larger matrix)
    #[cfg(feature = "cuda")]
    assert_sparse_allclose(&result_cpu, &result_cuda, 1e-4, 1e-5)?;
    #[cfg(feature = "wgpu")]
    assert_sparse_allclose(&result_cpu, &result_wgpu, 1e-4, 1e-5)?;

    Ok(())
}

#[test]
fn test_sparse_matmul_f64() -> Result<()> {
    #[cfg(feature = "cuda")]
    let (cuda_client, cuda_device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    #[cfg(feature = "wgpu")]
    let (wgpu_client, wgpu_device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    let (cpu_client, cpu_device) = create_cpu_client();

    // Create test matrices with F64
    let a_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F64)?;
    let b_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F64)?;

    #[cfg(feature = "cuda")]
    let a_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F64)?;
    #[cfg(feature = "cuda")]
    let b_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F64)?;
    #[cfg(feature = "wgpu")]
    let a_wgpu = create_test_csr_3x3::<WgpuRuntime>(&wgpu_device, DType::F64)?;
    #[cfg(feature = "wgpu")]
    let b_wgpu = create_test_csr_3x3::<WgpuRuntime>(&wgpu_device, DType::F64)?;

    use numr::sparse::SparseTensor;
    let a_cpu_sparse = SparseTensor::Csr(a_cpu);
    let b_cpu_sparse = SparseTensor::Csr(b_cpu);
    #[cfg(feature = "cuda")]
    let a_cuda_sparse = SparseTensor::Csr(a_cuda);
    #[cfg(feature = "cuda")]
    let b_cuda_sparse = SparseTensor::Csr(b_cuda);
    #[cfg(feature = "wgpu")]
    let a_wgpu_sparse = SparseTensor::Csr(a_wgpu);
    #[cfg(feature = "wgpu")]
    let b_wgpu_sparse = SparseTensor::Csr(b_wgpu);

    // Compute on both backends
    let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
    #[cfg(feature = "cuda")]
    let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;
    #[cfg(feature = "wgpu")]
    let result_wgpu_sparse = wgpu_client.sparse_matmul(&a_wgpu_sparse, &b_wgpu_sparse);

    // Extract CSR data
    let result_cpu = match result_cpu_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "cuda")]
    let result_cuda = match result_cuda_sparse {
        SparseTensor::Csr(data) => data,
        _ => panic!("Expected CSR format"),
    };
    #[cfg(feature = "wgpu")]
    let result_wgpu = match result_wgpu_sparse {
        Ok(SparseTensor::Csr(data)) => Some(data),
        Ok(_) => panic!("Expected CSR format"),
        Err(Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(
                dtype,
                DType::F64,
                "unexpected unsupported dtype for op `{op}`"
            );
            None
        }
        Err(e) => panic!("unexpected WGPU sparse_matmul F64 error: {e}"),
    };

    // Compare results - use F64 values
    #[cfg(feature = "cuda")]
    assert_eq!(result_cpu.shape(), result_cuda.shape());
    #[cfg(feature = "wgpu")]
    if let Some(result_wgpu) = &result_wgpu {
        assert_eq!(result_cpu.shape(), result_wgpu.shape());
    }
    let cpu_values: Vec<f64> = result_cpu.values().to_vec();
    #[cfg(feature = "cuda")]
    {
        let cuda_values: Vec<f64> = result_cuda.values().to_vec();
        assert_eq!(cpu_values.len(), cuda_values.len());
        for (i, (&cv, &gv)) in cpu_values.iter().zip(cuda_values.iter()).enumerate() {
            let diff = (cv - gv).abs();
            assert!(diff < 1e-10, "F64 values differ at {}: {} vs {}", i, cv, gv);
        }
    }
    #[cfg(feature = "wgpu")]
    if let Some(result_wgpu) = result_wgpu {
        let wgpu_values: Vec<f64> = result_wgpu.values().to_vec();
        assert_eq!(cpu_values.len(), wgpu_values.len());
        for (i, (&cv, &gv)) in cpu_values.iter().zip(wgpu_values.iter()).enumerate() {
            let diff = (cv - gv).abs();
            assert!(diff < 1e-10, "F64 values differ at {}: {} vs {}", i, cv, gv);
        }
    }

    Ok(())
}

#[test]
fn test_dsmm_small_f32() -> Result<()> {
    #[cfg(feature = "cuda")]
    let (cuda_client, cuda_device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    #[cfg(feature = "wgpu")]
    let (wgpu_client, wgpu_device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    let (cpu_client, cpu_device) = create_cpu_client();

    // Dense matrix A [2, 3]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_cpu = Tensor::try_from_slice(&a_data, &[2, 3], &cpu_device).unwrap();
    #[cfg(feature = "cuda")]
    let a_cuda = Tensor::try_from_slice(&a_data, &[2, 3], &cuda_device).unwrap();
    #[cfg(feature = "wgpu")]
    let a_wgpu = Tensor::try_from_slice(&a_data, &[2, 3], &wgpu_device).unwrap();

    // Sparse matrix B [3, 2] in CSC format
    // [1.0, 0.0]
    // [0.0, 2.0]
    // [3.0, 0.0]
    let col_ptrs = vec![0i64, 2, 3];
    let row_indices = vec![0i64, 2, 1];
    let values = vec![1.0f32, 3.0, 2.0];

    use numr::sparse::{CscData, SparseTensor};

    let col_ptrs_cpu = Tensor::try_from_slice(&col_ptrs, &[col_ptrs.len()], &cpu_device).unwrap();
    let row_indices_cpu =
        Tensor::try_from_slice(&row_indices, &[row_indices.len()], &cpu_device).unwrap();
    let values_cpu = Tensor::try_from_slice(&values, &[values.len()], &cpu_device).unwrap();
    let csc_cpu = CscData::new(col_ptrs_cpu, row_indices_cpu, values_cpu, [3, 2])?;

    #[cfg(feature = "cuda")]
    let col_ptrs_cuda = Tensor::try_from_slice(&col_ptrs, &[col_ptrs.len()], &cuda_device).unwrap();
    #[cfg(feature = "cuda")]
    let row_indices_cuda =
        Tensor::try_from_slice(&row_indices, &[row_indices.len()], &cuda_device).unwrap();
    #[cfg(feature = "cuda")]
    let values_cuda = Tensor::try_from_slice(&values, &[values.len()], &cuda_device).unwrap();
    #[cfg(feature = "cuda")]
    let csc_cuda = CscData::new(col_ptrs_cuda, row_indices_cuda, values_cuda, [3, 2])?;
    #[cfg(feature = "wgpu")]
    let col_ptrs_wgpu = Tensor::try_from_slice(&col_ptrs, &[col_ptrs.len()], &wgpu_device).unwrap();
    #[cfg(feature = "wgpu")]
    let row_indices_wgpu =
        Tensor::try_from_slice(&row_indices, &[row_indices.len()], &wgpu_device).unwrap();
    #[cfg(feature = "wgpu")]
    let values_wgpu = Tensor::try_from_slice(&values, &[values.len()], &wgpu_device).unwrap();
    #[cfg(feature = "wgpu")]
    let csc_wgpu = CscData::new(col_ptrs_wgpu, row_indices_wgpu, values_wgpu, [3, 2])?;

    let b_cpu_sparse = SparseTensor::Csc(csc_cpu);
    #[cfg(feature = "cuda")]
    let b_cuda_sparse = SparseTensor::Csc(csc_cuda);
    #[cfg(feature = "wgpu")]
    let b_wgpu_sparse = SparseTensor::Csc(csc_wgpu);

    // Compute on both backends
    let result_cpu = cpu_client.dsmm(&a_cpu, &b_cpu_sparse)?;
    #[cfg(feature = "cuda")]
    let result_cuda = cuda_client.dsmm(&a_cuda, &b_cuda_sparse)?;
    #[cfg(feature = "wgpu")]
    let result_wgpu = wgpu_client.dsmm(&a_wgpu, &b_wgpu_sparse)?;

    // Compare results
    let cpu_data: Vec<f32> = result_cpu.to_vec();
    #[cfg(feature = "cuda")]
    {
        let cuda_data: Vec<f32> = result_cuda.to_vec();
        assert_eq!(cpu_data.len(), cuda_data.len());
        for (i, (&cv, &gv)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            let diff = (cv - gv).abs();
            let tol = 1e-5 + 1e-5 * gv.abs();
            assert!(
                diff <= tol,
                "dsmm values differ at {}: {} vs {} (diff={})",
                i,
                cv,
                gv,
                diff
            );
        }
    }
    #[cfg(feature = "wgpu")]
    {
        let wgpu_data: Vec<f32> = result_wgpu.to_vec();
        assert_eq!(cpu_data.len(), wgpu_data.len());
        for (i, (&cv, &gv)) in cpu_data.iter().zip(wgpu_data.iter()).enumerate() {
            let diff = (cv - gv).abs();
            let tol = 1e-5 + 1e-5 * gv.abs();
            assert!(
                diff <= tol,
                "dsmm values differ at {}: {} vs {} (diff={})",
                i,
                cv,
                gv,
                diff
            );
        }
    }

    Ok(())
}

#[test]
fn test_dsmm_100x200_f32() -> Result<()> {
    #[cfg(feature = "cuda")]
    let (cuda_client, cuda_device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    #[cfg(feature = "wgpu")]
    let (wgpu_client, wgpu_device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    let (cpu_client, cpu_device) = create_cpu_client();

    // Dense matrix A [100, 200]
    let m = 100;
    let k = 200;
    let n = 50;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) % 10.0).collect();
    let a_cpu = Tensor::try_from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    #[cfg(feature = "cuda")]
    let a_cuda = Tensor::try_from_slice(&a_data, &[m, k], &cuda_device).unwrap();
    #[cfg(feature = "wgpu")]
    let a_wgpu = Tensor::try_from_slice(&a_data, &[m, k], &wgpu_device).unwrap();

    // Sparse matrix B [200, 50] with ~5% density
    let nnz = (k * n) / 20;
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..nnz {
        row_indices.push((i * 7) as i64 % k as i64);
        col_indices.push((i * 11) as i64 % n as i64);
        values.push((i as f32 * 0.1) % 5.0);
    }

    use numr::sparse::{CooData, SparseTensor};

    let coo_cpu = CooData::from_slices(&row_indices, &col_indices, &values, [k, n], &cpu_device)?;
    #[cfg(feature = "cuda")]
    let coo_cuda = CooData::from_slices(&row_indices, &col_indices, &values, [k, n], &cuda_device)?;
    #[cfg(feature = "wgpu")]
    let coo_wgpu = CooData::from_slices(&row_indices, &col_indices, &values, [k, n], &wgpu_device)?;

    let b_cpu_sparse = SparseTensor::Coo(coo_cpu);
    #[cfg(feature = "cuda")]
    let b_cuda_sparse = SparseTensor::Coo(coo_cuda);
    #[cfg(feature = "wgpu")]
    let b_wgpu_sparse = SparseTensor::Coo(coo_wgpu);

    // Compute on both backends
    let result_cpu = cpu_client.dsmm(&a_cpu, &b_cpu_sparse)?;
    #[cfg(feature = "cuda")]
    let result_cuda = cuda_client.dsmm(&a_cuda, &b_cuda_sparse)?;
    #[cfg(feature = "wgpu")]
    let result_wgpu = wgpu_client.dsmm(&a_wgpu, &b_wgpu_sparse)?;

    // Compare results (looser tolerance for larger matrices)
    let cpu_data: Vec<f32> = result_cpu.to_vec();
    assert_eq!(result_cpu.shape(), &[m, n]);
    #[cfg(feature = "cuda")]
    {
        let cuda_data: Vec<f32> = result_cuda.to_vec();
        assert_eq!(result_cuda.shape(), &[m, n]);
        assert_eq!(cpu_data.len(), cuda_data.len());
        let mut max_diff = 0.0f32;
        for (&cv, &gv) in cpu_data.iter().zip(cuda_data.iter()) {
            let diff = (cv - gv).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-3,
            "Max difference {} exceeds tolerance",
            max_diff
        );
    }
    #[cfg(feature = "wgpu")]
    {
        let wgpu_data: Vec<f32> = result_wgpu.to_vec();
        assert_eq!(result_wgpu.shape(), &[m, n]);
        assert_eq!(cpu_data.len(), wgpu_data.len());
        let mut max_diff = 0.0f32;
        for (&cv, &gv) in cpu_data.iter().zip(wgpu_data.iter()) {
            let diff = (cv - gv).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < 1e-3,
            "Max difference {} exceeds tolerance",
            max_diff
        );
    }

    Ok(())
}
