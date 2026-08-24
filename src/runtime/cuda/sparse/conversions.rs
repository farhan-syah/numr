//! Sparse format conversion operations
//!
//! GPU-native conversions between COO, CSR, and CSC formats using Thrust sorting.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::kernels;
use crate::tensor::Tensor;

impl CudaClient {
    pub(crate) fn coo_to_csr_impl<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _ncols] = shape;
        let nnz = row_indices.numel();

        // Validate nnz fits in i32 (Thrust sort uses i32 permutation indices)
        // This limits sparse matrices to 2^31-1 (~2.1 billion) non-zeros
        if nnz > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "COO->CSR nnz ({}) exceeds i32::MAX ({}). Matrices with > 2.1B non-zeros not supported.",
                nnz,
                i32::MAX
            )));
        }
        let nnz_u32 = nnz as u32;
        let device = values.device();

        // Step 1: Allocate temporary arrays for sorting (use i32 for permutation indices)
        // Note: Thrust sort requires i32 permutation array (limits nnz to 2^31-1)
        let perm_indices = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I32, device)?;
        let sorted_rows = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_cols = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_values = Tensor::<CudaRuntime>::zeros(&[nnz], values.dtype(), device)?;

        // Step 2: Initialize permutation indices [0, 1, 2, ..., nnz-1]
        unsafe {
            kernels::launch_coo_init_indices(
                &self.context,
                &self.stream,
                self.device.index,
                perm_indices.ptr(),
                nnz,
            )?;
        }

        // Step 3: Sort by row indices using Thrust (keys=row_indices, values=perm_indices)
        // This sorts the permutation array based on row indices
        unsafe {
            // Copy row_indices to sorted_rows for in-place sorting
            CudaRuntime::copy_within_device(
                row_indices.ptr(),
                sorted_rows.ptr(),
                row_indices.storage().size_in_bytes(),
                device,
            )?;

            kernels::launch_thrust_sort_pairs_i64_i32(
                &self.context,
                &self.stream,
                self.device.index,
                sorted_rows.ptr(),
                perm_indices.ptr(),
                nnz_u32,
            )?;
        }

        // Step 4: Gather col_indices and values using sorted permutation
        unsafe {
            match values.dtype() {
                DType::F32 => kernels::launch_coo_gather::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                DType::F64 => kernels::launch_coo_gather::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                #[cfg(feature = "f16")]
                DType::F16 => kernels::launch_coo_gather::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                #[cfg(feature = "f16")]
                DType::BF16 => kernels::launch_coo_gather::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                _ => {
                    return Err(Error::Internal(format!(
                        "Unsupported dtype for COO gather: {:?}",
                        values.dtype()
                    )));
                }
            }

            kernels::launch_coo_gather_i64(
                &self.context,
                &self.stream,
                self.device.index,
                col_indices.ptr(),
                perm_indices.ptr(),
                sorted_cols.ptr(),
                nnz,
            )?;
        }

        // Step 5: Build row_ptrs from sorted row indices
        let row_ptrs = Tensor::<CudaRuntime>::zeros(&[nrows + 1], DType::I64, device)?;
        unsafe {
            kernels::launch_build_ptrs_from_sorted(
                &self.context,
                &self.stream,
                self.device.index,
                sorted_rows.ptr(),
                row_ptrs.ptr(),
                nnz,
                nrows,
            )?;
        }

        // Synchronize to ensure all operations complete
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

        Ok((row_ptrs, sorted_cols, sorted_values))
    }

    pub(crate) fn coo_to_csc_impl<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [_nrows, ncols] = shape;
        let nnz = row_indices.numel();

        // Validate nnz fits in i32 (Thrust sort uses i32 permutation indices)
        // This limits sparse matrices to 2^31-1 (~2.1 billion) non-zeros
        if nnz > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "COO->CSC nnz ({}) exceeds i32::MAX ({}). Matrices with > 2.1B non-zeros not supported.",
                nnz,
                i32::MAX
            )));
        }
        let nnz_u32 = nnz as u32;
        let device = values.device();

        // Step 1: Allocate temporary arrays for sorting (use i32 for permutation indices)
        // Note: Thrust sort requires i32 permutation array (limits nnz to 2^31-1)
        let perm_indices = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I32, device)?;
        let sorted_cols = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_rows = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_values = Tensor::<CudaRuntime>::zeros(&[nnz], values.dtype(), device)?;

        // Step 2: Initialize permutation indices
        unsafe {
            kernels::launch_coo_init_indices(
                &self.context,
                &self.stream,
                self.device.index,
                perm_indices.ptr(),
                nnz,
            )?;
        }

        // Step 3: Sort by column indices using Thrust
        unsafe {
            CudaRuntime::copy_within_device(
                col_indices.ptr(),
                sorted_cols.ptr(),
                col_indices.storage().size_in_bytes(),
                device,
            )?;

            kernels::launch_thrust_sort_pairs_i64_i32(
                &self.context,
                &self.stream,
                self.device.index,
                sorted_cols.ptr(),
                perm_indices.ptr(),
                nnz_u32,
            )?;
        }

        // Step 4: Gather row_indices and values using sorted permutation
        unsafe {
            match values.dtype() {
                DType::F32 => kernels::launch_coo_gather::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                DType::F64 => kernels::launch_coo_gather::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                #[cfg(feature = "f16")]
                DType::F16 => kernels::launch_coo_gather::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                #[cfg(feature = "f16")]
                DType::BF16 => kernels::launch_coo_gather::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    values.ptr(),
                    perm_indices.ptr(),
                    sorted_values.ptr(),
                    nnz,
                )?,
                _ => {
                    return Err(Error::Internal(format!(
                        "Unsupported dtype for COO gather: {:?}",
                        values.dtype()
                    )));
                }
            }

            kernels::launch_coo_gather_i64(
                &self.context,
                &self.stream,
                self.device.index,
                row_indices.ptr(),
                perm_indices.ptr(),
                sorted_rows.ptr(),
                nnz,
            )?;
        }

        // Step 5: Build col_ptrs from sorted column indices
        let col_ptrs = Tensor::<CudaRuntime>::zeros(&[ncols + 1], DType::I64, device)?;
        unsafe {
            kernels::launch_build_ptrs_from_sorted(
                &self.context,
                &self.stream,
                self.device.index,
                sorted_cols.ptr(),
                col_ptrs.ptr(),
                nnz,
                ncols,
            )?;
        }

        // Synchronize
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

        Ok((col_ptrs, sorted_rows, sorted_values))
    }

    pub(crate) fn csr_to_coo_impl<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU kernel for pointer expansion (CSR → COO)
        let [nrows, _ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let _dtype = values.dtype();

        // Allocate output row_indices on GPU
        let row_indices = Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device)?;

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.ptr();
        let row_indices_ptr = row_indices.ptr();

        // Launch pointer expansion kernel
        unsafe {
            kernels::launch_expand_ptrs(
                &self.context,
                &self.stream,
                self.device.index,
                row_ptrs_ptr,
                row_indices_ptr,
                nrows,
            )?;
        }

        // Sync stream to ensure kernel completion
        self.stream.synchronize()?;

        // col_indices and values remain unchanged
        let col_indices_out = col_indices.clone();
        let values_out = values.clone();

        Ok((row_indices, col_indices_out, values_out))
    }

    pub(crate) fn csc_to_coo_impl<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU kernel for pointer expansion (CSC → COO)
        let [_nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let _dtype = values.dtype();

        // Allocate output col_indices on GPU
        let col_indices = Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device)?;

        // Get device pointers (no data transfer!)
        let col_ptrs_ptr = col_ptrs.ptr();
        let col_indices_ptr = col_indices.ptr();

        // Launch pointer expansion kernel
        unsafe {
            kernels::launch_expand_ptrs(
                &self.context,
                &self.stream,
                self.device.index,
                col_ptrs_ptr,
                col_indices_ptr,
                ncols,
            )?;
        }

        // Sync stream to ensure kernel completion
        self.stream.synchronize()?;

        // row_indices and values remain unchanged
        let row_indices_out = row_indices.clone();
        let values_out = values.clone();

        Ok((row_indices_out, col_indices, values_out))
    }

    pub(crate) fn csr_to_csc_impl<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU implementation: histogram + scan + scatter
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Step 1: Count NNZ per column (histogram)
        let col_counts = Tensor::<CudaRuntime>::zeros(&[ncols], crate::dtype::DType::I64, device)?;

        unsafe {
            kernels::launch_histogram_csr_columns(
                &self.context,
                &self.stream,
                self.device.index,
                row_ptrs.ptr(),
                col_indices.ptr(),
                col_counts.ptr(),
                nrows,
            )?;
        }

        // Step 2: Exclusive scan to build column pointers
        let (col_ptrs, _total_nnz) = unsafe {
            kernels::exclusive_scan_i64_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                &col_counts,
            )?
        };

        // Step 3: Make a copy of column pointers for scatter (will be modified)
        let col_ptrs_working = col_ptrs.clone();

        // Step 4: Allocate output arrays
        let row_indices_out =
            Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device)?;
        let values_out = Tensor::<CudaRuntime>::zeros(&[nnz], dtype, device)?;

        // Step 5: Launch scatter kernel with dtype dispatch
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                kernels::launch_csr_to_csc_transpose::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs.ptr(),
                    col_indices.ptr(),
                    values.ptr(),
                    col_ptrs_working.ptr(),
                    row_indices_out.ptr(),
                    values_out.ptr(),
                    nrows,
                    ncols,
                )?;
            },
            DType::F64 => unsafe {
                kernels::launch_csr_to_csc_transpose::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs.ptr(),
                    col_indices.ptr(),
                    values.ptr(),
                    col_ptrs_working.ptr(),
                    row_indices_out.ptr(),
                    values_out.ptr(),
                    nrows,
                    ncols,
                )?;
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CSR→CSC: {:?}",
                    dtype
                )));
            }
        }

        // Sync to ensure completion
        self.stream.synchronize()?;

        // Return col_ptrs (unmodified), row_indices_out, values_out
        Ok((col_ptrs, row_indices_out, values_out))
    }

    pub(crate) fn csc_to_csr_impl<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU implementation: histogram + scan + scatter
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Step 1: Count NNZ per row (histogram)
        let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], crate::dtype::DType::I64, device)?;

        unsafe {
            kernels::launch_histogram_csc_rows(
                &self.context,
                &self.stream,
                self.device.index,
                col_ptrs.ptr(),
                row_indices.ptr(),
                row_counts.ptr(),
                ncols,
            )?;
        }

        // Step 2: Exclusive scan to build row pointers
        let (row_ptrs, _total_nnz) = unsafe {
            kernels::exclusive_scan_i64_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                &row_counts,
            )?
        };

        // Step 3: Make a copy of row pointers for scatter (will be modified)
        let row_ptrs_working = row_ptrs.clone();

        // Step 4: Allocate output arrays
        let col_indices_out =
            Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device)?;
        let values_out = Tensor::<CudaRuntime>::zeros(&[nnz], dtype, device)?;

        // Step 5: Launch scatter kernel with dtype dispatch
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                kernels::launch_csc_to_csr_transpose::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    col_ptrs.ptr(),
                    row_indices.ptr(),
                    values.ptr(),
                    row_ptrs_working.ptr(),
                    col_indices_out.ptr(),
                    values_out.ptr(),
                    nrows,
                    ncols,
                )?;
            },
            DType::F64 => unsafe {
                kernels::launch_csc_to_csr_transpose::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    col_ptrs.ptr(),
                    row_indices.ptr(),
                    values.ptr(),
                    row_ptrs_working.ptr(),
                    col_indices_out.ptr(),
                    values_out.ptr(),
                    nrows,
                    ncols,
                )?;
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CSC→CSR: {:?}",
                    dtype
                )));
            }
        }

        // Sync to ensure completion
        self.stream.synchronize()?;

        // Return row_ptrs (unmodified), col_indices_out, values_out
        Ok((row_ptrs, col_indices_out, values_out))
    }

    pub(crate) fn sparse_transpose_impl(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Transpose is efficient format conversion
        match a {
            SparseTensor::Csr(data) => {
                let csc = data.to_csc()?;
                Ok(SparseTensor::Csc(csc))
            }
            SparseTensor::Csc(data) => {
                let csr = data.to_csr()?;
                Ok(SparseTensor::Csr(csr))
            }
            SparseTensor::Coo(data) => {
                let [nrows, ncols] = data.shape;
                let transposed = crate::sparse::CooData {
                    row_indices: data.col_indices.clone(),
                    col_indices: data.row_indices.clone(),
                    values: data.values.clone(),
                    shape: [ncols, nrows],
                    sorted: false,
                };
                Ok(SparseTensor::Coo(transposed))
            }
        }
    }
}
