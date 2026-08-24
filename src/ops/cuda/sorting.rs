//! Sorting and searching operations for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{SortingOps, compute_reduce_strides};
use crate::runtime::cuda::kernels::{
    launch_argsort, launch_bincount, launch_count_nonzero, launch_count_unique,
    launch_extract_unique, launch_flat_to_multi_index, launch_gather_nonzero, launch_searchsorted,
    launch_sort, launch_sort_values_only, launch_topk,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;

impl SortingOps<CudaRuntime> for CudaClient {
    fn sort(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(a.clone());
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;

        unsafe {
            launch_sort_values_only(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok(out)
    }

    fn sort_with_indices(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            let indices = Tensor::<CudaRuntime>::try_zeros(shape, DType::I64, &self.device)?;
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a)?;
        let out_values = Tensor::<CudaRuntime>::try_empty(shape, dtype, &self.device)?;
        let out_indices = Tensor::<CudaRuntime>::try_empty(shape, DType::I64, &self.device)?;

        unsafe {
            launch_sort(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out_values.ptr(),
                out_indices.ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok((out_values, out_indices))
    }

    fn argsort(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return Tensor::<CudaRuntime>::try_zeros(shape, DType::I64, &self.device);
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::try_empty(shape, DType::I64, &self.device)?;

        unsafe {
            launch_argsort(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                outer_size,
                sort_size,
                inner_size,
                descending,
            )?;
        }

        Ok(out)
    }

    fn topk(
        &self,
        a: &Tensor<CudaRuntime>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        if ndim == 0 {
            if k > 1 {
                return Err(Error::InvalidArgument {
                    arg: "k",
                    reason: "k cannot be greater than 1 for scalar tensors".to_string(),
                });
            }
            let indices = Tensor::<CudaRuntime>::try_zeros(shape, DType::I64, &self.device)?;
            return Ok((a.clone(), indices));
        }

        let dim_idx = normalize_dim(dim, ndim)?;
        let dim_size = shape[dim_idx];
        if k > dim_size {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!(
                    "k ({}) cannot be greater than dimension size ({})",
                    k, dim_size
                ),
            });
        }

        if k == 0 {
            let mut out_shape = shape.to_vec();
            out_shape[dim_idx] = 0;
            let out_values = Tensor::<CudaRuntime>::try_empty(&out_shape, dtype, &self.device)?;
            let out_indices =
                Tensor::<CudaRuntime>::try_empty(&out_shape, DType::I64, &self.device)?;
            return Ok((out_values, out_indices));
        }

        let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
        let a_contig = ensure_contiguous(a)?;

        let mut out_shape = shape.to_vec();
        out_shape[dim_idx] = k;

        let out_values = Tensor::<CudaRuntime>::try_empty(&out_shape, dtype, &self.device)?;
        let out_indices = Tensor::<CudaRuntime>::try_empty(&out_shape, DType::I64, &self.device)?;

        unsafe {
            launch_topk(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out_values.ptr(),
                out_indices.ptr(),
                outer_size,
                sort_size,
                inner_size,
                k,
                largest,
                sorted,
            )?;
        }

        Ok((out_values, out_indices))
    }

    fn unique(&self, a: &Tensor<CudaRuntime>, _sorted: bool) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let numel = a.numel();

        if numel == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[0], dtype, &self.device);
        }

        // Flatten and make contiguous
        let a_flat = a.reshape(&[numel])?;
        let a_contig = ensure_contiguous(&a_flat)?;

        // Sort first
        let sorted_tensor = self.sort(&a_contig, 0, false)?;

        // Allocate counter on device (using U32)
        let counter = Tensor::<CudaRuntime>::try_zeros(&[1], DType::U32, &self.device)?;

        // Count unique elements
        unsafe {
            launch_count_unique(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                sorted_tensor.ptr(),
                counter.ptr(),
                numel,
            )?;
        }

        // Synchronize and read count
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("CUDA sync failed: {:?}", e)))?;
        let count_data = counter.to_vec::<u32>();
        let unique_count = count_data[0] as usize;

        if unique_count == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[0], dtype, &self.device);
        }

        // Reset counter and allocate output
        let counter = Tensor::<CudaRuntime>::try_zeros(&[1], DType::U32, &self.device)?;
        let out = Tensor::<CudaRuntime>::try_empty(&[unique_count], dtype, &self.device)?;

        // Extract unique elements
        unsafe {
            launch_extract_unique(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                sorted_tensor.ptr(),
                out.ptr(),
                counter.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn unique_with_counts(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = a.dtype();
        let numel = a.numel();

        if numel == 0 {
            let unique = Tensor::<CudaRuntime>::try_empty(&[0], dtype, &self.device)?;
            let inverse = Tensor::<CudaRuntime>::try_empty(&[0], DType::I64, &self.device)?;
            let counts = Tensor::<CudaRuntime>::try_empty(&[0], DType::I64, &self.device)?;
            return Ok((unique, inverse, counts));
        }

        // Get unique values (GPU-native)
        let unique = self.unique(a, true)?;
        let unique_count = unique.numel();

        // Compute inverse indices via searchsorted (GPU-native)
        let a_flat = a.reshape(&[numel])?;
        let inverse = self.searchsorted(&unique, &a_flat, false)?;

        // Count occurrences using GPU bincount kernel (no CPU round-trip)
        let counts = Tensor::<CudaRuntime>::try_zeros(&[unique_count], DType::I64, &self.device)?;

        unsafe {
            launch_bincount(
                &self.context,
                &self.stream,
                self.device.index,
                inverse.ptr(),
                counts.ptr(),
                numel,
                unique_count,
            )?;
        }

        Ok((unique, inverse, counts))
    }

    fn nonzero(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();
        let numel = a.numel();

        if numel == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[0, ndim], DType::I64, &self.device);
        }

        let a_contig = ensure_contiguous(a)?;

        // Phase 1: Count nonzero elements
        let counter = Tensor::<CudaRuntime>::try_zeros(&[1], DType::U32, &self.device)?;

        unsafe {
            launch_count_nonzero(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                counter.ptr(),
                numel,
            )?;
        }

        // Synchronize and read count
        self.stream
            .synchronize()
            .map_err(|e| Error::Internal(format!("CUDA sync failed: {:?}", e)))?;
        let count_data = counter.to_vec::<u32>();
        let nnz = count_data[0] as usize;

        if nnz == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[0, ndim], DType::I64, &self.device);
        }

        if ndim == 0 {
            return Tensor::<CudaRuntime>::try_empty(&[1, 0], DType::I64, &self.device);
        }

        // Phase 2: Gather flat indices
        let counter = Tensor::<CudaRuntime>::try_zeros(&[1], DType::U32, &self.device)?;
        let flat_indices = Tensor::<CudaRuntime>::try_empty(&[nnz], DType::I64, &self.device)?;

        unsafe {
            launch_gather_nonzero(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                flat_indices.ptr(),
                counter.ptr(),
                numel,
            )?;
        }

        // Phase 3: Convert flat indices to multi-indices
        let shape_tensor = Tensor::<CudaRuntime>::try_from_slice(
            &shape.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            &[ndim],
            &self.device,
        )?;
        let out = Tensor::<CudaRuntime>::try_empty(&[nnz, ndim], DType::I64, &self.device)?;

        unsafe {
            launch_flat_to_multi_index(
                &self.context,
                &self.stream,
                self.device.index,
                flat_indices.ptr(),
                out.ptr(),
                nnz,
                ndim,
                shape_tensor.ptr(),
            )?;
        }

        Ok(out)
    }

    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        right: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        if sorted_sequence.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![sorted_sequence.numel()],
                got: sorted_sequence.shape().to_vec(),
            });
        }

        if sorted_sequence.dtype() != values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: sorted_sequence.dtype(),
                rhs: values.dtype(),
            });
        }

        let dtype = sorted_sequence.dtype();
        let seq_len = sorted_sequence.numel();
        let num_values = values.numel();

        if num_values == 0 {
            return Tensor::<CudaRuntime>::try_empty(values.shape(), DType::I64, &self.device);
        }

        let seq_contig = ensure_contiguous(sorted_sequence)?;
        let values_contig = ensure_contiguous(values)?;
        let out = Tensor::<CudaRuntime>::try_empty(values.shape(), DType::I64, &self.device)?;

        unsafe {
            launch_searchsorted(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seq_contig.ptr(),
                values_contig.ptr(),
                out.ptr(),
                seq_len,
                num_values,
                right,
            )?;
        }

        Ok(out)
    }
}
