//! Sparse format conversion operations for WebGPU.
//!
//! Implements conversions between COO, CSR, and CSC sparse matrix formats.
//! Also provides sparse↔dense conversions.
//! All operations stay on GPU without data transfers.
//!
//! # Algorithms
//!
//! - **CSR/CSC → COO**: Parallel pointer expansion
//! - **COO → CSR/CSC**: Histogram + exclusive scan + scatter
//! - **CSR ↔ CSC**: Direct transpose via histogram + scan + scatter
//! - **CSR → Dense**: Parallel scatter per row
//! - **Dense → COO**: Count + atomic scatter

use super::super::ops::helpers::{get_tensor_buffer, read_u32_from_buffer};
use super::super::shaders::{
    launch_coo_to_csc_scatter, launch_coo_to_csr_scatter, launch_copy_ptrs, launch_count_nonzeros,
    launch_csc_to_csr_scatter, launch_csr_to_csc_scatter, launch_csr_to_dense,
    launch_dense_to_coo_scatter, launch_exclusive_scan_i32, launch_expand_col_ptrs,
    launch_expand_row_ptrs, launch_histogram,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::common::validate_wgpu_dtype;
use super::merge::ScanParams;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::sparse::{CooData, SparseTensor};
use crate::tensor::Tensor;

/// Uniform buffer params for expand kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExpandParams {
    pub n: u32, // nrows or ncols
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for histogram kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HistogramParams {
    pub nnz: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for scatter kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterParams {
    pub nnz: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for transpose scatter kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransposeParams {
    pub n: u32, // nrows or ncols
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for copy kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CopyParams {
    pub n: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for CSR to dense kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CsrToDenseParams {
    pub nrows: u32,
    pub ncols: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Uniform buffer params for count non-zeros kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CountNonzerosParams {
    pub total_elems: u32,
    pub threshold_bits: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Uniform buffer params for dense to COO scatter kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenseToCooParams {
    pub nrows: u32,
    pub ncols: u32,
    pub threshold_bits: u32,
    pub _pad0: u32,
}

impl WgpuClient {
    /// CSR → COO: Expand row pointers to explicit row indices.
    ///
    /// # Arguments
    /// - `row_ptrs`: [nrows + 1] row pointers
    /// - `col_indices`: [nnz] column indices
    /// - `values`: [nnz] values
    /// - `shape`: [nrows, ncols]
    ///
    /// # Returns
    /// (row_indices, col_indices, values) in COO format
    pub(crate) fn csr_to_coo_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();

        // Allocate output row_indices
        let row_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;

        // Setup params
        let expand_params = ExpandParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let expand_params_buffer = self.create_uniform_buffer("expand_row_params", 16);
        self.write_buffer(
            &expand_params_buffer,
            &[
                expand_params.n,
                expand_params._pad0,
                expand_params._pad1,
                expand_params._pad2,
            ],
        );

        let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
        let row_indices_buf = get_tensor_buffer(&row_indices)?;

        // Launch expand kernel
        launch_expand_row_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buf,
            &row_indices_buf,
            &expand_params_buffer,
            nrows,
        )?;

        // col_indices and values stay the same
        Ok((row_indices, col_indices.clone(), values.clone()))
    }

    /// CSC → COO: Expand column pointers to explicit column indices.
    pub(crate) fn csc_to_coo_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();

        // Allocate output col_indices
        let col_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;

        // Setup params
        let expand_params = ExpandParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let expand_params_buffer = self.create_uniform_buffer("expand_col_params", 16);
        self.write_buffer(
            &expand_params_buffer,
            &[
                expand_params.n,
                expand_params._pad0,
                expand_params._pad1,
                expand_params._pad2,
            ],
        );

        let col_ptrs_buf = get_tensor_buffer(col_ptrs)?;
        let col_indices_buf = get_tensor_buffer(&col_indices)?;

        // Launch expand kernel
        launch_expand_col_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_ptrs_buf,
            &col_indices_buf,
            &expand_params_buffer,
            ncols,
        )?;

        // row_indices and values stay the same
        Ok((row_indices.clone(), col_indices, values.clone()))
    }

    /// COO → CSR: Histogram + exclusive scan + scatter.
    ///
    /// # Algorithm
    /// 1. Count elements per row (histogram)
    /// 2. Exclusive scan to build row_ptrs
    /// 3. Copy row_ptrs for atomic scatter
    /// 4. Scatter elements to their positions
    pub(crate) fn coo_to_csr_impl<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();
        let dtype = values.dtype();
        validate_wgpu_dtype(dtype, "coo_to_csr")?;

        // Step 1: Count elements per row (histogram)
        let row_counts = Tensor::<WgpuRuntime>::zeros(&[nrows], DType::I32, &self.device_id)?;
        let histogram_params = HistogramParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let histogram_params_buffer = self.create_uniform_buffer("histogram_params", 16);
        self.write_buffer(
            &histogram_params_buffer,
            &[
                histogram_params.nnz,
                histogram_params._pad0,
                histogram_params._pad1,
                histogram_params._pad2,
            ],
        );

        let row_indices_buf = get_tensor_buffer(row_indices)?;
        let row_counts_buf = get_tensor_buffer(&row_counts)?;

        launch_histogram(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_indices_buf,
            &row_counts_buf,
            &histogram_params_buffer,
            nnz,
        )?;

        // Step 2: Exclusive scan to build row_ptrs
        let row_ptrs = Tensor::<WgpuRuntime>::zeros(&[nrows + 1], DType::I32, &self.device_id)?;
        let scan_params = ScanParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scan_params_buffer = self.create_uniform_buffer("scan_params", 16);
        self.write_buffer(
            &scan_params_buffer,
            &[
                scan_params.n,
                scan_params._pad0,
                scan_params._pad1,
                scan_params._pad2,
            ],
        );

        let row_ptrs_buf = get_tensor_buffer(&row_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_counts_buf,
            &row_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Copy row_ptrs for atomic scatter (scatter modifies positions atomically)
        let row_ptrs_atomic =
            Tensor::<WgpuRuntime>::zeros(&[nrows + 1], DType::I32, &self.device_id)?;
        let copy_params = CopyParams {
            n: (nrows + 1) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let copy_params_buffer = self.create_uniform_buffer("copy_params", 16);
        self.write_buffer(
            &copy_params_buffer,
            &[
                copy_params.n,
                copy_params._pad0,
                copy_params._pad1,
                copy_params._pad2,
            ],
        );

        let row_ptrs_atomic_buf = get_tensor_buffer(&row_ptrs_atomic)?;

        launch_copy_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buf,
            &row_ptrs_atomic_buf,
            &copy_params_buffer,
            nrows + 1,
        )?;

        // Step 4: Scatter elements to their positions
        let out_col_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[nnz], dtype, &self.device_id)?;
        let scatter_params = ScatterParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scatter_params_buffer = self.create_uniform_buffer("scatter_params", 16);
        self.write_buffer(
            &scatter_params_buffer,
            &[
                scatter_params.nnz,
                scatter_params._pad0,
                scatter_params._pad1,
                scatter_params._pad2,
            ],
        );

        let col_indices_buf = get_tensor_buffer(col_indices)?;
        let values_buf = get_tensor_buffer(values)?;
        let out_col_indices_buf = get_tensor_buffer(&out_col_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        launch_coo_to_csr_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_indices_buf,
            &col_indices_buf,
            &values_buf,
            &row_ptrs_atomic_buf,
            &out_col_indices_buf,
            &out_values_buf,
            &scatter_params_buffer,
            nnz,
            dtype,
        )?;

        Ok((row_ptrs, out_col_indices, out_values))
    }

    /// COO → CSC: Histogram + exclusive scan + scatter.
    pub(crate) fn coo_to_csc_impl<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();
        let dtype = values.dtype();
        validate_wgpu_dtype(dtype, "coo_to_csc")?;

        // Step 1: Count elements per column (histogram)
        let col_counts = Tensor::<WgpuRuntime>::zeros(&[ncols], DType::I32, &self.device_id)?;
        let histogram_params = HistogramParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let histogram_params_buffer = self.create_uniform_buffer("histogram_params", 16);
        self.write_buffer(
            &histogram_params_buffer,
            &[
                histogram_params.nnz,
                histogram_params._pad0,
                histogram_params._pad1,
                histogram_params._pad2,
            ],
        );

        let col_indices_buf = get_tensor_buffer(col_indices)?;
        let col_counts_buf = get_tensor_buffer(&col_counts)?;

        launch_histogram(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_indices_buf,
            &col_counts_buf,
            &histogram_params_buffer,
            nnz,
        )?;

        // Step 2: Exclusive scan to build col_ptrs
        let col_ptrs = Tensor::<WgpuRuntime>::zeros(&[ncols + 1], DType::I32, &self.device_id)?;
        let scan_params = ScanParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scan_params_buffer = self.create_uniform_buffer("scan_params", 16);
        self.write_buffer(
            &scan_params_buffer,
            &[
                scan_params.n,
                scan_params._pad0,
                scan_params._pad1,
                scan_params._pad2,
            ],
        );

        let col_ptrs_buf = get_tensor_buffer(&col_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_counts_buf,
            &col_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Copy col_ptrs for atomic scatter
        let col_ptrs_atomic =
            Tensor::<WgpuRuntime>::zeros(&[ncols + 1], DType::I32, &self.device_id)?;
        let copy_params = CopyParams {
            n: (ncols + 1) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let copy_params_buffer = self.create_uniform_buffer("copy_params", 16);
        self.write_buffer(
            &copy_params_buffer,
            &[
                copy_params.n,
                copy_params._pad0,
                copy_params._pad1,
                copy_params._pad2,
            ],
        );

        let col_ptrs_atomic_buf = get_tensor_buffer(&col_ptrs_atomic)?;

        launch_copy_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_ptrs_buf,
            &col_ptrs_atomic_buf,
            &copy_params_buffer,
            ncols + 1,
        )?;

        // Step 4: Scatter elements to their positions
        let out_row_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[nnz], dtype, &self.device_id)?;
        let scatter_params = ScatterParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scatter_params_buffer = self.create_uniform_buffer("scatter_params", 16);
        self.write_buffer(
            &scatter_params_buffer,
            &[
                scatter_params.nnz,
                scatter_params._pad0,
                scatter_params._pad1,
                scatter_params._pad2,
            ],
        );

        let row_indices_buf = get_tensor_buffer(row_indices)?;
        let values_buf = get_tensor_buffer(values)?;
        let out_row_indices_buf = get_tensor_buffer(&out_row_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        launch_coo_to_csc_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_indices_buf,
            &col_indices_buf,
            &values_buf,
            &col_ptrs_atomic_buf,
            &out_row_indices_buf,
            &out_values_buf,
            &scatter_params_buffer,
            nnz,
            dtype,
        )?;

        Ok((col_ptrs, out_row_indices, out_values))
    }

    /// CSR → CSC: Direct transpose via histogram + scan + scatter.
    ///
    /// More efficient than CSR→COO→CSC as it avoids the intermediate format.
    pub(crate) fn csr_to_csc_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let dtype = values.dtype();
        validate_wgpu_dtype(dtype, "csr_to_csc")?;

        // Step 1: Count elements per column (histogram over col_indices)
        let col_counts = Tensor::<WgpuRuntime>::zeros(&[ncols], DType::I32, &self.device_id)?;
        let histogram_params = HistogramParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let histogram_params_buffer = self.create_uniform_buffer("histogram_params", 16);
        self.write_buffer(
            &histogram_params_buffer,
            &[
                histogram_params.nnz,
                histogram_params._pad0,
                histogram_params._pad1,
                histogram_params._pad2,
            ],
        );

        let col_indices_buf = get_tensor_buffer(col_indices)?;
        let col_counts_buf = get_tensor_buffer(&col_counts)?;

        launch_histogram(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_indices_buf,
            &col_counts_buf,
            &histogram_params_buffer,
            nnz,
        )?;

        // Step 2: Exclusive scan to build col_ptrs
        let col_ptrs = Tensor::<WgpuRuntime>::zeros(&[ncols + 1], DType::I32, &self.device_id)?;
        let scan_params = ScanParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scan_params_buffer = self.create_uniform_buffer("scan_params", 16);
        self.write_buffer(
            &scan_params_buffer,
            &[
                scan_params.n,
                scan_params._pad0,
                scan_params._pad1,
                scan_params._pad2,
            ],
        );

        let col_ptrs_buf = get_tensor_buffer(&col_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_counts_buf,
            &col_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Copy col_ptrs for atomic scatter
        let col_ptrs_atomic =
            Tensor::<WgpuRuntime>::zeros(&[ncols + 1], DType::I32, &self.device_id)?;
        let copy_params = CopyParams {
            n: (ncols + 1) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let copy_params_buffer = self.create_uniform_buffer("copy_params", 16);
        self.write_buffer(
            &copy_params_buffer,
            &[
                copy_params.n,
                copy_params._pad0,
                copy_params._pad1,
                copy_params._pad2,
            ],
        );

        let col_ptrs_atomic_buf = get_tensor_buffer(&col_ptrs_atomic)?;

        launch_copy_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_ptrs_buf,
            &col_ptrs_atomic_buf,
            &copy_params_buffer,
            ncols + 1,
        )?;

        // Step 4: Scatter CSR elements to CSC positions
        let out_row_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[nnz], dtype, &self.device_id)?;
        let transpose_params = TransposeParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let transpose_params_buffer = self.create_uniform_buffer("transpose_params", 16);
        self.write_buffer(
            &transpose_params_buffer,
            &[
                transpose_params.n,
                transpose_params._pad0,
                transpose_params._pad1,
                transpose_params._pad2,
            ],
        );

        let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
        let values_buf = get_tensor_buffer(values)?;
        let out_row_indices_buf = get_tensor_buffer(&out_row_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        launch_csr_to_csc_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buf,
            &col_indices_buf,
            &values_buf,
            &col_ptrs_atomic_buf,
            &out_row_indices_buf,
            &out_values_buf,
            &transpose_params_buffer,
            nrows,
            dtype,
        )?;

        Ok((col_ptrs, out_row_indices, out_values))
    }

    /// CSC → CSR: Direct transpose via histogram + scan + scatter.
    pub(crate) fn csc_to_csr_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let dtype = values.dtype();
        validate_wgpu_dtype(dtype, "csc_to_csr")?;

        // Step 1: Count elements per row (histogram over row_indices)
        let row_counts = Tensor::<WgpuRuntime>::zeros(&[nrows], DType::I32, &self.device_id)?;
        let histogram_params = HistogramParams {
            nnz: nnz as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let histogram_params_buffer = self.create_uniform_buffer("histogram_params", 16);
        self.write_buffer(
            &histogram_params_buffer,
            &[
                histogram_params.nnz,
                histogram_params._pad0,
                histogram_params._pad1,
                histogram_params._pad2,
            ],
        );

        let row_indices_buf = get_tensor_buffer(row_indices)?;
        let row_counts_buf = get_tensor_buffer(&row_counts)?;

        launch_histogram(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_indices_buf,
            &row_counts_buf,
            &histogram_params_buffer,
            nnz,
        )?;

        // Step 2: Exclusive scan to build row_ptrs
        let row_ptrs = Tensor::<WgpuRuntime>::zeros(&[nrows + 1], DType::I32, &self.device_id)?;
        let scan_params = ScanParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let scan_params_buffer = self.create_uniform_buffer("scan_params", 16);
        self.write_buffer(
            &scan_params_buffer,
            &[
                scan_params.n,
                scan_params._pad0,
                scan_params._pad1,
                scan_params._pad2,
            ],
        );

        let row_ptrs_buf = get_tensor_buffer(&row_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_counts_buf,
            &row_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Copy row_ptrs for atomic scatter
        let row_ptrs_atomic =
            Tensor::<WgpuRuntime>::zeros(&[nrows + 1], DType::I32, &self.device_id)?;
        let copy_params = CopyParams {
            n: (nrows + 1) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let copy_params_buffer = self.create_uniform_buffer("copy_params", 16);
        self.write_buffer(
            &copy_params_buffer,
            &[
                copy_params.n,
                copy_params._pad0,
                copy_params._pad1,
                copy_params._pad2,
            ],
        );

        let row_ptrs_atomic_buf = get_tensor_buffer(&row_ptrs_atomic)?;

        launch_copy_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buf,
            &row_ptrs_atomic_buf,
            &copy_params_buffer,
            nrows + 1,
        )?;

        // Step 4: Scatter CSC elements to CSR positions
        let out_col_indices = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[nnz], dtype, &self.device_id)?;
        let transpose_params = TransposeParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let transpose_params_buffer = self.create_uniform_buffer("transpose_params", 16);
        self.write_buffer(
            &transpose_params_buffer,
            &[
                transpose_params.n,
                transpose_params._pad0,
                transpose_params._pad1,
                transpose_params._pad2,
            ],
        );

        let col_ptrs_buf = get_tensor_buffer(col_ptrs)?;
        let values_buf = get_tensor_buffer(values)?;
        let out_col_indices_buf = get_tensor_buffer(&out_col_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        launch_csc_to_csr_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_ptrs_buf,
            &row_indices_buf,
            &values_buf,
            &row_ptrs_atomic_buf,
            &out_col_indices_buf,
            &out_values_buf,
            &transpose_params_buffer,
            ncols,
            dtype,
        )?;

        Ok((row_ptrs, out_col_indices, out_values))
    }

    /// CSR → Dense: Scatter sparse values into dense matrix.
    ///
    /// # Arguments
    /// - `row_ptrs`: [nrows + 1] row pointers
    /// - `col_indices`: [nnz] column indices
    /// - `values`: [nnz] values
    /// - `shape`: [nrows, ncols]
    ///
    /// # Returns
    /// Dense tensor of shape [nrows, ncols]
    pub(crate) fn sparse_to_dense_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        let [nrows, ncols] = shape;
        let dtype = values.dtype();
        validate_wgpu_dtype(dtype, "sparse_to_dense")?;

        // Allocate dense output initialized to zeros
        let dense = Tensor::<WgpuRuntime>::zeros(&[nrows, ncols], dtype, &self.device_id)?;

        // Setup params
        let params = CsrToDenseParams {
            nrows: nrows as u32,
            ncols: ncols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self.create_uniform_buffer("csr_to_dense_params", 16);
        self.write_buffer(
            &params_buffer,
            &[params.nrows, params.ncols, params._pad0, params._pad1],
        );

        let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
        let col_indices_buf = get_tensor_buffer(col_indices)?;
        let values_buf = get_tensor_buffer(values)?;
        let dense_buf = get_tensor_buffer(&dense)?;

        // Launch kernel
        launch_csr_to_dense(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buf,
            &col_indices_buf,
            &values_buf,
            &dense_buf,
            &params_buffer,
            nrows,
            dtype,
        )?;

        Ok(dense)
    }

    /// Dense → COO: Count non-zeros + atomic scatter.
    ///
    /// # Algorithm
    /// 1. Count non-zeros using atomic counter
    /// 2. Read back total_nnz (single u32 transfer)
    /// 3. Allocate output COO tensors
    /// 4. Scatter non-zero elements using atomic write position
    ///
    /// # Arguments
    /// - `dense`: 2D dense tensor [nrows, ncols]
    /// - `threshold`: Values with |value| < threshold become zero
    ///
    /// # Returns
    /// SparseTensor in COO format
    pub(crate) fn dense_to_coo_impl(
        &self,
        dense: &Tensor<WgpuRuntime>,
        threshold: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Validate input
        if dense.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for dense_to_sparse, got {}D",
                dense.ndim()
            )));
        }

        let shape = dense.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        let total_elems = nrows * ncols;
        let dtype = dense.dtype();
        let device = dense.device();

        validate_wgpu_dtype(dtype, "dense_to_sparse")?;

        // Convert threshold to appropriate bits representation
        let threshold_bits = match dtype {
            DType::F32 => (threshold as f32).to_bits(),
            #[cfg(feature = "f16")]
            DType::F16 => {
                let f16_val = half::f16::from_f64(threshold);
                f16_val.to_bits() as u32
            }
            DType::I32 | DType::U32 => threshold as u32,
            _ => {
                return Err(Error::NotImplemented {
                    feature: "dense_to_sparse for this dtype",
                });
            }
        };

        // Step 1: Create atomic counter buffer (initialized to 0)
        let count_buffer = self.create_storage_buffer("nnz_count", 4);
        self.queue.write_buffer(&count_buffer, 0, &[0u8; 4]);

        // Setup count params
        let count_params = CountNonzerosParams {
            total_elems: total_elems as u32,
            threshold_bits,
            _pad0: 0,
            _pad1: 0,
        };
        let count_params_buffer = self.create_uniform_buffer("count_params", 16);
        self.write_buffer(
            &count_params_buffer,
            &[
                count_params.total_elems,
                count_params.threshold_bits,
                count_params._pad0,
                count_params._pad1,
            ],
        );

        let dense_buf = get_tensor_buffer(dense)?;

        // Launch count kernel
        launch_count_nonzeros(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &dense_buf,
            &count_buffer,
            &count_params_buffer,
            total_elems,
            dtype,
        )?;

        // Step 2: Read back total_nnz (single u32 transfer - unavoidable)
        let total_nnz = read_u32_from_buffer(self, &count_buffer)? as usize;

        // Handle empty case
        if total_nnz == 0 {
            return Ok(SparseTensor::Coo(CooData::empty(
                [nrows, ncols],
                dtype,
                device,
            )?));
        }

        // Step 3: Allocate output COO tensors
        // Note: WGSL shader uses i32 for indices, we create I32 tensors
        let row_indices = Tensor::<WgpuRuntime>::zeros(&[total_nnz], DType::I32, &self.device_id)?;
        let col_indices = Tensor::<WgpuRuntime>::zeros(&[total_nnz], DType::I32, &self.device_id)?;
        let values = Tensor::<WgpuRuntime>::zeros(&[total_nnz], dtype, &self.device_id)?;

        // Create atomic write position buffer (initialized to 0)
        let write_pos_buffer = self.create_storage_buffer("write_pos", 4);
        self.queue.write_buffer(&write_pos_buffer, 0, &[0u8; 4]);

        // Setup scatter params
        let scatter_params = DenseToCooParams {
            nrows: nrows as u32,
            ncols: ncols as u32,
            threshold_bits,
            _pad0: 0,
        };
        let scatter_params_buffer = self.create_uniform_buffer("scatter_params", 16);
        self.write_buffer(
            &scatter_params_buffer,
            &[
                scatter_params.nrows,
                scatter_params.ncols,
                scatter_params.threshold_bits,
                scatter_params._pad0,
            ],
        );

        let row_indices_buf = get_tensor_buffer(&row_indices)?;
        let col_indices_buf = get_tensor_buffer(&col_indices)?;
        let values_buf = get_tensor_buffer(&values)?;

        // Step 4: Launch scatter kernel
        launch_dense_to_coo_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &dense_buf,
            &row_indices_buf,
            &col_indices_buf,
            &values_buf,
            &write_pos_buffer,
            &scatter_params_buffer,
            total_elems,
            dtype,
        )?;

        // Convert I32 indices to I64 for CooData
        // Note: CooData expects I64 indices, but WGSL only supports i32
        // We need to cast them
        let row_indices_i64 = self.cast(&row_indices, DType::I64)?;
        let col_indices_i64 = self.cast(&col_indices, DType::I64)?;

        // Create COO data
        // Note: The scatter kernel doesn't guarantee sorted order due to atomic operations
        let coo = CooData::new(row_indices_i64, col_indices_i64, values, [nrows, ncols])?;

        Ok(SparseTensor::Coo(coo))
    }
}
