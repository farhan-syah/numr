//! Sparse matrix element-wise merge operations for WebGPU.
//!
//! Implements CSR and CSC format merge operations (add, sub, mul, div) using
//! two-pass algorithms:
//! 1. Count pass: count output elements per row/column
//! 2. Exclusive scan: compute offsets
//! 3. Compute pass: perform merge and operation

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::{
    launch_csc_add_compute, launch_csc_div_compute, launch_csc_merge_count, launch_csc_mul_compute,
    launch_csc_mul_count, launch_csc_sub_compute, launch_csr_add_compute, launch_csr_div_compute,
    launch_csr_merge_count, launch_csr_mul_compute, launch_csr_mul_count, launch_csr_sub_compute,
    launch_exclusive_scan_i32,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::common::validate_wgpu_dtype;
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::tensor::Tensor;

/// Uniform buffer params for count kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CountParams {
    pub n: u32, // nrows for CSR, ncols for CSC
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for compute kernels
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParams {
    pub n: u32, // nrows for CSR, ncols for CSC
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Uniform buffer params for exclusive scan
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScanParams {
    pub n: u32, // number of elements to scan
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl WgpuClient {
    // =========================================================================
    // CSR Operations
    // =========================================================================

    /// CSR element-wise addition: C = A + B
    ///
    /// Two-pass algorithm:
    /// 1. Count output nnz per row (union semantics)
    /// 2. Exclusive scan to get row_ptrs
    /// 3. Merge and add values
    pub(crate) fn add_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_merge_binary_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeOp::Add,
        )
    }

    /// CSR element-wise subtraction: C = A - B
    pub(crate) fn sub_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_merge_binary_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeOp::Sub,
        )
    }

    /// CSR element-wise multiplication: C = A .* B (intersection semantics)
    pub(crate) fn mul_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_merge_binary_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeOp::Mul,
        )
    }

    /// CSR element-wise division: C = A ./ B (intersection semantics)
    pub(crate) fn div_csr_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_merge_binary_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeOp::Div,
        )
    }

    /// Internal CSR merge implementation
    fn csr_merge_binary_impl<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
        op: MergeOp,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [nrows, _ncols] = shape;
        let dtype = a_values.dtype();
        validate_wgpu_dtype(dtype, "csr_merge")?;

        // Step 1: Count output nnz per row
        let row_counts = Tensor::<WgpuRuntime>::try_zeros(&[nrows], DType::I32, &self.device_id)?;
        let count_params = CountParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let count_params_buffer = self.create_uniform_buffer("csr_count_params", 16);
        self.write_buffer(
            &count_params_buffer,
            &[
                count_params.n,
                count_params._pad0,
                count_params._pad1,
                count_params._pad2,
            ],
        );

        let a_row_ptrs_buf = get_tensor_buffer(a_row_ptrs)?;
        let a_col_indices_buf = get_tensor_buffer(a_col_indices)?;
        let b_row_ptrs_buf = get_tensor_buffer(b_row_ptrs)?;
        let b_col_indices_buf = get_tensor_buffer(b_col_indices)?;
        let row_counts_buf = get_tensor_buffer(&row_counts)?;

        // Launch appropriate count kernel
        if matches!(op, MergeOp::Mul | MergeOp::Div) {
            // Intersection semantics
            launch_csr_mul_count(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &a_row_ptrs_buf,
                &a_col_indices_buf,
                &b_row_ptrs_buf,
                &b_col_indices_buf,
                &row_counts_buf,
                &count_params_buffer,
                nrows,
            )?;
        } else {
            // Union semantics
            launch_csr_merge_count(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &a_row_ptrs_buf,
                &a_col_indices_buf,
                &b_row_ptrs_buf,
                &b_col_indices_buf,
                &row_counts_buf,
                &count_params_buffer,
                nrows,
            )?;
        }

        // Step 2: Exclusive scan to get output row_ptrs
        let out_row_ptrs =
            Tensor::<WgpuRuntime>::try_zeros(&[nrows + 1], DType::I32, &self.device_id)?;
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

        let out_row_ptrs_buf = get_tensor_buffer(&out_row_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_counts_buf,
            &out_row_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Read total nnz from last element of out_row_ptrs
        // Note: This requires a sync and GPU->CPU read. We'll estimate conservatively.
        // For now, we'll allocate worst-case (sum of both nnzs for union, min for intersection)
        let a_nnz = a_values.numel();
        let b_nnz = b_values.numel();
        let max_nnz = if matches!(op, MergeOp::Mul | MergeOp::Div) {
            // Intersection: at most min(a_nnz, b_nnz)
            std::cmp::min(a_nnz, b_nnz)
        } else {
            // Union: at most a_nnz + b_nnz
            a_nnz + b_nnz
        };

        // Allocate output arrays (may have unused space at end)
        let out_col_indices =
            Tensor::<WgpuRuntime>::try_zeros(&[max_nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::try_zeros(&[max_nnz], dtype, &self.device_id)?;

        let out_col_indices_buf = get_tensor_buffer(&out_col_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        // Step 4: Compute pass
        let compute_params = ComputeParams {
            n: nrows as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let compute_params_buffer = self.create_uniform_buffer("csr_compute_params", 16);
        self.write_buffer(
            &compute_params_buffer,
            &[
                compute_params.n,
                compute_params._pad0,
                compute_params._pad1,
                compute_params._pad2,
            ],
        );

        let a_values_buf = get_tensor_buffer(a_values)?;
        let b_values_buf = get_tensor_buffer(b_values)?;

        match op {
            MergeOp::Add => {
                launch_csr_add_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_row_ptrs_buf,
                    &a_col_indices_buf,
                    &a_values_buf,
                    &b_row_ptrs_buf,
                    &b_col_indices_buf,
                    &b_values_buf,
                    &out_row_ptrs_buf,
                    &out_col_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    nrows,
                    dtype,
                )?;
            }
            MergeOp::Sub => {
                launch_csr_sub_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_row_ptrs_buf,
                    &a_col_indices_buf,
                    &a_values_buf,
                    &b_row_ptrs_buf,
                    &b_col_indices_buf,
                    &b_values_buf,
                    &out_row_ptrs_buf,
                    &out_col_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    nrows,
                    dtype,
                )?;
            }
            MergeOp::Mul => {
                launch_csr_mul_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_row_ptrs_buf,
                    &a_col_indices_buf,
                    &a_values_buf,
                    &b_row_ptrs_buf,
                    &b_col_indices_buf,
                    &b_values_buf,
                    &out_row_ptrs_buf,
                    &out_col_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    nrows,
                    dtype,
                )?;
            }
            MergeOp::Div => {
                launch_csr_div_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_row_ptrs_buf,
                    &a_col_indices_buf,
                    &a_values_buf,
                    &b_row_ptrs_buf,
                    &b_col_indices_buf,
                    &b_values_buf,
                    &out_row_ptrs_buf,
                    &out_col_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    nrows,
                    dtype,
                )?;
            }
        }

        Ok((out_row_ptrs, out_col_indices, out_values))
    }

    // =========================================================================
    // CSC Operations
    // =========================================================================

    /// CSC element-wise addition
    pub(crate) fn add_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_merge_binary_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            MergeOp::Add,
        )
    }

    /// CSC element-wise subtraction
    pub(crate) fn sub_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_merge_binary_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            MergeOp::Sub,
        )
    }

    /// CSC element-wise multiplication
    pub(crate) fn mul_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_merge_binary_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            MergeOp::Mul,
        )
    }

    /// CSC element-wise division
    pub(crate) fn div_csc_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_merge_binary_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            MergeOp::Div,
        )
    }

    /// Internal CSC merge implementation
    fn csc_merge_binary_impl<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
        op: MergeOp,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        let [_nrows, ncols] = shape;
        let dtype = a_values.dtype();
        validate_wgpu_dtype(dtype, "csc_merge")?;

        // Step 1: Count output nnz per column
        let col_counts = Tensor::<WgpuRuntime>::try_zeros(&[ncols], DType::I32, &self.device_id)?;
        let count_params = CountParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let count_params_buffer = self.create_uniform_buffer("csc_count_params", 16);
        self.write_buffer(
            &count_params_buffer,
            &[
                count_params.n,
                count_params._pad0,
                count_params._pad1,
                count_params._pad2,
            ],
        );

        let a_col_ptrs_buf = get_tensor_buffer(a_col_ptrs)?;
        let a_row_indices_buf = get_tensor_buffer(a_row_indices)?;
        let b_col_ptrs_buf = get_tensor_buffer(b_col_ptrs)?;
        let b_row_indices_buf = get_tensor_buffer(b_row_indices)?;
        let col_counts_buf = get_tensor_buffer(&col_counts)?;

        if matches!(op, MergeOp::Mul | MergeOp::Div) {
            launch_csc_mul_count(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &a_col_ptrs_buf,
                &a_row_indices_buf,
                &b_col_ptrs_buf,
                &b_row_indices_buf,
                &col_counts_buf,
                &count_params_buffer,
                ncols,
            )?;
        } else {
            launch_csc_merge_count(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &a_col_ptrs_buf,
                &a_row_indices_buf,
                &b_col_ptrs_buf,
                &b_row_indices_buf,
                &col_counts_buf,
                &count_params_buffer,
                ncols,
            )?;
        }

        // Step 2: Exclusive scan
        let out_col_ptrs =
            Tensor::<WgpuRuntime>::try_zeros(&[ncols + 1], DType::I32, &self.device_id)?;
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

        let out_col_ptrs_buf = get_tensor_buffer(&out_col_ptrs)?;

        launch_exclusive_scan_i32(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &col_counts_buf,
            &out_col_ptrs_buf,
            &scan_params_buffer,
        )?;

        // Step 3: Allocate output
        let a_nnz = a_values.numel();
        let b_nnz = b_values.numel();
        let max_nnz = if matches!(op, MergeOp::Mul | MergeOp::Div) {
            std::cmp::min(a_nnz, b_nnz)
        } else {
            a_nnz + b_nnz
        };

        let out_row_indices =
            Tensor::<WgpuRuntime>::try_zeros(&[max_nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::try_zeros(&[max_nnz], dtype, &self.device_id)?;

        let out_row_indices_buf = get_tensor_buffer(&out_row_indices)?;
        let out_values_buf = get_tensor_buffer(&out_values)?;

        // Step 4: Compute
        let compute_params = ComputeParams {
            n: ncols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let compute_params_buffer = self.create_uniform_buffer("csc_compute_params", 16);
        self.write_buffer(
            &compute_params_buffer,
            &[
                compute_params.n,
                compute_params._pad0,
                compute_params._pad1,
                compute_params._pad2,
            ],
        );

        let a_values_buf = get_tensor_buffer(a_values)?;
        let b_values_buf = get_tensor_buffer(b_values)?;

        match op {
            MergeOp::Add => {
                launch_csc_add_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_col_ptrs_buf,
                    &a_row_indices_buf,
                    &a_values_buf,
                    &b_col_ptrs_buf,
                    &b_row_indices_buf,
                    &b_values_buf,
                    &out_col_ptrs_buf,
                    &out_row_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    ncols,
                    dtype,
                )?;
            }
            MergeOp::Sub => {
                launch_csc_sub_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_col_ptrs_buf,
                    &a_row_indices_buf,
                    &a_values_buf,
                    &b_col_ptrs_buf,
                    &b_row_indices_buf,
                    &b_values_buf,
                    &out_col_ptrs_buf,
                    &out_row_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    ncols,
                    dtype,
                )?;
            }
            MergeOp::Mul => {
                launch_csc_mul_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_col_ptrs_buf,
                    &a_row_indices_buf,
                    &a_values_buf,
                    &b_col_ptrs_buf,
                    &b_row_indices_buf,
                    &b_values_buf,
                    &out_col_ptrs_buf,
                    &out_row_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    ncols,
                    dtype,
                )?;
            }
            MergeOp::Div => {
                launch_csc_div_compute(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &a_col_ptrs_buf,
                    &a_row_indices_buf,
                    &a_values_buf,
                    &b_col_ptrs_buf,
                    &b_row_indices_buf,
                    &b_values_buf,
                    &out_col_ptrs_buf,
                    &out_row_indices_buf,
                    &out_values_buf,
                    &compute_params_buffer,
                    ncols,
                    dtype,
                )?;
            }
        }

        Ok((out_col_ptrs, out_row_indices, out_values))
    }

    // =========================================================================
    // COO Operations (convert to CSR, operate, optionally convert back)
    // =========================================================================

    /// COO element-wise addition - converts to CSR, adds, returns as CSR
    pub(crate) fn add_coo_impl<T: Element>(
        &self,
        _a_row_indices: &Tensor<WgpuRuntime>,
        _a_col_indices: &Tensor<WgpuRuntime>,
        _a_values: &Tensor<WgpuRuntime>,
        _b_row_indices: &Tensor<WgpuRuntime>,
        _b_col_indices: &Tensor<WgpuRuntime>,
        _b_values: &Tensor<WgpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        // COO operations require format conversion which isn't implemented yet
        Err(crate::error::Error::NotImplemented {
            feature: "WebGPU COO merge operations - requires format conversion",
        })
    }

    /// COO element-wise subtraction
    pub(crate) fn sub_coo_impl<T: Element>(
        &self,
        _a_row_indices: &Tensor<WgpuRuntime>,
        _a_col_indices: &Tensor<WgpuRuntime>,
        _a_values: &Tensor<WgpuRuntime>,
        _b_row_indices: &Tensor<WgpuRuntime>,
        _b_col_indices: &Tensor<WgpuRuntime>,
        _b_values: &Tensor<WgpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(crate::error::Error::NotImplemented {
            feature: "WebGPU COO merge operations - requires format conversion",
        })
    }

    /// COO element-wise multiplication
    pub(crate) fn mul_coo_impl<T: Element>(
        &self,
        _a_row_indices: &Tensor<WgpuRuntime>,
        _a_col_indices: &Tensor<WgpuRuntime>,
        _a_values: &Tensor<WgpuRuntime>,
        _b_row_indices: &Tensor<WgpuRuntime>,
        _b_col_indices: &Tensor<WgpuRuntime>,
        _b_values: &Tensor<WgpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(crate::error::Error::NotImplemented {
            feature: "WebGPU COO merge operations - requires format conversion",
        })
    }

    /// COO element-wise division
    pub(crate) fn div_coo_impl<T: Element>(
        &self,
        _a_row_indices: &Tensor<WgpuRuntime>,
        _a_col_indices: &Tensor<WgpuRuntime>,
        _a_values: &Tensor<WgpuRuntime>,
        _b_row_indices: &Tensor<WgpuRuntime>,
        _b_col_indices: &Tensor<WgpuRuntime>,
        _b_values: &Tensor<WgpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(crate::error::Error::NotImplemented {
            feature: "WebGPU COO merge operations - requires format conversion",
        })
    }
}

/// Merge operation type
#[derive(Debug, Clone, Copy)]
enum MergeOp {
    Add,
    Sub,
    Mul,
    Div,
}
