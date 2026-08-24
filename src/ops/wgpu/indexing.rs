//! Indexing operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IndexingOps, ReduceOps, ScalarOps, ScatterReduceOp, TypeConversionOps};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    BincountParams, Gather2dParams, GatherNdParams, MeanDivParams, ScatterReduceParams,
    alloc_output, create_params_buffer, ensure_i32_indices, get_tensor_buffer,
};
use crate::runtime::wgpu::ops::native::{
    native_argreduce_op, native_embedding_lookup, native_gather, native_index_put,
    native_index_select, native_masked_fill, native_masked_select, native_scatter,
    native_slice_assign,
};
use crate::runtime::wgpu::shaders::{
    launch_bincount, launch_gather_2d, launch_gather_nd, launch_scatter_reduce,
    launch_scatter_reduce_count, launch_scatter_reduce_mean_div, launch_scatter_reduce_prod,
};
use crate::tensor::Tensor;

impl IndexingOps<WgpuRuntime> for WgpuClient {
    fn argmax(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmax", a, dim, keepdim)
    }

    fn argmin(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmin", a, dim, keepdim)
    }

    fn gather(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_gather(self, a, dim, index)
    }

    fn scatter(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_scatter(self, a, dim, index, src)
    }

    fn index_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_index_select(self, a, dim, index)
    }

    fn index_put(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_index_put(self, a, dim, index, src)
    }

    fn masked_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_select(self, a, mask)
    }

    fn masked_fill(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
        value: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_fill(self, a, mask, value)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_embedding_lookup(self, embeddings, indices)
    }

    fn scatter_reduce(
        &self,
        dst: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
        op: ScatterReduceOp,
        include_self: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = dst.dtype();

        // Only float types supported for scatter_reduce on WebGPU
        // (atomics use CAS loops with bitcast for floats)
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_reduce",
            });
        }

        // Mean only supports F32 on WebGPU
        if matches!(op, ScatterReduceOp::Mean) && dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_reduce_mean",
            });
        }

        // Validate index dtype
        if !matches!(index.dtype(), DType::I32 | DType::I64) {
            return Err(Error::InvalidArgument {
                arg: "index",
                reason: "scatter_reduce index must be I32 or I64".to_string(),
            });
        }

        // Element-wise semantics (matching CPU/CUDA/PyTorch): index and src must
        // have the same shape — `index[e]` gives the destination coordinate along
        // `dim` for source element `e`.
        if index.shape() != src.shape() {
            return Err(Error::ShapeMismatch {
                expected: src.shape().to_vec(),
                got: index.shape().to_vec(),
            });
        }

        // Ensure contiguous
        let dst = ensure_contiguous(dst)?;
        let index_i32 = ensure_i32_indices(self, index)?;
        let index = ensure_contiguous(&index_i32)?;
        let src = ensure_contiguous(src)?;

        // Compute shape parameters
        let dst_shape = dst.shape();
        let ndim = dst_shape.len();
        if dim >= ndim {
            return Err(Error::InvalidArgument {
                arg: "dim",
                reason: format!("dim {} out of bounds for tensor with {} dims", dim, ndim),
            });
        }

        let outer_size: usize = dst_shape[..dim].iter().product();
        let dim_size = dst_shape[dim];
        let inner_size: usize = dst_shape[dim + 1..].iter().product();
        let src_dim_size = src.shape().get(dim).copied().unwrap_or(1);
        let total_src = src.numel();

        // Initialize output with identity for the operation
        let identity = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => 0.0f64,
            ScatterReduceOp::Max => f64::NEG_INFINITY,
            ScatterReduceOp::Min => f64::INFINITY,
            ScatterReduceOp::Prod => 1.0,
        };
        let output = if include_self {
            // Must deep-copy: clone() shares the GPU buffer, but scatter_reduce
            // modifies it in-place via atomics, which would corrupt the original.
            self.add_scalar(&dst, 0.0)?
        } else {
            Tensor::full_scalar(dst_shape, dtype, identity, self.device())
        };

        // Create shared params
        let params = ScatterReduceParams {
            dim: dim as u32,
            outer_size: outer_size as u32,
            dim_size: dim_size as u32,
            inner_size: inner_size as u32,
            src_dim_size: src_dim_size as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        let src_buf = get_tensor_buffer(&src)?;
        let index_buf = get_tensor_buffer(&index)?;
        let output_buf = get_tensor_buffer(&output)?;

        match op {
            ScatterReduceOp::Prod => {
                launch_scatter_reduce_prod(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &src_buf,
                    &index_buf,
                    &output_buf,
                    &params_buf,
                    total_src,
                    dtype,
                )?;
                Ok(output)
            }
            ScatterReduceOp::Mean => {
                // Step 1: scatter sum
                launch_scatter_reduce(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &src_buf,
                    &index_buf,
                    &output_buf,
                    &params_buf,
                    total_src,
                    dtype,
                    "sum",
                )?;

                // Step 2: scatter count (u32 buffer)
                let numel = dst.numel();
                let count_init = if include_self { 1u32 } else { 0u32 };
                let count_data = vec![count_init; numel];
                let count_tensor =
                    Tensor::<WgpuRuntime>::from_slice(&count_data, dst_shape, self.device());
                let count_buf = get_tensor_buffer(&count_tensor)?;

                launch_scatter_reduce_count(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &index_buf,
                    &count_buf,
                    &params_buf,
                    total_src,
                    dtype,
                )?;

                // Step 3: divide sum by count
                let result = alloc_output(self, dst_shape, dtype)?;
                let result_buf = get_tensor_buffer(&result)?;

                let mean_params = MeanDivParams {
                    n: numel as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                let mean_params_buf = create_params_buffer(self, &mean_params);

                launch_scatter_reduce_mean_div(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &output_buf,
                    &count_buf,
                    &result_buf,
                    &mean_params_buf,
                    numel,
                    dtype,
                )?;

                Ok(result)
            }
            _ => {
                // Sum, Max, Min - use existing shader
                let op_str = match op {
                    ScatterReduceOp::Sum => "sum",
                    ScatterReduceOp::Max => "max",
                    ScatterReduceOp::Min => "min",
                    _ => unreachable!(),
                };

                launch_scatter_reduce(
                    self.pipeline_cache(),
                    self.wgpu_queue(),
                    &src_buf,
                    &index_buf,
                    &output_buf,
                    &params_buf,
                    total_src,
                    dtype,
                    op_str,
                )?;

                Ok(output)
            }
        }
    }

    fn gather_nd(
        &self,
        input: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();

        // Check supported dtypes
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "gather_nd",
            });
        }

        // Validate indices dtype
        if !matches!(indices.dtype(), DType::I32 | DType::I64) {
            return Err(Error::InvalidArgument {
                arg: "indices",
                reason: "gather_nd indices must be I32 or I64".to_string(),
            });
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let indices_i32 = ensure_i32_indices(self, indices)?;
        let indices = ensure_contiguous(&indices_i32)?;

        let input_shape = input.shape();
        let indices_shape = indices.shape();

        // indices has shape [..., index_depth]
        // where index_depth <= input_ndim
        let index_depth = *indices_shape.last().unwrap_or(&0);
        let num_slices: usize = indices_shape[..indices_shape.len() - 1].iter().product();

        if index_depth > input_shape.len() {
            return Err(Error::InvalidArgument {
                arg: "indices",
                reason: format!(
                    "index depth {} exceeds input dimensions {}",
                    index_depth,
                    input_shape.len()
                ),
            });
        }

        // Compute output shape and slice size
        // Output shape = indices_shape[:-1] + input_shape[index_depth:]
        let slice_size: usize = input_shape[index_depth..].iter().product();
        let slice_size = if slice_size == 0 { 1 } else { slice_size };

        let mut output_shape: Vec<usize> = indices_shape[..indices_shape.len() - 1].to_vec();
        output_shape.extend_from_slice(&input_shape[index_depth..]);
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        let total_output = num_slices * slice_size;

        // Allocate output
        let output = alloc_output(self, &output_shape, dtype)?;

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let indices_buf = get_tensor_buffer(&indices)?;
        let output_buf = get_tensor_buffer(&output)?;

        // Compute strides
        let ndim = input_shape.len();
        let mut input_strides = [0u32; 8];
        let mut input_shape_arr = [0u32; 8];
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            if i < 8 {
                input_strides[i] = stride as u32;
                input_shape_arr[i] = input_shape[i] as u32;
            }
            stride *= input_shape[i];
        }

        // Create params
        let params = GatherNdParams {
            num_slices: num_slices as u32,
            slice_size: slice_size as u32,
            index_depth: index_depth as u32,
            ndim: ndim as u32,
            input_shape: input_shape_arr,
            input_strides,
        };
        let params_buf = create_params_buffer(self, &params);

        launch_gather_nd(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &indices_buf,
            &output_buf,
            &params_buf,
            total_output,
            dtype,
        )?;

        Ok(output)
    }

    fn bincount(
        &self,
        input: &Tensor<WgpuRuntime>,
        weights: Option<&Tensor<WgpuRuntime>>,
        minlength: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Validate input is 1D integer
        if input.ndim() != 1 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "bincount input must be 1D".to_string(),
            });
        }

        if !matches!(input.dtype(), DType::I32 | DType::I64) {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "bincount input must be integer type (I32 or I64)".to_string(),
            });
        }

        // Determine output dtype
        let output_dtype = if let Some(w) = weights {
            if !matches!(w.dtype(), DType::F32 | DType::I32 | DType::U32) {
                return Err(Error::UnsupportedDType {
                    dtype: w.dtype(),
                    op: "bincount weights",
                });
            }
            w.dtype()
        } else {
            DType::U32 // Unweighted bincount returns counts as U32
        };

        // Cast I64→I32 on GPU (WebGPU shaders use i32 indices)
        let input_i32 = ensure_i32_indices(self, input)?;
        let input = ensure_contiguous(&input_i32)?;
        let weights = weights.map(ensure_contiguous).transpose()?;

        let n = input.numel();

        // Determine output size: max reduction on GPU, read single scalar back.
        // This is a necessary system boundary (same as CPU/CUDA computing max first).
        let input_f32 = self.cast(&input, DType::F32)?;
        let max_tensor = self.max(&input_f32, &[0], true)?;
        let max_val = max_tensor.item::<f32>()? as i64;
        if max_val < 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "bincount requires non-negative values".to_string(),
            });
        }
        let output_len = ((max_val as usize) + 1).max(minlength);

        // Allocate zero-initialized output buffer.
        // Unweighted: U32 counts. Weighted: same dtype as weights (shader uses atomic<u32> bitcast).
        let output = if output_dtype == DType::U32 {
            let zeros = vec![0u32; output_len];
            Tensor::<WgpuRuntime>::from_slice(&zeros, &[output_len], self.device())
        } else {
            Tensor::zeros(&[output_len], output_dtype, self.device())
        };

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let output_buf = get_tensor_buffer(&output)?;

        let weights_buf = if let Some(ref w) = weights {
            Some(get_tensor_buffer(w)?)
        } else {
            None
        };

        // Create params
        let params = BincountParams {
            n: n as u32,
            minlength: output_len as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        launch_bincount(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            weights_buf.as_deref(),
            &output_buf,
            &params_buf,
            n,
            weights.as_ref().map(|w| w.dtype()),
        )?;

        // Cast U32 kernel output to I64 for parity with CPU backend (unweighted returns I64)
        if weights.is_none() {
            return self.cast(&output, DType::I64);
        }

        Ok(output)
    }

    fn gather_2d(
        &self,
        input: &Tensor<WgpuRuntime>,
        rows: &Tensor<WgpuRuntime>,
        cols: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();
        let shape = input.shape();

        // Check supported dtypes (WebGPU doesn't support f64)
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "gather_2d",
            });
        }

        // Validate input is 2D
        if shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: shape.to_vec(),
            });
        }

        let nrows = shape[0];
        let ncols = shape[1];

        // Validate index dtypes (WebGPU prefers I32)
        if !matches!(rows.dtype(), DType::I32 | DType::I64) {
            return Err(Error::InvalidArgument {
                arg: "rows",
                reason: "gather_2d rows must be I32 or I64".to_string(),
            });
        }

        if !matches!(cols.dtype(), DType::I32 | DType::I64) {
            return Err(Error::InvalidArgument {
                arg: "cols",
                reason: "gather_2d cols must be I32 or I64".to_string(),
            });
        }

        // Validate rows and cols are 1D and have same length
        if rows.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![rows.numel()],
                got: rows.shape().to_vec(),
            });
        }

        if cols.ndim() != 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![cols.numel()],
                got: cols.shape().to_vec(),
            });
        }

        let num_indices = rows.numel();
        if cols.numel() != num_indices {
            return Err(Error::ShapeMismatch {
                expected: vec![num_indices],
                got: cols.shape().to_vec(),
            });
        }

        // Make all inputs contiguous
        let input = ensure_contiguous(input)?;
        let rows_i32 = ensure_i32_indices(self, rows)?;
        let cols_i32 = ensure_i32_indices(self, cols)?;
        let rows = ensure_contiguous(&rows_i32)?;
        let cols = ensure_contiguous(&cols_i32)?;

        // Allocate output
        let output = alloc_output(self, &[num_indices], dtype)?;

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let rows_buf = get_tensor_buffer(&rows)?;
        let cols_buf = get_tensor_buffer(&cols)?;
        let output_buf = get_tensor_buffer(&output)?;

        // Create params
        let params = Gather2dParams {
            nrows: nrows as u32,
            ncols: ncols as u32,
            num_indices: num_indices as u32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        launch_gather_2d(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &rows_buf,
            &cols_buf,
            &output_buf,
            &params_buf,
            num_indices,
            dtype,
        )?;

        Ok(output)
    }

    fn slice_assign(
        &self,
        dst: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
        dim: usize,
        start: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_slice_assign(self, dst, src, dim, start)
    }
}
