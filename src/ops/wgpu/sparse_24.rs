//! WebGPU implementation of 2:4 structured sparsity operations.
//!
//! WebGPU uses decompress + standard matmul (no hardware sparse tensor cores).
//! F32 only (WebGPU constraint).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::MatmulOps;
use crate::ops::traits::Sparse24Ops;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{alloc_output, create_params_buffer, get_tensor_buffer};
use crate::runtime::wgpu::shaders::sparse_24::{
    Sparse24Params, launch_sparse_24_decompress, launch_sparse_24_prune,
};
use crate::sparse::structured::{Sparse24Tensor, meta_cols_for_k};
use crate::tensor::Tensor;

impl Sparse24Ops<WgpuRuntime> for WgpuClient {
    fn prune_to_24(&self, dense: &Tensor<WgpuRuntime>) -> Result<Sparse24Tensor<WgpuRuntime>> {
        if dense.ndim() != 2 {
            return Err(Error::InvalidArgument {
                arg: "dense",
                reason: format!("Expected 2D tensor, got {}D", dense.ndim()),
            });
        }

        let dtype = dense.dtype();
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_24_prune (WebGPU: F32 only)",
            });
        }

        let m = dense.shape()[0];
        let k = dense.shape()[1];

        if !k.is_multiple_of(4) {
            return Err(Error::InvalidArgument {
                arg: "dense",
                reason: format!("K dimension ({k}) must be divisible by 4 for 2:4 sparsity"),
            });
        }

        let dense_contig = ensure_contiguous(dense)?;
        let half_k = k / 2;
        let mc = meta_cols_for_k(k);
        let num_groups = k / 4;
        let total_groups = m * num_groups;

        let compressed = alloc_output(self, &[m, half_k], dtype)?;
        let metadata = alloc_output(self, &[m, mc], DType::U32)?;

        // wgpu buffers are zero-initialized by default (spec requirement)

        let dense_buf = get_tensor_buffer(&dense_contig)?;
        let comp_buf = get_tensor_buffer(&compressed)?;
        let meta_buf = get_tensor_buffer(&metadata)?;

        let params = Sparse24Params {
            total_groups: total_groups as u32,
            num_groups_per_row: num_groups as u32,
            meta_cols: mc as u32,
            half_k: half_k as u32,
            k: k as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        launch_sparse_24_prune(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &dense_buf,
            &comp_buf,
            &meta_buf,
            &params_buf,
            total_groups,
        )?;

        Sparse24Tensor::new(compressed, metadata, [m, k])
    }

    fn sparse_24_to_dense(
        &self,
        sparse: &Sparse24Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let [m, k] = sparse.shape();
        let dtype = sparse.dtype();

        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_24_to_dense (WebGPU: F32 only)",
            });
        }

        let num_groups = k / 4;
        let total_groups = m * num_groups;
        let mc = meta_cols_for_k(k);
        let half_k = k / 2;

        let vals = ensure_contiguous(sparse.compressed_values())?;
        let meta = ensure_contiguous(sparse.metadata())?;
        let dense = alloc_output(self, &[m, k], dtype)?;

        let vals_buf = get_tensor_buffer(&vals)?;
        let meta_buf = get_tensor_buffer(&meta)?;
        let dense_buf = get_tensor_buffer(&dense)?;

        let params = Sparse24Params {
            total_groups: total_groups as u32,
            num_groups_per_row: num_groups as u32,
            meta_cols: mc as u32,
            half_k: half_k as u32,
            k: k as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        launch_sparse_24_decompress(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &vals_buf,
            &meta_buf,
            &dense_buf,
            &params_buf,
            total_groups,
        )?;

        Ok(dense)
    }

    fn sparse_24_matmul(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Sparse24Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU: decompress weight to dense, then standard matmul
        let dense_weight = self.sparse_24_to_dense(weight)?;
        let weight_t = dense_weight.t()?;
        self.matmul(input, &weight_t)
    }
}
