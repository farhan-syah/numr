//! CPU implementation of 2:4 structured sparsity operations.

use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::MatmulOps;
use crate::ops::traits::Sparse24Ops;
use crate::runtime::cpu::kernels::sparse_24;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::ensure_contiguous;
use crate::sparse::structured::{Sparse24Tensor, meta_cols_for_k};
use crate::tensor::Tensor;

impl Sparse24Ops<CpuRuntime> for CpuClient {
    fn prune_to_24(&self, dense: &Tensor<CpuRuntime>) -> Result<Sparse24Tensor<CpuRuntime>> {
        if dense.ndim() != 2 {
            return Err(Error::InvalidArgument {
                arg: "dense",
                reason: format!("Expected 2D tensor, got {}D", dense.ndim()),
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

        let dtype = dense.dtype();
        let device = dense.device().clone();
        let dense_contig = ensure_contiguous(dense)?;

        let half_k = k / 2;
        let mc = meta_cols_for_k(k);

        let compressed = Tensor::<CpuRuntime>::empty(&[m, half_k], dtype, &device)?;
        let metadata = Tensor::<CpuRuntime>::empty(&[m, mc], DType::U32, &device)?;

        dispatch_dtype!(dtype, T => {
            unsafe {
                sparse_24::prune_to_24_kernel::<T>(
                    dense_contig.ptr() as *const T,
                    compressed.ptr() as *mut T,
                    metadata.ptr() as *mut u32,
                    m,
                    k,
                );
            }
        }, "prune_to_24");

        Sparse24Tensor::new(compressed, metadata, [m, k])
    }

    fn sparse_24_to_dense(
        &self,
        sparse: &Sparse24Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let [m, k] = sparse.shape();
        let dtype = sparse.dtype();
        let device = sparse.compressed_values().device().clone();

        let dense = Tensor::<CpuRuntime>::empty(&[m, k], dtype, &device)?;

        let vals = ensure_contiguous(sparse.compressed_values())?;
        let meta = ensure_contiguous(sparse.metadata())?;

        dispatch_dtype!(dtype, T => {
            unsafe {
                sparse_24::decompress_24_kernel::<T>(
                    vals.ptr() as *const T,
                    meta.ptr() as *const u32,
                    dense.ptr() as *mut T,
                    m,
                    k,
                );
            }
        }, "sparse_24_to_dense");

        Ok(dense)
    }

    fn sparse_24_matmul(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Sparse24Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // CPU fallback: decompress weight to dense, then standard matmul
        // input: [N, K], weight: [M, K] → output: [N, M]
        // matmul(input, weight^T) = matmul(input [N,K], dense_weight^T [K,M]) → [N, M]
        let dense_weight = self.sparse_24_to_dense(weight)?;
        let weight_t = dense_weight.t()?;
        self.matmul(input, &weight_t)
    }
}
