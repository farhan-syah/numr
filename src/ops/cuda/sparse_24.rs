//! CUDA implementation of 2:4 structured sparsity operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::Sparse24Ops;
use crate::runtime::cuda::kernels::{
    launch_sparse_24_decompress, launch_sparse_24_matmul, launch_sparse_24_prune,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::sparse::structured::{Sparse24Tensor, meta_cols_for_k};
use crate::tensor::Tensor;

impl Sparse24Ops<CudaRuntime> for CudaClient {
    fn prune_to_24(&self, dense: &Tensor<CudaRuntime>) -> Result<Sparse24Tensor<CudaRuntime>> {
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
        let dense_contig = ensure_contiguous(dense)?;
        let half_k = k / 2;
        let mc = meta_cols_for_k(k);

        let compressed = Tensor::<CudaRuntime>::empty(&[m, half_k], dtype, &self.device)?;
        // Metadata must be zeroed before kernel's atomic OR operations
        let metadata = Tensor::<CudaRuntime>::zeros(&[m, mc], DType::U32, &self.device)?;

        unsafe {
            launch_sparse_24_prune(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                dense_contig.ptr(),
                compressed.ptr(),
                metadata.ptr(),
                m,
                k,
            )?;
        }

        Sparse24Tensor::new(compressed, metadata, [m, k])
    }

    fn sparse_24_to_dense(
        &self,
        sparse: &Sparse24Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let [m, k] = sparse.shape();
        let dtype = sparse.dtype();

        let dense = Tensor::<CudaRuntime>::empty(&[m, k], dtype, &self.device)?;

        let vals = ensure_contiguous(sparse.compressed_values())?;
        let meta = ensure_contiguous(sparse.metadata())?;

        unsafe {
            launch_sparse_24_decompress(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                vals.ptr(),
                meta.ptr(),
                dense.ptr(),
                m,
                k,
            )?;
        }

        Ok(dense)
    }

    fn sparse_24_matmul(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Sparse24Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        if input.ndim() != 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!("Expected 2D tensor, got {}D", input.ndim()),
            });
        }

        let n = input.shape()[0];
        let input_k = input.shape()[1];
        let [m, weight_k] = weight.shape();

        if input_k != weight_k {
            return Err(Error::ShapeMismatch {
                expected: vec![n, weight_k],
                got: vec![n, input_k],
            });
        }

        let dtype = input.dtype();
        let input_contig = ensure_contiguous(input)?;
        let vals = ensure_contiguous(weight.compressed_values())?;
        let meta = ensure_contiguous(weight.metadata())?;

        let output = Tensor::<CudaRuntime>::empty(&[n, m], dtype, &self.device)?;

        unsafe {
            launch_sparse_24_matmul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                vals.ptr(),
                meta.ptr(),
                output.ptr(),
                n,
                m,
                weight_k,
            )?;
        }

        Ok(output)
    }
}
