//! Indexing operations for CUDA runtime

mod advanced;
mod argmax;
mod bincount;
mod gather_scatter;
mod helpers;
mod masked;

use crate::error::Result;
use crate::ops::{IndexingOps, ScatterReduceOp};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl IndexingOps<CudaRuntime> for CudaClient {
    fn argmax(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        argmax::argmax(self, a, dim, keepdim)
    }

    fn argmin(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        argmax::argmin(self, a, dim, keepdim)
    }

    fn gather(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::gather(self, a, dim, index)
    }

    fn scatter(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::scatter(self, a, dim, index, src)
    }

    fn index_select(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::index_select(self, a, dim, index)
    }

    fn index_put(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::index_put(self, a, dim, index, src)
    }

    fn masked_select(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        masked::masked_select(self, a, mask)
    }

    fn masked_fill(
        &self,
        a: &Tensor<CudaRuntime>,
        mask: &Tensor<CudaRuntime>,
        value: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        masked::masked_fill(self, a, mask, value)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<CudaRuntime>,
        indices: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        advanced::embedding_lookup(self, embeddings, indices)
    }

    fn scatter_reduce(
        &self,
        dst: &Tensor<CudaRuntime>,
        dim: usize,
        index: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
        op: ScatterReduceOp,
        include_self: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        advanced::scatter_reduce(self, dst, dim, index, src, op, include_self)
    }

    fn gather_nd(
        &self,
        input: &Tensor<CudaRuntime>,
        indices: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        advanced::gather_nd(self, input, indices)
    }

    fn bincount(
        &self,
        input: &Tensor<CudaRuntime>,
        weights: Option<&Tensor<CudaRuntime>>,
        minlength: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        bincount::bincount(self, input, weights, minlength)
    }

    fn bincount_with_len(
        &self,
        input: &Tensor<CudaRuntime>,
        weights: Option<&Tensor<CudaRuntime>>,
        len: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        bincount::bincount_with_len(self, input, weights, len)
    }

    fn gather_2d(
        &self,
        input: &Tensor<CudaRuntime>,
        rows: &Tensor<CudaRuntime>,
        cols: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::gather_2d(self, input, rows, cols)
    }

    fn slice_assign(
        &self,
        dst: &Tensor<CudaRuntime>,
        src: &Tensor<CudaRuntime>,
        dim: usize,
        start: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        gather_scatter::slice_assign(self, dst, src, dim, start)
    }
}
