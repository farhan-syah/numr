//! IndexingOps trait implementation for the WebGPU client.
//!
//! Every method here is a dispatch line: the small ops forward to the shared
//! native launchers, and the four that carry real logic live in the sibling
//! modules named after them.

use crate::error::Result;
use crate::ops::{IndexingOps, ScatterReduceOp};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_argreduce_op, native_embedding_lookup, native_gather, native_index_put,
    native_index_select, native_masked_fill, native_masked_select, native_scatter,
    native_slice_assign,
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
        super::scatter_reduce::scatter_reduce(self, dst, dim, index, src, op, include_self)
    }

    fn gather_nd(
        &self,
        input: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::gather_nd::gather_nd(self, input, indices)
    }

    fn bincount(
        &self,
        input: &Tensor<WgpuRuntime>,
        weights: Option<&Tensor<WgpuRuntime>>,
        minlength: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::bincount::bincount(self, input, weights, minlength)
    }

    fn gather_2d(
        &self,
        input: &Tensor<WgpuRuntime>,
        rows: &Tensor<WgpuRuntime>,
        cols: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::gather_2d::gather_2d(self, input, rows, cols)
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
