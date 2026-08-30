//! Indexing operations trait.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Reduction operations for scatter_reduce.
///
/// When multiple source values scatter to the same destination index,
/// this enum determines how they are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduceOp {
    /// Sum all values that scatter to the same index
    Sum,
    /// Take the mean of all values (sum / count)
    Mean,
    /// Take the maximum value
    Max,
    /// Take the minimum value
    Min,
    /// Multiply all values together
    Prod,
}

/// Validate that indices tensor has an integer dtype (I32 or I64).
fn validate_index_dtype<R: Runtime<DType = DType>>(indices: &Tensor<R>) -> Result<()> {
    match indices.dtype() {
        DType::I32 | DType::I64 => Ok(()),
        other => Err(Error::InvalidArgument {
            arg: "indices",
            reason: format!(
                "indices must have integer dtype (I32 or I64), got {:?}",
                other
            ),
        }),
    }
}

/// Indexing operations
pub trait IndexingOps<R: Runtime> {
    /// Argmax: returns indices of maximum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the maximum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the maximum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of maximum values
    fn argmax(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>> {
        let _ = (a, dim, keepdim);
        Err(Error::NotImplemented {
            feature: "IndexingOps::argmax",
        })
    }

    /// Argmin: returns indices of minimum values along a dimension.
    ///
    /// Returns a tensor of I64 indices indicating the position of the minimum
    /// value along the specified dimension. The output shape is the input shape
    /// with the specified dimension removed (or kept as size 1 if keepdim=true).
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to find the minimum index
    /// * `keepdim` - If true, the reduced dimension is retained with size 1
    ///
    /// # Returns
    ///
    /// Tensor of I64 containing indices of minimum values
    fn argmin(&self, a: &Tensor<R>, dim: usize, keepdim: bool) -> Result<Tensor<R>> {
        let _ = (a, dim, keepdim);
        Err(Error::NotImplemented {
            feature: "IndexingOps::argmin",
        })
    }

    /// Gather elements along a dimension using an index tensor.
    ///
    /// For a 3D tensor with dim=1:
    /// `` `out[i][j][k] = input[i][index[i][j][k]][k]` ``
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to gather
    /// * `index` - Index tensor (I32 or I64) with same number of dimensions as input
    ///
    /// # Returns
    ///
    /// Tensor with same shape as index tensor, same dtype as input
    fn gather(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, dim, index);
        Err(Error::NotImplemented {
            feature: "IndexingOps::gather",
        })
    }

    /// Scatter values into a tensor at positions specified by an index tensor.
    ///
    /// Creates a new tensor (copy of `a`) with values from `src` scattered at positions
    /// specified by `index` along dimension `dim`.
    ///
    /// For a 3D tensor with dim=1:
    /// `` `out[i][index[i][j][k]][k] = src[i][j][k]` ``
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor (values to scatter into)
    /// * `dim` - Dimension along which to scatter
    /// * `index` - Index tensor (I32 or I64) specifying scatter positions
    /// * `src` - Source tensor with values to scatter
    ///
    /// # Returns
    ///
    /// New tensor with scattered values
    fn scatter(
        &self,
        a: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let _ = (a, dim, index, src);
        Err(Error::NotImplemented {
            feature: "IndexingOps::scatter",
        })
    }

    /// Select elements along a dimension using a 1D index tensor.
    ///
    /// Simpler than gather - the index tensor is 1D and applies to all positions
    /// in the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to select
    /// * `index` - 1D index tensor (I32 or I64) of length m
    ///
    /// # Returns
    ///
    /// Tensor with dimension `dim` having size m (length of index)
    ///
    /// # Bounds Checking
    ///
    /// Returns `IndexOutOfBounds` error if any index is negative or >= dim_size.
    /// Indices must be in the range `[0, dim_size)`. Negative indices are not supported.
    fn index_select(&self, a: &Tensor<R>, dim: usize, index: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, dim, index);
        Err(Error::NotImplemented {
            feature: "IndexingOps::index_select",
        })
    }

    /// Put values at specified indices along a dimension.
    ///
    /// This is the inverse of `index_select` - it assigns values from `src` into
    /// positions specified by `index` along dimension `dim`.
    ///
    /// # Example
    ///
    /// ```text
    /// # Replace row 2 of a [5, 3] matrix with new values:
    /// a = [[1, 2, 3],
    ///      [4, 5, 6],
    ///      [7, 8, 9],      # <- row 2 will be replaced
    ///      [10, 11, 12],
    ///      [13, 14, 15]]
    /// index = [2]          # indices along dim 0
    /// src = [[100, 200, 300]]
    ///
    /// result = index_put(a, 0, index, src)
    /// # result = [[1, 2, 3],
    /// #           [4, 5, 6],
    /// #           [100, 200, 300],  # replaced!
    /// #           [10, 11, 12],
    /// #           [13, 14, 15]]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor to modify (copied, not mutated)
    /// * `dim` - Dimension along which to put values
    /// * `index` - 1D index tensor (I32 or I64) specifying positions
    /// * `src` - Source tensor with values to insert. Shape must match `a` except
    ///   at `dim` where it must equal `index.numel()`
    ///
    /// # Returns
    ///
    /// New tensor with values at indexed positions replaced by `src`
    ///
    /// # Bounds Checking
    ///
    /// Returns `IndexOutOfBounds` error if any index is negative or >= dim_size.
    /// Indices must be in the range `[0, dim_size)`. Negative indices are not supported.
    fn index_put(
        &self,
        a: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let _ = (a, dim, index, src);
        Err(Error::NotImplemented {
            feature: "IndexingOps::index_put",
        })
    }

    /// Take values from a tensor using flat indices.
    ///
    /// The input tensor is treated as flattened 1D storage, and values are gathered
    /// at positions specified by `indices`. The output shape matches `indices.shape()`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to gather from
    /// * `indices` - Index tensor (I32 or I64) containing flat indices
    ///
    /// # Returns
    ///
    /// Tensor of shape `indices.shape()` with gathered values
    fn take(&self, tensor: &Tensor<R>, indices: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
    {
        validate_index_dtype(indices)?;
        let flat = tensor.contiguous()?.flatten()?;
        let indices_flat = indices.contiguous()?.flatten()?;
        let out_flat = self.index_select(&flat, 0, &indices_flat)?;
        out_flat.reshape(indices.shape())
    }

    /// Put values into a tensor at flat indices (functional, non-mutating).
    ///
    /// Returns a new tensor with `values` written at positions specified by `indices`,
    /// treating the input tensor as flattened 1D storage.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor to update
    /// * `indices` - Index tensor (I32 or I64) containing flat indices
    /// * `values` - Values to write. Must have the same number of elements as `indices`.
    ///
    /// # Returns
    ///
    /// New tensor with the same shape as `tensor` and updated values
    fn put(&self, tensor: &Tensor<R>, indices: &Tensor<R>, values: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
    {
        validate_index_dtype(indices)?;
        let flat = tensor.contiguous()?.flatten()?;
        let indices_flat = indices.contiguous()?.flatten()?;
        let values_flat = values.contiguous()?.flatten()?;

        if values_flat.numel() != indices_flat.numel() {
            return Err(Error::ShapeMismatch {
                expected: vec![indices_flat.numel()],
                got: vec![values_flat.numel()],
            });
        }

        let out_flat = self.index_put(&flat, 0, &indices_flat, &values_flat)?;
        out_flat.reshape(tensor.shape())
    }

    /// Select elements where mask is true, returning a flattened 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    ///
    /// # Returns
    ///
    /// 1D tensor containing only elements where mask is true
    fn masked_select(&self, a: &Tensor<R>, mask: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, mask);
        Err(Error::NotImplemented {
            feature: "IndexingOps::masked_select",
        })
    }

    /// Fill elements where mask is true with a scalar value.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `mask` - Boolean mask tensor (U8: 0=false, non-zero=true), must be broadcastable to `a`
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    ///
    /// New tensor with masked positions filled with value
    fn masked_fill(&self, a: &Tensor<R>, mask: &Tensor<R>, value: f64) -> Result<Tensor<R>> {
        let _ = (a, mask, value);
        Err(Error::NotImplemented {
            feature: "IndexingOps::masked_fill",
        })
    }

    /// Look up embeddings from an embedding table using indices.
    ///
    /// This is the standard embedding lookup operation used in neural networks
    /// for word embeddings, entity embeddings, etc. It is equivalent to
    /// `index_select(embeddings, 0, indices)` but optimized for the common case
    /// where the embedding table is 2D and indices index into the first dimension.
    ///
    /// # Algorithm
    ///
    /// For each index value i in the indices tensor:
    /// ```text
    /// output[..., i, :] = embeddings[indices[..., i], :]
    /// ```
    ///
    /// The output shape is `indices.shape() + [embedding_dim]` where `embedding_dim`
    /// is `embeddings.shape()[1]`.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - 2D embedding table of shape `` `[vocab_size, embedding_dim]` ``
    /// * `indices` - Index tensor of any shape containing indices into the embedding table.
    ///   Must be I32 or I64. Values must be in range `` `[0, vocab_size)` ``.
    ///
    /// # Returns
    ///
    /// Tensor of shape `` `indices.shape() + [embedding_dim]` `` containing the looked-up embeddings.
    ///
    /// # Example
    ///
    /// ```text
    /// embeddings = [[1.0, 2.0],   # word 0
    ///               [3.0, 4.0],   # word 1
    ///               [5.0, 6.0]]   # word 2
    /// indices = [2, 0, 1]
    ///
    /// output = [[5.0, 6.0],   # word 2
    ///           [1.0, 2.0],   # word 0
    ///           [3.0, 4.0]]   # word 1
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if embeddings is not 2D
    /// * `DTypeMismatch` - if indices is not I32 or I64
    /// * Index out of bounds results in undefined behavior (implementation may return zeros)
    ///
    /// # Performance
    ///
    /// On GPU, this operation is memory-bound and optimized for coalesced reads
    /// from the embedding table. Each thread handles one index lookup and writes
    /// a full embedding vector.
    fn embedding_lookup(&self, embeddings: &Tensor<R>, indices: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (embeddings, indices);
        Err(Error::NotImplemented {
            feature: "IndexingOps::embedding_lookup",
        })
    }

    /// Scatter values with reduction into a destination tensor.
    ///
    /// Unlike regular `scatter` which overwrites values, `scatter_reduce` applies
    /// a reduction operation when multiple source values scatter to the same
    /// destination index.
    ///
    /// # Algorithm
    ///
    /// For each position in `src`:
    /// ```text
    /// dst[..., index[...], ...] = reduce(dst[..., index[...], ...], src[...])
    /// ```
    ///
    /// Where `reduce` is determined by the `op` parameter.
    ///
    /// # Arguments
    ///
    /// * `dst` - Destination tensor to scatter into (used as initial values)
    /// * `dim` - Dimension along which to scatter
    /// * `index` - Index tensor (I32 or I64) specifying scatter positions
    /// * `src` - Source tensor with values to scatter
    /// * `op` - Reduction operation to apply (Sum, Mean, Max, Min, Prod)
    /// * `include_self` - If true, include `dst` values in reduction; if false, initialize
    ///   destination positions from `src` only
    ///
    /// # Returns
    ///
    /// New tensor with scattered and reduced values
    ///
    /// # Example
    ///
    /// ```text
    /// dst = [0, 0, 0, 0]
    /// index = [0, 0, 2]
    /// src = [1, 2, 3]
    /// scatter_reduce(dst, 0, index, src, Sum, include_self=true)
    /// # Result: [3, 0, 3, 0]  // src[0]+src[1]=3 at index 0, src[2]=3 at index 2
    /// ```
    fn scatter_reduce(
        &self,
        dst: &Tensor<R>,
        dim: usize,
        index: &Tensor<R>,
        src: &Tensor<R>,
        op: ScatterReduceOp,
        include_self: bool,
    ) -> Result<Tensor<R>> {
        let _ = (dst, dim, index, src, op, include_self);
        Err(Error::NotImplemented {
            feature: "IndexingOps::scatter_reduce",
        })
    }

    /// Gather elements using N-dimensional indices.
    ///
    /// Unlike regular `gather` which gathers along a single dimension,
    /// `gather_nd` uses an index tensor where the last dimension specifies
    /// coordinates into the input tensor.
    ///
    /// # Algorithm
    ///
    /// If `indices` has shape `` `[..., M]` `` and `input` has `N` dimensions:
    /// - If M == N: output has shape `` `indices.shape()[:-1]` ``
    /// - If M < N: output has shape `` `indices.shape()[:-1] + input.shape()[M:]` ``
    ///
    /// Each index vector `` `indices[..., :]` `` specifies coordinates for the first
    /// M dimensions of `input`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor to gather from
    /// * `indices` - Index tensor where last dimension contains coordinates
    ///
    /// # Returns
    ///
    /// Tensor with gathered values
    ///
    /// # Example
    ///
    /// ```text
    /// input = [[0, 1], [2, 3]]  # shape [2, 2]
    /// indices = [[0, 0], [1, 1]]  # shape [2, 2], last dim=2 means full coordinates
    /// gather_nd(input, indices)
    /// # Result: [0, 3]  # input[0,0]=0, input[1,1]=3
    ///
    /// indices = [[0], [1]]  # shape [2, 1], last dim=1 means gather rows
    /// gather_nd(input, indices)
    /// # Result: [[0, 1], [2, 3]]  # input[0,:], input[1,:]
    /// ```
    fn gather_nd(&self, input: &Tensor<R>, indices: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (input, indices);
        Err(Error::NotImplemented {
            feature: "IndexingOps::gather_nd",
        })
    }

    /// Count occurrences of each value in an integer tensor.
    ///
    /// Returns a histogram where `` `output[i]` `` contains the count (or weighted sum)
    /// of how many times value `i` appears in the input.
    ///
    /// # Arguments
    ///
    /// * `input` - 1D integer tensor with non-negative values (I32 or I64)
    /// * `weights` - Optional weights tensor, same shape as input. If provided,
    ///   the output is the sum of weights for each bin instead of counts.
    /// * `minlength` - Minimum length of the output tensor. Useful when the
    ///   maximum value is known ahead of time.
    ///
    /// # Returns
    ///
    /// 1D tensor of length `` `max(max(input)+1, minlength)` `` containing counts
    /// or weighted sums. An EMPTY input has no maximum, so the result is
    /// `minlength` zeroed bins — NumPy's answer, and not an error: an empty
    /// input holds no value, so it holds no negative one.
    ///
    /// # Example
    ///
    /// ```text
    /// input = [0, 1, 1, 3, 2, 1, 3]
    /// bincount(input, None, 0)
    /// # Result: [1, 3, 1, 2]  // counts: 0->1, 1->3, 2->1, 3->2
    ///
    /// weights = [0.5, 1.0, 1.5, 2.0, 1.0, 0.5, 3.0]
    /// bincount(input, Some(weights), 0)
    /// # Result: [0.5, 3.0, 1.0, 5.0]  // weighted sums per bin
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if input is not 1D or weights shape doesn't match input
    /// * `DTypeMismatch` - if input is not an integer type
    /// * `InvalidArgument` - if input contains a negative value, on every
    ///   backend. A negative index has no bin, so it is rejected rather than
    ///   dropped, matching NumPy's `ValueError`.
    fn bincount(
        &self,
        input: &Tensor<R>,
        weights: Option<&Tensor<R>>,
        minlength: usize,
    ) -> Result<Tensor<R>> {
        let _ = (input, weights, minlength);
        Err(Error::NotImplemented {
            feature: "IndexingOps::bincount",
        })
    }

    /// Count occurrences of each value into a histogram of exactly `len` bins.
    ///
    /// Same operation as [`IndexingOps::bincount`], except the caller supplies
    /// the output length instead of it being derived from `max(input) + 1`.
    /// Callers that already know the bin count — a vocabulary size, a class
    /// count, a bucket count — should prefer this method: on a GPU backend the
    /// derived length lives on the device, so `bincount` must read one scalar
    /// back to the host and synchronise before it can allocate. This method
    /// skips both the max reduction and the readback.
    ///
    /// # Out-of-range contract
    ///
    /// Values outside `` `[0, len)` `` — negative or too large — are **ignored**,
    /// not rejected. Detecting them would require the very device
    /// synchronisation this method exists to avoid. A caller that needs
    /// out-of-range values reported must use [`IndexingOps::bincount`].
    ///
    /// Note this differs from [`IndexingOps::bincount`], which rejects a
    /// negative input with `InvalidArgument`.
    ///
    /// # Arguments
    ///
    /// * `input` - 1D integer tensor (I32 or I64)
    /// * `weights` - Optional weights tensor, same shape as input. If provided,
    ///   the output is the sum of weights for each bin instead of counts.
    /// * `len` - Exact length of the output tensor. `0` yields an empty tensor.
    ///
    /// # Returns
    ///
    /// 1D tensor of length `len`, with the same dtype [`IndexingOps::bincount`]
    /// produces for the same inputs: the weights dtype when weighted, I64 counts
    /// otherwise.
    ///
    /// # Example
    ///
    /// ```text
    /// input = [0, 1, 1, 3, 2, 1, 3]
    /// bincount_with_len(input, None, 6)
    /// # Result: [1, 3, 1, 2, 0, 0]  // trailing bins stay zero
    ///
    /// bincount_with_len(input, None, 2)
    /// # Result: [1, 3]  // values 2 and 3 fall outside [0, 2) and are ignored
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if input is not 1D or weights shape doesn't match input
    /// * `DTypeMismatch` - if input is not an integer type
    fn bincount_with_len(
        &self,
        input: &Tensor<R>,
        weights: Option<&Tensor<R>>,
        len: usize,
    ) -> Result<Tensor<R>> {
        let _ = (input, weights, len);
        Err(Error::NotImplemented {
            feature: "IndexingOps::bincount_with_len",
        })
    }

    /// Gather elements from a 2D matrix using row and column index vectors.
    ///
    /// For each index i, extracts `` `input[rows[i], cols[i]]` ``.
    ///
    /// This is a specialized gather operation optimized for sparse matrix
    /// applications where you need to extract values at specific (row, col)
    /// coordinates.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// output[i] = input[rows[i], cols[i]]  for i in 0..len(rows)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - 2D input tensor of shape `` `[nrows, ncols]` ``
    /// * `rows` - 1D index tensor (I32 or I64) specifying row indices
    /// * `cols` - 1D index tensor (I32 or I64) specifying column indices
    ///
    /// # Returns
    ///
    /// 1D tensor of length `` `rows.numel()` `` with gathered values.
    /// Same dtype as input.
    ///
    /// # Example
    ///
    /// ```text
    /// input = [[1, 2, 3],
    ///          [4, 5, 6],
    ///          [7, 8, 9]]
    /// rows = [0, 1, 2, 0]
    /// cols = [0, 1, 2, 2]
    ///
    /// gather_2d(input, rows, cols)
    /// # Result: [1, 5, 9, 3]  // input[0,0], input[1,1], input[2,2], input[0,2]
    /// ```
    ///
    /// # Errors
    ///
    /// * `ShapeMismatch` - if input is not 2D or rows/cols have different lengths
    /// * `DTypeMismatch` - if rows or cols are not I32 or I64
    /// * `IndexOutOfBounds` - if any (row, col) pair is out of bounds
    fn gather_2d(
        &self,
        input: &Tensor<R>,
        rows: &Tensor<R>,
        cols: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let _ = (input, rows, cols);
        Err(Error::NotImplemented {
            feature: "IndexingOps::gather_2d",
        })
    }

    /// Assign `src` into a slice of `dst` along dimension `dim` starting at `start`.
    ///
    /// Returns a new tensor equal to `dst` except that the region
    /// `dst[..., start..start+src.shape[dim], ...]` is replaced by `src`.
    ///
    /// # Arguments
    ///
    /// * `dst` - Destination tensor
    /// * `src` - Source tensor. Must have same shape as `dst` except at `dim`,
    ///   where `src.shape[dim] + start <= dst.shape[dim]`
    /// * `dim` - Dimension along which to assign
    /// * `start` - Starting index in `dst` along `dim`
    ///
    /// # Returns
    ///
    /// New tensor with the slice replaced
    fn slice_assign(
        &self,
        dst: &Tensor<R>,
        src: &Tensor<R>,
        dim: usize,
        start: usize,
    ) -> Result<Tensor<R>> {
        let _ = (dst, src, dim, start);
        Err(Error::NotImplemented {
            feature: "IndexingOps::slice_assign",
        })
    }
}
