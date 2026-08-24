//! Sorting and search operations trait.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Sorting and search operations trait
///
/// Provides operations for sorting tensors, finding top-k elements, searching,
/// and computing unique values.
///
/// # Ordering
///
/// All backends sort by the total order in
/// [`Element::sort_cmp`](crate::dtype::Element::sort_cmp): NaN is greater than
/// every non-NaN value (last ascending, first descending, and top-ranked for
/// `topk` with `largest`), NaNs tie with each other, `-0.0` ties with `+0.0`,
/// and ties keep input order in both directions. `searchsorted` expects its
/// sequence sorted by this same order.
///
/// # Index dtype
///
/// Index outputs are `I64` on CPU and CUDA but `I32` on WebGPU, a 32-bit-only
/// backend. Read them back in the returned tensor's own dtype.
pub trait SortingOps<R: Runtime> {
    /// Sort tensor along a dimension.
    ///
    /// Returns the sorted values. For both values and indices, use `sort_with_indices`.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to sort (supports negative indexing)
    /// * `descending` - If true, sort in descending order
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, containing sorted values along the specified dimension.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device)?;
    /// let sorted = client.sort(&a, 0, false)?; // [1.0, 1.0, 3.0, 4.0, 5.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn sort(&self, a: &Tensor<R>, dim: isize, descending: bool) -> Result<Tensor<R>> {
        let _ = (a, dim, descending);
        Err(Error::NotImplemented {
            feature: "SortingOps::sort",
        })
    }

    /// Sort tensor along a dimension, returning both sorted values and indices.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to sort (supports negative indexing)
    /// * `descending` - If true, sort in descending order
    ///
    /// # Returns
    ///
    /// Tuple of (sorted_values, indices) where:
    /// - `sorted_values`: Tensor with same shape and dtype as input, containing sorted values
    /// - `indices`: I64 tensor with same shape, containing original indices
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0], &[3], &device)?;
    /// let (values, indices) = client.sort_with_indices(&a, 0, false)?;
    /// // values = [1.0, 3.0, 4.0]
    /// // indices = [1, 0, 2]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn sort_with_indices(
        &self,
        a: &Tensor<R>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (a, dim, descending);
        Err(Error::NotImplemented {
            feature: "SortingOps::sort_with_indices",
        })
    }

    /// Return indices that would sort the tensor along a dimension.
    ///
    /// Equivalent to `sort_with_indices(...).1`, but more efficient when only
    /// indices are needed.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dim` - Dimension along which to compute sort indices (supports negative indexing)
    /// * `descending` - If true, return indices for descending order
    ///
    /// # Returns
    ///
    /// I64 tensor with same shape as input, containing indices that would sort the tensor.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0], &[3], &device)?;
    /// let indices = client.argsort(&a, 0, false)?; // [1, 0, 2]
    /// // a[indices] would give [1.0, 3.0, 4.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn argsort(&self, a: &Tensor<R>, dim: isize, descending: bool) -> Result<Tensor<R>> {
        let _ = (a, dim, descending);
        Err(Error::NotImplemented {
            feature: "SortingOps::argsort",
        })
    }

    /// Return top K largest (or smallest) values and their indices along a dimension.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `k` - Number of top elements to return
    /// * `dim` - Dimension along which to find top-k (supports negative indexing)
    /// * `largest` - If true, return largest elements; if false, return smallest
    /// * `sorted` - If true, return in sorted order; if false, maintain relative input order
    ///
    /// # Returns
    ///
    /// Tuple of (values, indices) where:
    /// - `values`: Tensor with shape `` `[..., k, ...]` `` (dim replaced with k), same dtype as input
    /// - `indices`: I64 tensor with same shape as values, containing original indices
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if k > dim_size.
    ///
    /// # Backend Limitations
    ///
    /// - **WebGPU**: Max 512 elements per sort dimension (bitonic sort uses shared memory)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5], &device)?;
    /// let (values, indices) = client.topk(&a, 2, 0, true, true)?;
    /// // values = [5.0, 4.0] (largest 2, sorted)
    /// // indices = [4, 2]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn topk(
        &self,
        a: &Tensor<R>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (a, k, dim, largest, sorted);
        Err(Error::NotImplemented {
            feature: "SortingOps::topk",
        })
    }

    /// Return unique elements of the input tensor.
    ///
    /// Flattens the tensor and returns unique elements in sorted order.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `sorted` - If true (default), return unique elements in sorted order
    ///
    /// # Returns
    ///
    /// 1D tensor containing unique elements. Length may vary depending on input.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0, 1.0], &[5], &device)?;
    /// let unique = client.unique(&a, true)?; // [1.0, 2.0, 3.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn unique(&self, a: &Tensor<R>, sorted: bool) -> Result<Tensor<R>> {
        let _ = (a, sorted);
        Err(Error::NotImplemented {
            feature: "SortingOps::unique",
        })
    }

    /// Return unique elements with inverse indices and counts.
    ///
    /// More complete version of `unique` that also returns:
    /// - Inverse indices: for each element in input, index into unique output
    /// - Counts: how many times each unique element appears
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// Tuple of (unique, inverse_indices, counts) where:
    /// - `unique`: 1D tensor of unique values (sorted)
    /// - `inverse_indices`: I64 tensor with same shape as flattened input
    /// - `counts`: I64 tensor with same length as unique, containing occurrence counts
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 3.0, 1.0], &[5], &device)?;
    /// let (unique, inverse, counts) = client.unique_with_counts(&a)?;
    /// // unique = [1.0, 2.0, 3.0]
    /// // inverse = [0, 1, 1, 2, 0] (maps each input to index in unique)
    /// // counts = [2, 2, 1] (1.0 appears 2x, 2.0 appears 2x, 3.0 appears 1x)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn unique_with_counts(&self, a: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "SortingOps::unique_with_counts",
        })
    }

    /// Return indices of non-zero elements.
    ///
    /// Returns a 2D tensor where each row contains the indices of a non-zero element.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    ///
    /// # Returns
    ///
    /// I64 tensor of shape `` `[N, ndim]` `` where N is the number of non-zero elements
    /// and ndim is the number of dimensions in the input tensor. Each row contains
    /// the multi-dimensional index of a non-zero element.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, 0.0, 2.0], &[2, 2], &device)?;
    /// let indices = client.nonzero(&a)?;
    /// // indices = [[0, 1], [1, 1]] (positions of 1.0 and 2.0)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn nonzero(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "SortingOps::nonzero",
        })
    }

    /// Find insertion points for values in a sorted sequence.
    ///
    /// For each value in `values`, finds the index in `sorted_sequence` where the value
    /// should be inserted to maintain sorted order.
    ///
    /// # Arguments
    ///
    /// * `sorted_sequence` - 1D sorted tensor
    /// * `values` - Values to search for (any shape)
    /// * `right` - If true, find rightmost insertion point; if false, leftmost
    ///
    /// # Returns
    ///
    /// I64 tensor with same shape as `values`, containing insertion indices.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::SortingOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let sorted = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 5.0, 7.0], &[4], &device)?;
    /// let values = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0], &[3], &device)?;
    /// let indices = client.searchsorted(&sorted, &values, false)?;
    /// // indices = [1, 2, 3] (insert positions to maintain order)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<R>,
        values: &Tensor<R>,
        right: bool,
    ) -> Result<Tensor<R>> {
        let _ = (sorted_sequence, values, right);
        Err(Error::NotImplemented {
            feature: "SortingOps::searchsorted",
        })
    }
}
