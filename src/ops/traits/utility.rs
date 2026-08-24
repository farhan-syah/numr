//! Utility operations trait.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Indexing mode for meshgrid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshgridIndexing {
    /// Matrix indexing (default): first dimension corresponds to first input
    Ij,
    /// Cartesian indexing: first two inputs are swapped (x=columns, y=rows)
    Xy,
}

/// Utility operations
pub trait UtilityOps<R: Runtime> {
    /// Clamp tensor values to a range: clamp(x, min, max) = min(max(x, min), max)
    ///
    /// Element-wise clamps each value to be within [min_val, max_val].
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `min_val` - Minimum value (inclusive)
    /// * `max_val` - Maximum value (inclusive)
    ///
    /// # Returns
    ///
    /// Tensor with same shape and dtype as input, with values clamped to range
    fn clamp(&self, a: &Tensor<R>, min_val: f64, max_val: f64) -> Result<Tensor<R>> {
        let _ = (a, min_val, max_val);
        Err(Error::NotImplemented {
            feature: "UtilityOps::clamp",
        })
    }

    /// Fill tensor with a constant value
    ///
    /// Creates a new tensor with the specified shape and dtype, filled with the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `value` - Value to fill the tensor with
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// New tensor filled with the constant value
    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<R>> {
        let _ = (shape, value, dtype);
        Err(Error::NotImplemented {
            feature: "UtilityOps::fill",
        })
    }

    /// Create a 1D tensor with evenly spaced values within a half-open interval [start, stop)
    ///
    /// Values are generated using the formula: start + step * i for i in 0..n
    /// where n = ceil((stop - start) / step)
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval (inclusive)
    /// * `stop` - End of the interval (exclusive)
    /// * `step` - Spacing between values (must be positive if start < stop, negative if start > stop)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::UtilityOps;
    ///
    /// let t = client.arange(0.0, 5.0, 1.0, DType::F32)?; // [0, 1, 2, 3, 4]
    /// let t = client.arange(0.0, 5.0, 2.0, DType::F32)?; // [0, 2, 4]
    /// let t = client.arange(5.0, 0.0, -1.0, DType::F32)?; // [5, 4, 3, 2, 1]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn arange(&self, start: f64, stop: f64, step: f64, dtype: DType) -> Result<Tensor<R>> {
        let _ = (start, stop, step, dtype);
        Err(Error::NotImplemented {
            feature: "UtilityOps::arange",
        })
    }

    /// Create a 1D tensor with evenly spaced values over a specified interval
    ///
    /// Returns `steps` evenly spaced values from `start` to `stop` (inclusive).
    /// Values are: start + (stop - start) * i / (steps - 1) for i in 0..steps
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the interval
    /// * `stop` - End of the interval (inclusive)
    /// * `steps` - Number of values to generate (must be >= 2)
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// 1D tensor with evenly spaced values
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::UtilityOps;
    ///
    /// let t = client.linspace(0.0, 10.0, 5, DType::F32)?; // [0, 2.5, 5, 7.5, 10]
    /// let t = client.linspace(0.0, 1.0, 3, DType::F64)?; // [0, 0.5, 1]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn linspace(&self, start: f64, stop: f64, steps: usize, dtype: DType) -> Result<Tensor<R>> {
        let _ = (start, stop, steps, dtype);
        Err(Error::NotImplemented {
            feature: "UtilityOps::linspace",
        })
    }

    /// Create a 2D identity matrix (or batch of identity matrices)
    ///
    /// Creates a tensor where the diagonal elements are 1 and all others are 0.
    /// For rectangular matrices, the diagonal is the main diagonal.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows
    /// * `m` - Number of columns (if None, defaults to n for square matrix)
    /// * `dtype` - Data type of the output tensor
    ///
    /// # Returns
    ///
    /// 2D tensor of shape [n, m] with ones on the diagonal
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::UtilityOps;
    ///
    /// let eye = client.eye(3, None, DType::F32)?;    // 3x3 identity matrix
    /// let rect = client.eye(2, Some(4), DType::F32)?; // 2x4 matrix with diagonal ones
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<R>> {
        let _ = (n, m, dtype);
        Err(Error::NotImplemented {
            feature: "UtilityOps::eye",
        })
    }

    /// One-hot encode integer indices
    ///
    /// Creates a tensor where each index value is expanded into a one-hot vector.
    /// The output has one additional dimension of size `num_classes` appended.
    ///
    /// # Arguments
    ///
    /// * `indices` - Integer tensor of any shape [...]. Values must be in [0, num_classes).
    /// * `num_classes` - Number of classes (size of the one-hot dimension)
    ///
    /// # Returns
    ///
    /// F32 tensor of shape [..., num_classes] where output[..., k] = 1.0
    /// if indices[...] == k, else 0.0.
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if indices is not an integer type
    /// - `InvalidArgument` if num_classes == 0
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::UtilityOps;
    ///
    /// let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1], &[3], &device)?;
    /// let oh = client.one_hot(&indices, 4)?;
    /// // oh = [[1, 0, 0, 0],
    /// //       [0, 0, 1, 0],
    /// //       [0, 1, 0, 0]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn one_hot(&self, indices: &Tensor<R>, num_classes: usize) -> Result<Tensor<R>> {
        let _ = (indices, num_classes);
        Err(Error::NotImplemented {
            feature: "UtilityOps::one_hot",
        })
    }

    /// Create coordinate grids from 1-D coordinate vectors
    ///
    /// Given N 1-D tensors, returns N N-D tensors where each output tensor
    /// represents one coordinate along one axis of the N-D grid.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of 1-D input tensors (the coordinate vectors)
    /// * `indexing` - Grid indexing convention (Ij for matrix, Xy for Cartesian)
    ///
    /// # Returns
    ///
    /// Vec of N tensors, each with shape [len(t0), len(t1), ..., len(tN-1)]
    /// (or with first two dims swapped for Xy indexing)
    fn meshgrid(
        &self,
        tensors: &[&Tensor<R>],
        indexing: MeshgridIndexing,
    ) -> Result<Vec<Tensor<R>>> {
        let _ = (tensors, indexing);
        Err(Error::NotImplemented {
            feature: "UtilityOps::meshgrid",
        })
    }
}
