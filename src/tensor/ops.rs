//! Convenience methods on Tensor that delegate to Client ops
//!
//! These methods provide ergonomic `tensor.add(&other)` style calls
//! that internally get the client and delegate to the appropriate trait.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::traits::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, CumulativeOps, IndexingOps, MatmulOps,
    NormalizationOps, PaddingMode, ReduceOps, ScalarOps, ShapeOps, TypeConversionOps, UnaryOps,
    UtilityOps,
};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Binary arithmetic
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: BinaryOps<R>,
{
    /// Element-wise addition: self + other
    pub fn add(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.add(self, other)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.sub(self, other)
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.mul(self, other)
    }

    /// Element-wise division: self / other
    pub fn div(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.div(self, other)
    }

    /// Element-wise power: self ^ other
    pub fn pow(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.pow(self, other)
    }

    /// Element-wise maximum: max(self, other)
    pub fn maximum(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.maximum(self, other)
    }

    /// Element-wise minimum: min(self, other)
    pub fn minimum(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.minimum(self, other)
    }
}

// ============================================================================
// Unary operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: UnaryOps<R>,
{
    /// Element-wise negation
    pub fn neg(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.neg(self)
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.abs(self)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.sqrt(self)
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.exp(self)
    }

    /// Element-wise natural log
    pub fn log(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.log(self)
    }

    /// Element-wise sine
    pub fn sin(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.sin(self)
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.cos(self)
    }

    /// Element-wise tangent
    pub fn tan(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.tan(self)
    }

    /// Element-wise hyperbolic tangent
    pub fn tanh(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.tanh(self)
    }

    /// Element-wise reciprocal (1/x)
    pub fn recip(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.recip(self)
    }

    /// Element-wise floor
    pub fn floor(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.floor(self)
    }

    /// Element-wise ceil
    pub fn ceil(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.ceil(self)
    }

    /// Element-wise round to nearest integer, ties away from zero.
    ///
    /// Follows Rust's `f32::round`: 2.5 becomes 3.0, -2.5 becomes -3.0.
    /// For the `np.round`/`torch.round` behaviour, use
    /// [`Tensor::round_ties_even`].
    pub fn round(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.round(self)
    }

    /// Element-wise round to nearest integer, ties to even.
    ///
    /// IEEE 754 roundTiesToEven, matching `np.round` and `torch.round`:
    /// 2.5 becomes 2.0, 3.5 becomes 4.0.
    pub fn round_ties_even(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.round_ties_even(self)
    }
}

// ============================================================================
// Scalar operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: ScalarOps<R>,
{
    /// Add scalar: self + scalar
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.add_scalar(self, scalar)
    }

    /// Multiply by scalar: self * scalar
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.mul_scalar(self, scalar)
    }

    /// Scale alias for mul_scalar
    pub fn scale(&self, scalar: f64) -> Result<Tensor<R>> {
        self.mul_scalar(scalar)
    }
}

// ============================================================================
// Activation functions
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: ActivationOps<R>,
{
    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.relu(self)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.sigmoid(self)
    }

    /// GELU activation
    pub fn gelu(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.gelu(self)
    }

    /// SiLU/Swish activation: x * sigmoid(x)
    pub fn silu(&self) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.silu(self)
    }

    /// Softmax along dimension
    pub fn softmax(&self, dim: isize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.softmax(self, dim)
    }

    /// Log-softmax along dimension: log(softmax(x, dim))
    pub fn log_softmax(&self, dim: isize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.log_softmax(self, dim)
    }

    /// Dropout: randomly zero elements with probability `p` during training
    pub fn dropout(&self, p: f64, training: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.dropout(self, p, training)
    }
}

// ============================================================================
// Reduction operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: ReduceOps<R>,
{
    /// Sum along dimensions
    pub fn sum(&self, dims: &[usize], keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.sum(self, dims, keepdim)
    }

    /// Mean along dimensions
    pub fn mean(&self, dims: &[usize], keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.mean(self, dims, keepdim)
    }

    /// Max along dimensions
    pub fn max(&self, dims: &[usize], keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.max(self, dims, keepdim)
    }

    /// Min along dimensions
    pub fn min(&self, dims: &[usize], keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.min(self, dims, keepdim)
    }
}

// ============================================================================
// Matrix operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: MatmulOps<R>,
{
    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.matmul(self, other)
    }
}

// ============================================================================
// Normalization
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: NormalizationOps<R>,
{
    /// RMS normalization: x / RMS(x) * weight
    pub fn rms_norm(&self, weight: &Tensor<R>, eps: f32) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.rms_norm(self, weight, eps)
    }

    /// Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
    pub fn layer_norm(&self, weight: &Tensor<R>, bias: &Tensor<R>, eps: f32) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.layer_norm(self, weight, bias, eps)
    }
}

// ============================================================================
// Comparison operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: CompareOps<R>,
{
    /// Element-wise equality
    pub fn eq(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.eq(self, other)
    }

    /// Element-wise greater than
    pub fn gt(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.gt(self, other)
    }

    /// Element-wise less than
    pub fn lt(&self, other: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.lt(self, other)
    }
}

// ============================================================================
// Indexing operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: IndexingOps<R>,
{
    /// Select elements along a dimension using indices
    pub fn index_select(&self, dim: usize, indices: &Tensor<R>) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.index_select(self, dim, indices)
    }

    /// Argmax along a dimension
    pub fn argmax(&self, dim: usize, keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.argmax(self, dim, keepdim)
    }

    /// Argmin along a dimension
    pub fn argmin(&self, dim: usize, keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.argmin(self, dim, keepdim)
    }

    /// Fill tensor with value where mask is true
    pub fn masked_fill(&self, mask: &Tensor<R>, value: f64) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.masked_fill(self, mask, value)
    }

    /// Assign `src` into a slice of `self` along `dim` starting at `start`.
    ///
    /// Returns a new tensor with the slice region replaced by `src`.
    pub fn slice_assign(&self, src: &Tensor<R>, dim: usize, start: usize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.slice_assign(self, src, dim, start)
    }
}

// ============================================================================
// Shape operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: ShapeOps<R>,
{
    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>> {
        if tensors.is_empty() {
            return Err(crate::error::Error::InvalidArgument {
                arg: "tensors",
                reason: "cannot concatenate empty list".into(),
            });
        }
        let client = R::default_client(tensors[0].device());
        client.cat(tensors, dim)
    }

    /// Stack tensors along a new dimension
    pub fn stack(tensors: &[&Tensor<R>], dim: isize) -> Result<Tensor<R>> {
        if tensors.is_empty() {
            return Err(crate::error::Error::InvalidArgument {
                arg: "tensors",
                reason: "cannot stack empty list".into(),
            });
        }
        let client = R::default_client(tensors[0].device());
        client.stack(tensors, dim)
    }
}

// ============================================================================
// Cumulative operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: CumulativeOps<R>,
{
    /// Cumulative sum along a dimension
    pub fn cumsum(&self, dim: isize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.cumsum(self, dim)
    }

    /// Cumulative product along a dimension
    pub fn cumprod(&self, dim: isize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.cumprod(self, dim)
    }

    /// Log-sum-exp along specified dimensions (numerically stable)
    ///
    /// Computes `log(sum(exp(x)))` in a numerically stable way:
    /// `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`
    pub fn logsumexp(&self, dims: &[usize], keepdim: bool) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.logsumexp(self, dims, keepdim)
    }
}

// ============================================================================
// Type conversion
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Convert tensor to a different dtype
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.cast(self, dtype)
    }
}

// ============================================================================
// Utility operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: UtilityOps<R>,
{
    /// Clamp values to [min, max]
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.clamp(self, min, max)
    }

    /// One-hot encode indices
    pub fn one_hot(&self, num_classes: usize) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.one_hot(self, num_classes)
    }
}

// ============================================================================
// Convolution operations
// ============================================================================

impl<R: Runtime> Tensor<R>
where
    R::Client: ConvOps<R>,
{
    /// 1D convolution
    pub fn conv1d(
        &self,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<R>> {
        let client = R::default_client(self.device());
        client.conv1d(self, weight, bias, stride, padding, dilation, groups)
    }
}
