//! CPU implementation of activation operations.

use crate::error::{Error, Result};
use crate::ops::impl_generic::activation::{
    dropout_impl, log_softmax_impl, softmax_with_bias_impl, softplus_impl,
};
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps,
    activation::normalize_softmax_dim,
};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{
        ActivationOp, FusedActivationMulOp, activation_op_impl, dispatch_dtype, elu_impl,
        ensure_contiguous, fused_activation_mul_impl, leaky_relu_impl,
    },
    kernels,
};
use crate::tensor::Tensor;

/// ActivationOps implementation for CPU runtime.
impl ActivationOps<CpuRuntime> for CpuClient {
    fn relu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Relu, "relu")
    }

    fn sigmoid(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Sigmoid, "sigmoid")
    }

    fn silu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Silu, "silu")
    }

    fn gelu(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        activation_op_impl(self, a, ActivationOp::Gelu, "gelu")
    }

    fn silu_mul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_activation_mul_impl(self, a, b, FusedActivationMulOp::SiluMul, "silu_mul")
    }

    fn gelu_mul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_activation_mul_impl(self, a, b, FusedActivationMulOp::GeluMul, "gelu_mul")
    }

    fn relu_mul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_activation_mul_impl(self, a, b, FusedActivationMulOp::ReluMul, "relu_mul")
    }

    fn sigmoid_mul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_activation_mul_impl(self, a, b, FusedActivationMulOp::SigmoidMul, "sigmoid_mul")
    }

    fn silu_mul_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        // silu(a) = a * sigmoid(a)
        let silu_a = self.silu(a)?;
        let d_b = self.mul(grad, &silu_a)?;
        // silu'(x) = sigmoid(x) * (1 + x - silu(x))
        let sigmoid_a = self.sigmoid(a)?;
        let one_plus_a = self.add_scalar(a, 1.0)?;
        let one_plus_a_minus_silu = self.sub(&one_plus_a, &silu_a)?;
        let silu_deriv = self.mul(&sigmoid_a, &one_plus_a_minus_silu)?;
        let grad_times_b = self.mul(grad, b)?;
        let d_a = self.mul(&grad_times_b, &silu_deriv)?;
        Ok((d_a, d_b))
    }

    fn gelu_mul_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let gelu_a = self.gelu(a)?;
        let d_b = self.mul(grad, &gelu_a)?;
        // gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*inner'
        // inner = sqrt(2/π) * (x + 0.044715*x³), inner' = sqrt(2/π)*(1 + 3*0.044715*x²)
        let x_sq = self.mul(a, a)?;
        let x_cu = self.mul(&x_sq, a)?;
        let coef_x_cu = self.mul_scalar(&x_cu, 0.044715)?;
        let inner_arg = self.add(a, &coef_x_cu)?;
        let sqrt_2_pi: f64 = 0.7978845608028654;
        let inner = self.mul_scalar(&inner_arg, sqrt_2_pi)?;
        // Use tanh op directly — avoids exp overflow for low-precision dtypes (F16/FP8)
        let tanh_inner = self.tanh(&inner)?;
        // term1 = 0.5*(1+tanh(inner))
        let one_plus_tanh = self.add_scalar(&tanh_inner, 1.0)?;
        let term1 = self.mul_scalar(&one_plus_tanh, 0.5)?;
        // sech²(inner) = 1 - tanh²(inner)
        let tanh_sq = self.mul(&tanh_inner, &tanh_inner)?;
        let sech_sq = self.add_scalar(&tanh_sq, -1.0)?;
        let sech_sq = self.neg(&sech_sq)?;
        // inner' = sqrt(2/π) * (1 + 3*0.044715*x²)
        let three_coef_x_sq = self.mul_scalar(&x_sq, 3.0 * 0.044715)?;
        let inner_deriv_unscaled = self.add_scalar(&three_coef_x_sq, 1.0)?;
        let inner_deriv = self.mul_scalar(&inner_deriv_unscaled, sqrt_2_pi)?;
        // term2 = 0.5 * x * sech²(inner) * inner'
        let x_sech_sq = self.mul(a, &sech_sq)?;
        let x_sech_sq_inner_d = self.mul(&x_sech_sq, &inner_deriv)?;
        let term2 = self.mul_scalar(&x_sech_sq_inner_d, 0.5)?;
        let gelu_deriv = self.add(&term1, &term2)?;
        let grad_times_b = self.mul(grad, b)?;
        let d_a = self.mul(&grad_times_b, &gelu_deriv)?;
        Ok((d_a, d_b))
    }

    fn relu_mul_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let relu_a = self.relu(a)?;
        let d_b = self.mul(grad, &relu_a)?;
        // relu'(x) = 1 if x > 0, else 0
        let zeros = Tensor::<CpuRuntime>::try_zeros(a.shape(), a.dtype(), a.device())?;
        let ones = Tensor::<CpuRuntime>::try_ones(a.shape(), a.dtype(), a.device())?;
        let mask = self.gt(a, &zeros)?;
        let relu_deriv = self.where_cond(&mask, &ones, &zeros)?;
        let grad_times_b = self.mul(grad, b)?;
        let d_a = self.mul(&grad_times_b, &relu_deriv)?;
        Ok((d_a, d_b))
    }

    fn sigmoid_mul_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let sigmoid_a = self.sigmoid(a)?;
        let d_b = self.mul(grad, &sigmoid_a)?;
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let one_minus_sig = self.add_scalar(&sigmoid_a, -1.0)?;
        let one_minus_sig = self.neg(&one_minus_sig)?;
        let sigmoid_deriv = self.mul(&sigmoid_a, &one_minus_sig)?;
        let grad_times_b = self.mul(grad, b)?;
        let d_a = self.mul(&grad_times_b, &sigmoid_deriv)?;
        Ok((d_a, d_b))
    }

    fn leaky_relu(
        &self,
        a: &Tensor<CpuRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        leaky_relu_impl(self, a, negative_slope)
    }

    fn elu(&self, a: &Tensor<CpuRuntime>, alpha: f64) -> Result<Tensor<CpuRuntime>> {
        elu_impl(self, a, alpha)
    }

    fn softmax(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        // Normalize dimension
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::try_empty(a.shape(), dtype, &self.device)?;

        let shape = a.shape();

        // Calculate outer_size (product of dims before softmax dim)
        // and dim_size (size of softmax dim)
        // and inner_size (product of dims after softmax dim)
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        // For softmax, we need the data laid out so that the softmax dimension is contiguous
        // If dim is the last dimension, we can use the simple kernel
        // Otherwise, we need to iterate

        if dim_idx == ndim - 1 {
            // Simple case: softmax over last dimension
            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::softmax_kernel::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        dim_size,
                    );
                }
            }, "softmax");
        } else {
            // General case: softmax over non-last dimension
            // Pre-allocate buffer outside loops to avoid repeated allocations
            let a_ptr = a_contig.ptr();
            let out_ptr = out.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    softmax_non_last_dim::<T>(
                        a_ptr as *const T,
                        out_ptr as *mut T,
                        outer_size,
                        dim_size,
                        inner_size,
                    );
                }
            }, "softmax");
        }

        Ok(out)
    }

    fn softmax_bwd(
        &self,
        grad: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        dim: isize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = grad.dtype();
        let ndim = grad.ndim();
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let grad_contig = ensure_contiguous(grad)?;
        let output_contig = ensure_contiguous(output)?;
        let d_input = Tensor::<CpuRuntime>::try_empty(grad.shape(), dtype, &self.device)?;

        let shape = grad.shape();
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        if dim_idx == ndim - 1 {
            // Last dim: use fused SIMD kernel
            let g_ptr = grad_contig.ptr();
            let o_ptr = output_contig.ptr();
            let d_ptr = d_input.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    kernels::softmax_bwd_kernel::<T>(
                        g_ptr as *const T,
                        o_ptr as *const T,
                        d_ptr as *mut T,
                        outer_size,
                        dim_size,
                    );
                }
            }, "softmax_bwd");
        } else {
            // Non-last dim: strided access pattern
            let g_ptr = grad_contig.ptr();
            let o_ptr = output_contig.ptr();
            let d_ptr = d_input.ptr();

            dispatch_dtype!(dtype, T => {
                unsafe {
                    softmax_bwd_non_last_dim::<T>(
                        g_ptr as *const T,
                        o_ptr as *const T,
                        d_ptr as *mut T,
                        outer_size,
                        dim_size,
                        inner_size,
                    );
                }
            }, "softmax_bwd");
        }

        Ok(d_input)
    }

    fn softmax_with_bias(
        &self,
        a: &Tensor<CpuRuntime>,
        bias: &Tensor<CpuRuntime>,
        dim: isize,
    ) -> Result<Tensor<CpuRuntime>> {
        softmax_with_bias_impl(self, a, bias, dim)
    }

    fn softplus(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        softplus_impl(self, a)
    }

    fn log_softmax(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        log_softmax_impl(self, a, dim)
    }

    fn dropout(
        &self,
        a: &Tensor<CpuRuntime>,
        p: f64,
        training: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        dropout_impl(self, a, p, training)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::ActivationOps;
    use crate::runtime::cpu::CpuDevice;

    #[test]
    fn test_log_softmax_basic() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let result = client.log_softmax(&input, -1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // log_softmax should sum to something reasonable
        // exp(log_softmax) should sum to 1
        let exp_sum: f32 = data.iter().map(|x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);

        // Values should be negative (log of probability)
        for &v in &data {
            assert!(v < 0.0);
        }
    }

    #[test]
    fn test_log_softmax_2d() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &device,
        )
        .unwrap();
        let result = client.log_softmax(&input, -1).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Each row should independently sum (in exp space) to 1
        let row1_sum: f32 = data[0..3].iter().map(|x| x.exp()).sum();
        let row2_sum: f32 = data[3..6].iter().map(|x| x.exp()).sum();
        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dropout_training() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::try_ones(&[1000], crate::dtype::DType::F32, &device).unwrap();
        let result = client.dropout(&input, 0.5, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // Some values should be 0 (dropped), others should be 2.0 (scaled by 1/(1-0.5))
        let zeros = data.iter().filter(|&&v| v == 0.0).count();
        let scaled = data.iter().filter(|&&v| (v - 2.0).abs() < 1e-5).count();

        // With p=0.5, roughly half should be dropped (allow wide margin for randomness)
        assert!(zeros > 200, "too few zeros: {zeros}");
        assert!(zeros < 800, "too many zeros: {zeros}");
        assert_eq!(zeros + scaled, 1000);
    }

    #[test]
    fn test_dropout_inference() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let result = client.dropout(&input, 0.5, false).unwrap();
        let data: Vec<f32> = result.to_vec();

        // During inference, dropout is identity
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_p_zero() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let result = client.dropout(&input, 0.0, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // p=0 means no dropout
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_p_one() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let input =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let result = client.dropout(&input, 1.0, true).unwrap();
        let data: Vec<f32> = result.to_vec();

        // p=1 means all dropped
        for &v in &data {
            assert!((v).abs() < 1e-6);
        }
    }
}

/// Softmax backward for non-last dimension (strided access pattern).
///
/// d_input = output * (grad - dot), where dot = sum(grad * output) along dim.
unsafe fn softmax_bwd_non_last_dim<T: crate::dtype::Element>(
    grad: *const T,
    output: *const T,
    d_input: *mut T,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) {
    unsafe {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                let stride = inner_size;

                // Pass 1: dot = sum(grad * output) along dim
                let mut dot = 0.0f64;
                for d in 0..dim_size {
                    let idx = base_idx + d * stride;
                    dot += (*grad.add(idx)).to_f64() * (*output.add(idx)).to_f64();
                }

                // Pass 2: d_input = output * (grad - dot)
                for d in 0..dim_size {
                    let idx = base_idx + d * stride;
                    let g = (*grad.add(idx)).to_f64();
                    let o = (*output.add(idx)).to_f64();
                    *d_input.add(idx) = T::from_f64(o * (g - dot));
                }
            }
        }
    }
}

unsafe fn softmax_non_last_dim<T: crate::dtype::Element>(
    a_ptr: *const T,
    out_ptr: *mut T,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) {
    unsafe {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                let stride = inner_size;

                // Pass 1: Online max + sum (reads strided input once)
                let mut max_val = (*a_ptr.add(base_idx)).to_f64();
                let mut sum = 1.0f64;
                for d in 1..dim_size {
                    let idx = base_idx + d * stride;
                    let val = (*a_ptr.add(idx)).to_f64();
                    if val > max_val {
                        sum = sum * (max_val - val).exp() + 1.0;
                        max_val = val;
                    } else {
                        sum += (val - max_val).exp();
                    }
                }

                // Pass 2: exp(x - max) / sum (reads input, writes output)
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for d in 0..dim_size {
                    let idx = base_idx + d * stride;
                    let val = (*a_ptr.add(idx)).to_f64();
                    *out_ptr.add(idx) = T::from_f64((val - max_val).exp() * inv_sum);
                }
            }
        }
    }
}
