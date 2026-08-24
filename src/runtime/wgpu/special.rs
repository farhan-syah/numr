//! WebGPU implementation of special mathematical functions
//!
//! This module implements the [`SpecialFunctions`] trait for WebGPU using
//! native WGSL compute shaders.
//!
//! # Supported Functions
//!
//! Unary: erf, erfc, erfinv, gamma, lgamma, digamma
//! Binary: beta, gammainc, gammaincc
//! Ternary: betainc
//!
//! # Numerical Algorithms
//!
//! - erf/erfc: Abramowitz & Stegun 7.1.26 rational approximation
//! - erfinv: Newton-Raphson iteration
//! - gamma/lgamma: Lanczos approximation (g=7, n=9)
//! - digamma: Asymptotic expansion + recurrence relation
//! - beta: exp(lgamma(a) + lgamma(b) - lgamma(a+b))
//! - betainc: Continued fraction expansion (Lentz's method)
//! - gammainc/gammaincc: Power series and continued fractions

use super::WgpuRuntime;
use super::client::WgpuClient;
use super::shaders::{
    launch_special_binary, launch_special_binary_with_two_ints, launch_special_ternary,
    launch_special_unary, launch_special_unary_with_2f32, launch_special_unary_with_3f32,
    launch_special_unary_with_int, launch_special_unary_with_two_ints,
};
use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// Use helpers from ops module
use super::ops::helpers::{alloc_output, get_tensor_buffer};

impl SpecialFunctions<WgpuRuntime> for WgpuClient {
    fn erf(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erf")
    }

    fn erfc(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erfc")
    }

    fn erfinv(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erfinv")
    }

    fn gamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "gamma")
    }

    fn lgamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "lgamma")
    }

    fn digamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "digamma")
    }

    fn beta(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, b, "beta")
    }

    fn betainc(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_ternary_special(self, a, b, x, "betainc")
    }

    fn gammainc(
        &self,
        a: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, x, "gammainc")
    }

    fn gammaincc(
        &self,
        a: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, x, "gammaincc")
    }

    fn gammaincinv(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != p.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: p.dtype(),
            });
        }
        if a.shape() != p.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, p, "gammaincinv")
    }

    fn betaincinv(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        p: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != p.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != p.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: p.shape().to_vec(),
            });
        }
        compute_ternary_special(self, a, b, p, "betaincinv")
    }

    fn bessel_j0(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_j0")
    }

    fn bessel_j1(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_j1")
    }

    fn bessel_y0(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_y0")
    }

    fn bessel_y1(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_y1")
    }

    fn bessel_i0(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_i0")
    }

    fn bessel_i1(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_i1")
    }

    fn bessel_k0(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_k0")
    }

    fn bessel_k1(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "bessel_k1")
    }

    // ========================================================================
    // Extended Special Functions
    // ========================================================================

    fn ellipk(&self, m: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(m.dtype())?;
        compute_unary_special(self, m, "ellipk")
    }

    fn ellipe(&self, m: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(m.dtype())?;
        compute_unary_special(self, m, "ellipe")
    }

    fn hyp2f1(
        &self,
        a: f64,
        b: f64,
        c: f64,
        z: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(z.dtype())?;
        compute_unary_special_with_params_3f64(self, z, "hyp2f1", a, b, c)
    }

    fn hyp1f1(&self, a: f64, b: f64, z: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(z.dtype())?;
        compute_unary_special_with_params_2f64(self, z, "hyp1f1", a, b)
    }

    fn airy_ai(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "airy_ai")
    }

    fn airy_bi(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "airy_bi")
    }

    fn legendre_p(&self, n: i32, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special_with_int(self, x, "legendre_p", n)
    }

    fn legendre_p_assoc(
        &self,
        n: i32,
        m: i32,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special_with_two_ints(self, x, "legendre_p_assoc", n, m)
    }

    fn sph_harm(
        &self,
        n: i32,
        m: i32,
        theta: &Tensor<WgpuRuntime>,
        phi: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(theta.dtype())?;
        if theta.dtype() != phi.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: theta.dtype(),
                rhs: phi.dtype(),
            });
        }
        if theta.shape() != phi.shape() {
            return Err(Error::ShapeMismatch {
                expected: theta.shape().to_vec(),
                got: phi.shape().to_vec(),
            });
        }
        compute_binary_special_with_two_ints(self, theta, phi, "sph_harm", n, m)
    }

    fn fresnel_s(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "fresnel_s")
    }

    fn fresnel_c(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "fresnel_c")
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute unary special function using native WGSL shader
fn compute_unary_special(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = x.shape().to_vec();
    let numel = x.numel();

    // Get input buffer and allocate output
    let input_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_unary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}

/// Compute binary special function using native WGSL shader
fn compute_binary_special(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = a.shape().to_vec();
    let numel = a.numel();

    // Get input buffers and allocate output
    let input_a_buffer = get_tensor_buffer(a)?;
    let input_b_buffer = get_tensor_buffer(b)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_binary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_a_buffer,
        &input_b_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}

/// Compute ternary special function using native WGSL shader
fn compute_ternary_special(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = a.shape().to_vec();
    let numel = a.numel();

    // Get input buffers and allocate output
    let input_a_buffer = get_tensor_buffer(a)?;
    let input_b_buffer = get_tensor_buffer(b)?;
    let input_x_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_ternary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_a_buffer,
        &input_b_buffer,
        &input_x_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}

/// Compute unary special function with one i32 parameter
fn compute_unary_special_with_int(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    op: &str,
    n: i32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = x.shape().to_vec();
    let numel = x.numel();

    let input_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    launch_special_unary_with_int(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        n,
        dtype,
    )?;

    Ok(output)
}

/// Compute unary special function with two i32 parameters
fn compute_unary_special_with_two_ints(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    op: &str,
    n: i32,
    m: i32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = x.shape().to_vec();
    let numel = x.numel();

    let input_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    launch_special_unary_with_two_ints(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        n,
        m,
        dtype,
    )?;

    Ok(output)
}

/// Compute binary special function with two i32 parameters (for sph_harm)
fn compute_binary_special_with_two_ints(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    op: &str,
    n: i32,
    m: i32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = a.shape().to_vec();
    let numel = a.numel();

    let input_a_buffer = get_tensor_buffer(a)?;
    let input_b_buffer = get_tensor_buffer(b)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    launch_special_binary_with_two_ints(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_a_buffer,
        &input_b_buffer,
        &output_buffer,
        numel as u32,
        n,
        m,
        dtype,
    )?;

    Ok(output)
}

/// Compute unary special function with two f64 parameters (for hyp1f1)
fn compute_unary_special_with_params_2f64(
    client: &WgpuClient,
    z: &Tensor<WgpuRuntime>,
    op: &str,
    a: f64,
    b: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = z.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = z.shape().to_vec();
    let numel = z.numel();

    let input_buffer = get_tensor_buffer(z)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    launch_special_unary_with_2f32(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        a as f32,
        b as f32,
        dtype,
    )?;

    Ok(output)
}

/// Compute unary special function with three f64 parameters (for hyp2f1)
fn compute_unary_special_with_params_3f64(
    client: &WgpuClient,
    z: &Tensor<WgpuRuntime>,
    op: &str,
    a: f64,
    b: f64,
    c: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = z.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = z.shape().to_vec();
    let numel = z.numel();

    let input_buffer = get_tensor_buffer(z)?;
    let output = alloc_output(client, &shape, dtype)?;
    let output_buffer = get_tensor_buffer(&output)?;

    launch_special_unary_with_3f32(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        a as f32,
        b as f32,
        c as f32,
        dtype,
    )?;

    Ok(output)
}
