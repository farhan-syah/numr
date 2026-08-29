#![allow(clippy::approx_constant, clippy::excessive_precision)]
// Backend parity tests for UnaryOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::UnaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::assert_parity_u32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

#[derive(Clone)]
struct TestInput {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl TestInput {
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        TestInput { data, shape }
    }
}

/// Apply unary operation to tensor
fn apply_unary_op<R: Runtime>(
    client: &impl UnaryOps<R>,
    op: &str,
    x: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "neg" => client.neg(x),
        "abs" => client.abs(x),
        "sign" => client.sign(x),
        "sqrt" => client.sqrt(x),
        "rsqrt" => client.rsqrt(x),
        "square" => client.square(x),
        "cbrt" => client.cbrt(x),
        "recip" => client.recip(x),
        "exp" => client.exp(x),
        "exp2" => client.exp2(x),
        "expm1" => client.expm1(x),
        "log" => client.log(x),
        "log2" => client.log2(x),
        "log10" => client.log10(x),
        "log1p" => client.log1p(x),
        "sin" => client.sin(x),
        "cos" => client.cos(x),
        "tan" => client.tan(x),
        "asin" => client.asin(x),
        "acos" => client.acos(x),
        "atan" => client.atan(x),
        "sinh" => client.sinh(x),
        "cosh" => client.cosh(x),
        "tanh" => client.tanh(x),
        "asinh" => client.asinh(x),
        "acosh" => client.acosh(x),
        "atanh" => client.atanh(x),
        "floor" => client.floor(x),
        "ceil" => client.ceil(x),
        "round" => client.round(x),
        "round_ties_even" => client.round_ties_even(x),
        "trunc" => client.trunc(x),
        _ => panic!("Unknown unary op: {}", op),
    }
}

fn test_unary_parity(op: &str, test_inputs: &[TestInput], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_inputs
        .iter()
        .map(|input| {
            let tensor =
                tensor_from_f64(&input.data, &input.shape, dtype, &cpu_device, &cpu_client)
                    .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_unary_op(&cpu_client, op, &tensor)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, input) in test_inputs.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&input.data, &input.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_unary_op(&cuda_client, op, &tensor)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, input) in test_inputs.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&input.data, &input.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_unary_op(&wgpu_client, op, &tensor)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! unary_case {
    ($name:ident, $op:expr, $inputs:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_unary_parity($op, $inputs, dtype);
            }
        }
    };
}

// ============================================================================
// Unary Operation Parity Tests
// ============================================================================

unary_case!(
    test_neg_parity,
    "neg",
    &[
        TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
        TestInput::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
    ]
);

unary_case!(
    test_abs_parity,
    "abs",
    &[
        TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
        TestInput::new(vec![-1.0, -2.0, -3.0, -4.0], vec![2, 2]),
    ]
);

unary_case!(
    test_sign_parity,
    "sign",
    &[
        TestInput::new(vec![1.0, -2.0, 0.0, -4.0], vec![4]),
        TestInput::new(vec![-5.0, 0.0, 5.0, 0.0], vec![2, 2]),
    ]
);

unary_case!(
    test_sqrt_parity,
    "sqrt",
    &[
        TestInput::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]),
        TestInput::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    ]
);

unary_case!(
    test_rsqrt_parity,
    "rsqrt",
    &[
        TestInput::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]),
        TestInput::new(vec![2.0, 4.0, 8.0, 16.0], vec![2, 2]),
    ]
);

unary_case!(
    test_square_parity,
    "square",
    &[
        TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
        TestInput::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]),
    ]
);

unary_case!(
    test_cbrt_parity,
    "cbrt",
    &[
        TestInput::new(vec![1.0, 8.0, 27.0, 64.0], vec![4]),
        TestInput::new(vec![-8.0, 0.0, 8.0, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_recip_parity,
    "recip",
    &[
        TestInput::new(vec![1.0, 2.0, 4.0, 5.0], vec![4]),
        TestInput::new(vec![2.0, 4.0, 5.0, 10.0], vec![2, 2]),
    ]
);

unary_case!(
    test_exp_parity,
    "exp",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
        TestInput::new(vec![0.5, -0.5, 1.0, -1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_exp2_parity,
    "exp2",
    &[
        TestInput::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]),
        TestInput::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]),
    ]
);

unary_case!(
    test_expm1_parity,
    "expm1",
    &[
        TestInput::new(vec![0.0, 0.1, -0.1, 0.5], vec![4]),
        TestInput::new(vec![0.0, 0.01, -0.01, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_log_parity,
    "log",
    &[
        TestInput::new(vec![1.0, 2.0, 4.0, 10.0], vec![4]),
        TestInput::new(vec![1.0, 2.0, 5.0, 10.0], vec![2, 2]),
    ]
);

unary_case!(
    test_log2_parity,
    "log2",
    &[
        TestInput::new(vec![1.0, 2.0, 4.0, 8.0], vec![4]),
        TestInput::new(vec![2.0, 4.0, 8.0, 16.0], vec![2, 2]),
    ]
);

unary_case!(
    test_log10_parity,
    "log10",
    &[
        TestInput::new(vec![1.0, 10.0, 100.0, 1000.0], vec![4]),
        TestInput::new(vec![10.0, 100.0, 1000.0, 10000.0], vec![2, 2]),
    ]
);

unary_case!(
    test_log1p_parity,
    "log1p",
    &[
        TestInput::new(vec![0.0, 0.1, 1.0, 9.0], vec![4]),
        TestInput::new(vec![0.0, 0.01, 1.0, 99.0], vec![2, 2]),
    ]
);

unary_case!(
    test_sin_parity,
    "sin",
    &[
        TestInput::new(vec![0.0, 1.57079633, 3.14159265, -1.57079633], vec![4]),
        TestInput::new(vec![0.5, 1.0, -0.5, -1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_cos_parity,
    "cos",
    &[
        TestInput::new(vec![0.0, 1.57079633, 3.14159265, -1.57079633], vec![4]),
        TestInput::new(vec![0.5, 1.0, -0.5, -1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_tan_parity,
    "tan",
    &[
        TestInput::new(vec![0.0, 0.4, -0.4, 0.785398163], vec![4]),
        TestInput::new(vec![0.1, -0.1, 0.2, -0.2], vec![2, 2]),
    ]
);

unary_case!(
    test_asin_parity,
    "asin",
    &[
        TestInput::new(vec![0.0, 0.5, -0.5, 1.0], vec![4]),
        TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_acos_parity,
    "acos",
    &[
        TestInput::new(vec![0.0, 0.5, -0.5, 1.0], vec![4]),
        TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_atan_parity,
    "atan",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 10.0], vec![4]),
        TestInput::new(vec![-10.0, -1.0, 1.0, 10.0], vec![2, 2]),
    ]
);

unary_case!(
    test_sinh_parity,
    "sinh",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
        TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_cosh_parity,
    "cosh",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
        TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_tanh_parity,
    "tanh",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
        TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
    ]
);

unary_case!(
    test_asinh_parity,
    "asinh",
    &[
        TestInput::new(vec![0.0, 1.0, -1.0, 10.0], vec![4]),
        TestInput::new(vec![-10.0, -1.0, 1.0, 10.0], vec![2, 2]),
    ]
);

unary_case!(
    test_acosh_parity,
    "acosh",
    &[
        TestInput::new(vec![1.0, 2.0, 5.0, 10.0], vec![4]),
        TestInput::new(vec![1.0, 1.5, 2.5, 10.0], vec![2, 2]),
    ]
);

unary_case!(
    test_atanh_parity,
    "atanh",
    &[
        TestInput::new(vec![0.0, 0.5, -0.5, 0.9], vec![4]),
        TestInput::new(vec![-0.5, -0.1, 0.1, 0.5], vec![2, 2]),
    ]
);

unary_case!(
    test_floor_parity,
    "floor",
    &[
        TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
        TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
    ]
);

unary_case!(
    test_ceil_parity,
    "ceil",
    &[
        TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
        TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
    ]
);

unary_case!(
    test_round_parity,
    "round",
    &[
        TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
        TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
        // Both shapes of double-rounding error the old floor(x + 0.5) shader
        // produced: 0.5 - 2^-25 is the largest f32 below 0.5, which the
        // addition rounded up to 1.0 before the floor; 8388609.0 (2^23 + 1)
        // is where the addition itself loses a whole ULP.
        TestInput::new(
            vec![
                0.5 - 2f64.powi(-25),
                -(0.5 - 2f64.powi(-25)),
                8388609.0,
                -8388609.0,
            ],
            vec![4],
        ),
    ]
);

// Every element of the second case is a tie, where round (ties away from zero)
// and round_ties_even disagree; a backend that reused the wrong rounding mode
// shows up here.
unary_case!(
    test_round_ties_even_parity,
    "round_ties_even",
    &[
        TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
        TestInput::new(vec![0.5, 1.5, -0.5, -1.5, 2.5, 3.5, -2.5, -3.5], vec![2, 4]),
    ]
);

unary_case!(
    test_trunc_parity,
    "trunc",
    &[
        TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
        TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
    ]
);

// ============================================================================
// isnan / isinf - boolean output, F32-only input (NaN/Inf are float concepts)
// ============================================================================

#[test]
fn test_isnan_parity() {
    let data = vec![0.0f32, f32::NAN, 1.0, f32::NAN];
    let shape = vec![4];
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(&data, &shape, &cpu_device).unwrap();
    let cpu_result = cpu_client.isnan(&cpu_tensor).unwrap().to_vec::<u8>();
    let cpu_result_u32: Vec<u32> = cpu_result.iter().map(|&x| x as u32).collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(&data, &shape, &cuda_device).unwrap();
        let cuda_result = cuda_client.isnan(&cuda_tensor).unwrap().to_vec::<u8>();
        assert_parity_u32(
            &cpu_result_u32,
            &cuda_result.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            "isnan_cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(&data, &shape, &wgpu_device).unwrap();
        let wgpu_result = wgpu_client.isnan(&wgpu_tensor).unwrap().to_vec::<u32>();
        assert_parity_u32(&cpu_result_u32, &wgpu_result, "isnan_wgpu");
    });
}

#[test]
fn test_isinf_parity() {
    let data = vec![0.0f32, f32::INFINITY, 1.0, f32::NEG_INFINITY];
    let shape = vec![4];
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(&data, &shape, &cpu_device).unwrap();
    let cpu_result = cpu_client.isinf(&cpu_tensor).unwrap().to_vec::<u8>();
    let cpu_result_u32: Vec<u32> = cpu_result.iter().map(|&x| x as u32).collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(&data, &shape, &cuda_device).unwrap();
        let cuda_result = cuda_client.isinf(&cuda_tensor).unwrap().to_vec::<u8>();
        assert_parity_u32(
            &cpu_result_u32,
            &cuda_result.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            "isinf_cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(&data, &shape, &wgpu_device).unwrap();
        let wgpu_result = wgpu_client.isinf(&wgpu_tensor).unwrap().to_vec::<u32>();
        assert_parity_u32(&cpu_result_u32, &wgpu_result, "isinf_wgpu");
    });
}

// ============================================================================
// Empty tensor - CUDA parity (native_unary_op)
// ============================================================================

/// Before the fix, `elementwise_launch_config(0)` produced `grid.x == 0`,
/// which is an invalid CUDA launch. `neg` on a `[0]` F32 tensor must succeed
/// and return an empty tensor, matching the CPU backend's natural no-op.
#[cfg(feature = "cuda")]
#[test]
fn test_neg_empty_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let empty = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let result = cuda_client.neg(&empty);
        assert!(
            result.is_ok(),
            "neg on empty CUDA tensor should succeed, got {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.shape(), &[0]);
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.numel(), 0);
    });
}
