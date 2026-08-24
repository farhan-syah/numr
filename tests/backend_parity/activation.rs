// Backend parity tests for fused activation-mul operations (ActivationOps trait)
//
// Tests: silu_mul, gelu_mul, relu_mul, sigmoid_mul (forward)
//        silu_mul_bwd, gelu_mul_bwd, relu_mul_bwd, sigmoid_mul_bwd (backward)
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.

use numr::dtype::DType;
use numr::ops::ActivationOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
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
struct FusedTestCase {
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
}

impl FusedTestCase {
    fn new(a: Vec<f64>, b: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { a, b, shape }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(clippy::enum_variant_names)]
enum FusedActivationOp {
    SiluMul,
    GeluMul,
    ReluMul,
    SigmoidMul,
}

fn apply_fused_fwd<R: Runtime>(
    client: &impl ActivationOps<R>,
    op: FusedActivationOp,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        FusedActivationOp::SiluMul => client.silu_mul(a, b),
        FusedActivationOp::GeluMul => client.gelu_mul(a, b),
        FusedActivationOp::ReluMul => client.relu_mul(a, b),
        FusedActivationOp::SigmoidMul => client.sigmoid_mul(a, b),
    }
}

fn apply_fused_bwd<R: Runtime>(
    client: &impl ActivationOps<R>,
    op: FusedActivationOp,
    grad: &Tensor<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<(Tensor<R>, Tensor<R>)> {
    match op {
        FusedActivationOp::SiluMul => client.silu_mul_bwd(grad, a, b),
        FusedActivationOp::GeluMul => client.gelu_mul_bwd(grad, a, b),
        FusedActivationOp::ReluMul => client.relu_mul_bwd(grad, a, b),
        FusedActivationOp::SigmoidMul => client.sigmoid_mul_bwd(grad, a, b),
    }
}

// ============================================================================
// Forward parity tests
// ============================================================================

fn test_fused_fwd_parity(op: FusedActivationOp, test_cases: &[FusedTestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_fused_fwd(&cpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_fused_fwd(&cuda_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op:?} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_fused_fwd(&wgpu_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op:?} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Backward parity tests
// ============================================================================

fn test_fused_bwd_parity(op: FusedActivationOp, test_cases: &[FusedTestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    // Use the same data for grad as a simple ones-like pattern
    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            // Use ones as grad for simplicity
            let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
            let grad = tensor_from_f64(&grad_data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_fused_bwd(&cpu_client, op, &grad, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?}_bwd failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let (d_a, d_b) = apply_fused_bwd(&cuda_client, op, &grad, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op:?}_bwd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &d_a,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("{op:?}_bwd d_a CUDA vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &d_b,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("{op:?}_bwd d_b CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let (d_a, d_b) = apply_fused_bwd(&wgpu_client, op, &grad, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op:?}_bwd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &d_a,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("{op:?}_bwd d_a WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &d_b,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("{op:?}_bwd d_b WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Test data
// ============================================================================

fn standard_test_cases() -> Vec<FusedTestCase> {
    vec![
        // Small 1D
        FusedTestCase::new(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![0.5, 1.0, 1.5, 2.0, 0.3],
            vec![5],
        ),
        // 2D matrix
        FusedTestCase::new(
            vec![-1.0, 0.5, 1.5, -0.5, 2.0, -2.0],
            vec![1.0, 2.0, 0.5, 1.5, 0.3, 1.0],
            vec![2, 3],
        ),
        // Values near zero (important for derivative accuracy)
        FusedTestCase::new(
            vec![0.01, -0.01, 0.1, -0.1],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![4],
        ),
        // Larger values (tests saturation behavior)
        FusedTestCase::new(
            vec![5.0, -5.0, 10.0, -10.0],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![4],
        ),
        // Single element (edge case)
        FusedTestCase::new(vec![1.5], vec![2.0], vec![1]),
        // All zeros
        FusedTestCase::new(vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0], vec![4]),
        // Very large values (overflow risk for exp)
        FusedTestCase::new(
            vec![80.0, -80.0, 50.0, -50.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![4],
        ),
        // Very small values (subnormal territory)
        FusedTestCase::new(
            vec![1e-7, -1e-7, 1e-6, -1e-6],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![4],
        ),
        // Mixed signs in both operands
        FusedTestCase::new(
            vec![-3.0, 2.0, -1.0, 4.0],
            vec![-1.0, -0.5, 2.0, -2.0],
            vec![2, 2],
        ),
    ]
}

// ============================================================================
// Forward tests
// ============================================================================

#[test]
fn test_silu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::SiluMul, &cases, dtype);
    }
}

#[test]
fn test_gelu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::GeluMul, &cases, dtype);
    }
}

#[test]
fn test_relu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::ReluMul, &cases, dtype);
    }
}

#[test]
fn test_sigmoid_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::SigmoidMul, &cases, dtype);
    }
}

// ============================================================================
// Backward tests
// ============================================================================

#[test]
fn test_silu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::SiluMul, &cases, dtype);
    }
}

#[test]
fn test_gelu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::GeluMul, &cases, dtype);
    }
}

#[test]
fn test_relu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::ReluMul, &cases, dtype);
    }
}

#[test]
fn test_sigmoid_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::SigmoidMul, &cases, dtype);
    }
}

// ============================================================================
// Softmax parity tests
// ============================================================================

fn softmax_test_shapes() -> Vec<(Vec<f64>, Vec<usize>, isize)> {
    vec![
        // (data, shape, dim)
        (vec![1.0, 2.0, 3.0], vec![3], -1),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], -1),
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 0),
        (
            (0..24).map(|i| i as f64 * 0.1 - 1.0).collect(),
            vec![2, 3, 4],
            1,
        ),
        (
            (0..24).map(|i| i as f64 * 0.1 - 1.0).collect(),
            vec![2, 3, 4],
            -1,
        ),
        // Single element (should produce 1.0)
        (vec![5.0], vec![1], -1),
        // All identical values (uniform distribution)
        (vec![1.0, 1.0, 1.0, 1.0], vec![4], -1),
        // Very large values (overflow risk without max subtraction)
        (vec![100.0, 200.0, 300.0], vec![3], -1),
        // Very negative values
        (vec![-100.0, -200.0, -50.0], vec![3], -1),
        // Mixed extreme values (tests numerical stability)
        (vec![-80.0, 0.0, 80.0], vec![3], -1),
        // All zeros
        (vec![0.0, 0.0, 0.0], vec![3], -1),
        // 2D with dim=0 single row
        (vec![1.0, 2.0, 3.0], vec![1, 3], 0),
    ]
}

fn test_softmax_parity_for_dtype(dtype: DType) {
    if !is_dtype_supported("cpu", dtype) {
        return;
    }

    let (cpu_client, cpu_device) = create_cpu_client();

    for (data, shape, dim) in softmax_test_shapes() {
        let input_cpu = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client).unwrap();
        let result_cpu = cpu_client
            .softmax(&input_cpu, dim)
            .unwrap()
            .contiguous()
            .unwrap();

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let input_wgpu =
                    tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let result_wgpu = wgpu_client
                    .softmax(&input_wgpu, dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(&result_wgpu, &result_cpu, dtype, "softmax wgpu vs cpu");
            });
        }

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let input_cuda =
                    tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client).unwrap();
                let result_cuda = cuda_client
                    .softmax(&input_cuda, dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(&result_cuda, &result_cpu, dtype, "softmax cuda vs cpu");
            });
        }
    }
}

#[test]
fn test_softmax_parity() {
    for dtype in &[DType::F32, DType::F64] {
        test_softmax_parity_for_dtype(*dtype);
    }
}

fn test_softmax_bwd_parity_for_dtype(dtype: DType) {
    if !is_dtype_supported("cpu", dtype) {
        return;
    }

    let (cpu_client, cpu_device) = create_cpu_client();

    for (data, shape, dim) in softmax_test_shapes() {
        let input_cpu = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client).unwrap();
        let output_cpu = cpu_client
            .softmax(&input_cpu, dim)
            .unwrap()
            .contiguous()
            .unwrap();

        let grad_data: Vec<f64> = (0..data.len()).map(|i| (i as f64) * 0.1 - 0.5).collect();
        let grad_cpu =
            tensor_from_f64(&grad_data, &shape, dtype, &cpu_device, &cpu_client).unwrap();
        let d_input_cpu = cpu_client
            .softmax_bwd(&grad_cpu, &output_cpu, dim)
            .unwrap()
            .contiguous()
            .unwrap();

        // Get CPU output as f64 for creating GPU tensors
        let output_f64: Vec<f64> = if dtype == DType::F64 {
            output_cpu.to_vec::<f64>()
        } else {
            output_cpu
                .to_vec::<f32>()
                .iter()
                .map(|&x| x as f64)
                .collect()
        };

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let output_wgpu =
                    tensor_from_f64(&output_f64, &shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let grad_wgpu =
                    tensor_from_f64(&grad_data, &shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let d_input_wgpu = wgpu_client
                    .softmax_bwd(&grad_wgpu, &output_wgpu, dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_wgpu,
                    &d_input_cpu,
                    dtype,
                    "softmax_bwd wgpu vs cpu",
                );
            });
        }

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let output_cuda =
                    tensor_from_f64(&output_f64, &shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let grad_cuda =
                    tensor_from_f64(&grad_data, &shape, dtype, &cuda_device, &cuda_client).unwrap();
                let d_input_cuda = cuda_client
                    .softmax_bwd(&grad_cuda, &output_cuda, dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_cuda,
                    &d_input_cpu,
                    dtype,
                    "softmax_bwd cuda vs cpu",
                );
            });
        }
    }
}

#[test]
fn test_softmax_bwd_parity() {
    for dtype in &[DType::F32, DType::F64] {
        test_softmax_bwd_parity_for_dtype(*dtype);
    }
}

// ============================================================================
// softmax_with_bias parity tests
// ============================================================================

struct SoftmaxBiasCase {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    bias: Vec<f64>,
    bias_shape: Vec<usize>,
    dim: isize,
    label: &'static str,
}

fn softmax_bias_test_cases() -> Vec<SoftmaxBiasCase> {
    vec![
        // (i) [B,1,1,S] broadcast bias over [B,H,S,S] — the primary attention case
        SoftmaxBiasCase {
            a: (0..16).map(|i| i as f64 * 0.1 - 0.8).collect(),
            a_shape: vec![1, 2, 2, 4], // B=1,H=2,S=2,D=4
            bias: vec![0.0, -30000.0, 0.0, -30000.0],
            bias_shape: vec![1, 1, 1, 4],
            dim: -1,
            label: "broadcast [B,1,1,S]",
        },
        // (ii) Same-shape bias
        SoftmaxBiasCase {
            a: (0..12).map(|i| i as f64 * 0.2 - 1.0).collect(),
            a_shape: vec![2, 3, 2],
            bias: (0..12).map(|i| -(i as f64) * 0.05).collect(),
            bias_shape: vec![2, 3, 2],
            dim: -1,
            label: "same-shape bias",
        },
        // (iii) Ragged/non-power-of-2 last dim — triggers fallback path on CUDA
        //       (still semantically correct via add+softmax)
        SoftmaxBiasCase {
            a: (0..15).map(|i| i as f64 * 0.1 - 0.7).collect(),
            a_shape: vec![3, 5],
            bias: vec![0.0, -1.0, 0.0, -1.0, 0.0],
            bias_shape: vec![1, 5],
            dim: -1,
            label: "non-power-of-2 dim (5)",
        },
        // (iv) Large sentinel value matching the attention -30000 case, multiple heads
        SoftmaxBiasCase {
            a: (0..64).map(|i| (i % 8) as f64 * 0.5 - 2.0).collect(),
            a_shape: vec![2, 2, 4, 4],
            bias: vec![0.0, 0.0, -30000.0, -30000.0, 0.0, 0.0, -30000.0, -30000.0],
            bias_shape: vec![2, 1, 1, 4],
            dim: -1,
            label: "attention mask -30000 multi-batch",
        },
        // (v) Softmax over non-last dim (always uses fallback)
        SoftmaxBiasCase {
            a: (0..12).map(|i| i as f64 * 0.1).collect(),
            a_shape: vec![3, 4],
            bias: (0..12).map(|i| -(i as f64) * 0.02).collect(),
            bias_shape: vec![3, 4],
            dim: 0,
            label: "non-last dim softmax",
        },
    ]
}

fn test_softmax_with_bias_parity_for_dtype(dtype: DType) {
    if !is_dtype_supported("cpu", dtype) {
        return;
    }

    let (cpu_client, cpu_device) = create_cpu_client();

    for case in softmax_bias_test_cases() {
        let a_cpu =
            tensor_from_f64(&case.a, &case.a_shape, dtype, &cpu_device, &cpu_client).unwrap();
        let bias_cpu = tensor_from_f64(
            &case.bias,
            &case.bias_shape,
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();

        let result_cpu = cpu_client
            .softmax_with_bias(&a_cpu, &bias_cpu, case.dim)
            .unwrap()
            .contiguous()
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_cuda =
                    tensor_from_f64(&case.a, &case.a_shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let bias_cuda = tensor_from_f64(
                    &case.bias,
                    &case.bias_shape,
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let result_cuda = cuda_client
                    .softmax_with_bias(&a_cuda, &bias_cuda, case.dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(
                    &result_cuda,
                    &result_cpu,
                    dtype,
                    &format!("softmax_with_bias cuda vs cpu [{}]", case.label),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_wgpu =
                    tensor_from_f64(&case.a, &case.a_shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let bias_wgpu = tensor_from_f64(
                    &case.bias,
                    &case.bias_shape,
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let result_wgpu = wgpu_client
                    .softmax_with_bias(&a_wgpu, &bias_wgpu, case.dim)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                assert_tensor_allclose(
                    &result_wgpu,
                    &result_cpu,
                    dtype,
                    &format!("softmax_with_bias wgpu vs cpu [{}]", case.label),
                );
            });
        }
    }
}

#[test]
fn test_softmax_with_bias_parity() {
    #[cfg(feature = "f16")]
    {
        test_softmax_with_bias_parity_for_dtype(DType::F16);
        test_softmax_with_bias_parity_for_dtype(DType::BF16);
    }
    test_softmax_with_bias_parity_for_dtype(DType::F32);
    test_softmax_with_bias_parity_for_dtype(DType::F64);
}
