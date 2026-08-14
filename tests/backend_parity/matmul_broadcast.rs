// Backend parity tests for batched matmul with broadcast batch dimensions.
//
// Batch dims broadcast per dimension, so an operand's total batch count does not
// locate its data. With `A[2, 4, m, k] @ B[2, 1, k, n]` the output has 8 batches
// while B has 2, and B's index must advance once every 4 outputs. Collapsing the
// batch dims to a single count cannot express that, and the shapes still line up,
// so a wrong answer is silent.
//
// This is the shape of grouped-query attention, MoE routing, and SSM scans.

use numr::dtype::DType;
use numr::ops::{MatmulOps, SemiringMatmulOps, SemiringOp};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose, create_cpu_client, is_dtype_supported};

/// (a_shape, b_shape) pairs whose batch dims need per-dimension broadcasting.
const CASES: &[(&[usize], &[usize])] = &[
    // A middle batch dim broadcasts while the leading batch is > 1. The leading
    // batch must be > 1 for this to bite: at 1 the operand degenerates to a plain
    // broadcast and the collapsed-count arithmetic happens to be right.
    (&[2, 4, 3, 2], &[2, 1, 2, 1]),
    (&[3, 4, 3, 2], &[3, 1, 2, 1]),
    // Broadcast on the other operand.
    (&[2, 1, 3, 2], &[2, 4, 2, 3]),
    // Leading dim broadcasts instead of the middle one.
    (&[1, 4, 3, 2], &[2, 4, 2, 3]),
    // Operand carries fewer batch dims than the output.
    (&[2, 4, 3, 2], &[4, 2, 3]),
    // Both operands broadcast different dims.
    (&[2, 1, 3, 2], &[1, 4, 2, 3]),
    // Three batch dims, middle one broadcast.
    (&[2, 3, 2, 3, 2], &[2, 1, 2, 2, 4]),
];

fn ramp(n: usize, scale: f64, offset: f64) -> Vec<f64> {
    (0..n).map(|i| (i % 11) as f64 * scale + offset).collect()
}

/// Runs `op` on CPU and every available backend, asserting they agree.
///
/// `matmul`, `matmul_bias` and `semiring_matmul` each reach their own batched
/// kernel launch, so a broadcast fix applied to one says nothing about the others.
macro_rules! broadcast_parity {
    ($label:expr, $a_shape:expr, $b_shape:expr, |$client:ident, $a:ident, $b:ident| $body:expr) => {{
        let a_shape: &[usize] = $a_shape;
        let b_shape: &[usize] = $b_shape;
        let dtype = DType::F32;
        let a_data = ramp(a_shape.iter().product(), 0.5, 1.0);
        let b_data = ramp(b_shape.iter().product(), 0.25, 0.5);
        let label = format!("{} {a_shape:?} @ {b_shape:?}", $label);

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_out = {
            let $client = &cpu_client;
            let $a = tensor_from_f64(&a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
            let $b = tensor_from_f64(&b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
            $body.unwrap_or_else(|e| panic!("CPU {label} failed: {e}"))
        };

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let out = {
                    let $a = tensor_from_f64(&a_data, a_shape, dtype, &device, &client).unwrap();
                    let $b = tensor_from_f64(&b_data, b_shape, dtype, &device, &client).unwrap();
                    let $client = &client;
                    $body.unwrap_or_else(|e| panic!("CUDA {label} failed: {e}"))
                };
                assert_eq!(out.shape(), cpu_out.shape(), "CUDA shape: {label}");
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("CUDA vs CPU: {label}"));
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let out = {
                    let $a = tensor_from_f64(&a_data, a_shape, dtype, &device, &client).unwrap();
                    let $b = tensor_from_f64(&b_data, b_shape, dtype, &device, &client).unwrap();
                    let $client = &client;
                    $body.unwrap_or_else(|e| panic!("WGPU {label} failed: {e}"))
                };
                assert_eq!(out.shape(), cpu_out.shape(), "WGPU shape: {label}");
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("WGPU vs CPU: {label}"));
            });
        }
    }};
}

#[test]
fn test_matmul_broadcast_batch_parity() {
    let dtype = DType::F32;

    for &(a_shape, b_shape) in CASES {
        let a_data = ramp(a_shape.iter().product(), 0.5, 1.0);
        let b_data = ramp(b_shape.iter().product(), 0.25, 0.5);
        let label = format!("matmul {a_shape:?} @ {b_shape:?}");

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_a = tensor_from_f64(&a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_b = tensor_from_f64(&b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_out = cpu_client
            .matmul(&cpu_a, &cpu_b)
            .unwrap_or_else(|e| panic!("CPU {label} failed: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let a = tensor_from_f64(&a_data, a_shape, dtype, &device, &client).unwrap();
                let b = tensor_from_f64(&b_data, b_shape, dtype, &device, &client).unwrap();
                let out = client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {label} failed: {e}"));
                assert_eq!(out.shape(), cpu_out.shape(), "CUDA shape: {label}");
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("CUDA vs CPU: {label}"));
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let a = tensor_from_f64(&a_data, a_shape, dtype, &device, &client).unwrap();
                let b = tensor_from_f64(&b_data, b_shape, dtype, &device, &client).unwrap();
                let out = client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("WGPU {label} failed: {e}"));
                assert_eq!(out.shape(), cpu_out.shape(), "WGPU shape: {label}");
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("WGPU vs CPU: {label}"));
            });
        }
    }
}

/// `matmul_bias` reaches its own batched kernel launch, which took its pointers
/// and its batch counts from different tensors.
#[test]
fn test_matmul_bias_broadcast_batch_parity() {
    use numr::ops::MatmulOps;

    for &(a_shape, b_shape) in CASES {
        let n = b_shape[b_shape.len() - 1];
        broadcast_parity!("matmul_bias", a_shape, b_shape, |client, a, b| {
            let bias_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 - 0.2).collect();
            let bias = tensor_from_f64(&bias_data, &[n], DType::F32, a.device(), client).unwrap();
            client.matmul_bias(&a, &b, &bias)
        });
    }
}

/// `semiring_matmul` reaches a third batched kernel launch with the same shape.
#[test]
fn test_semiring_matmul_broadcast_batch_parity() {
    for &(a_shape, b_shape) in CASES {
        broadcast_parity!("semiring_matmul", a_shape, b_shape, |client, a, b| {
            client.semiring_matmul(&a, &b, SemiringOp::MinPlus)
        });
    }
}
