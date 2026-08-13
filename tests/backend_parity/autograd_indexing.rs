// Backend parity tests for autograd indexing backward passes.
//
// The forward ops are covered in indexing_advanced.rs; these check that the
// gradients agree across backends too.

use numr::autograd::var_ops::{var_embedding_lookup, var_sum};
use numr::autograd::{Var, backward};
use numr::dtype::DType;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{assert_tensor_allclose, create_cpu_client, is_dtype_supported};

/// Weight rows, flattened. Row 3 is never indexed, so its gradient stays zero.
fn weight_data() -> Vec<f64> {
    vec![
        0.5, -1.0, 2.0, 3.5, 4.0, -2.5, 1.25, 0.75, -0.5, 2.25, 5.0, -3.0,
    ]
}

/// Token IDs with a repeat, so the backward path exercises accumulation.
const TOKEN_IDS: [i32; 4] = [0, 2, 1, 0];

/// The backward scatters gradient rows into a `[4, 3]` zero tensor, which is the
/// `inner_size > 1` scatter case rather than a flat one.
#[test]
fn test_var_embedding_lookup_backward_parity() {
    let dtype = DType::F32;
    let data = weight_data();

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_weight = Var::new(
        tensor_from_f64(&data, &[4, 3], dtype, &cpu_device, &cpu_client)
            .expect("CPU tensor_from_f64 failed"),
        true,
    );
    let cpu_idx = Tensor::from_slice(&TOKEN_IDS, &[4], &cpu_device);
    let cpu_out =
        var_embedding_lookup(&cpu_weight, &cpu_idx, &cpu_client).expect("CPU forward failed");
    let cpu_loss = var_sum(&cpu_out, &[0, 1], false, &cpu_client).expect("CPU sum failed");
    let cpu_grads = backward(&cpu_loss, &cpu_client).expect("CPU backward failed");
    let cpu_grad = cpu_grads
        .get(cpu_weight.id())
        .expect("CPU weight gradient missing")
        .contiguous()
        .expect("CPU contiguous failed");

    // Row 0 is indexed twice, rows 1 and 2 once, row 3 never.
    assert_eq!(
        cpu_grad.to_vec::<f32>(),
        vec![2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    );

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|client, device| {
            let weight = Var::new(
                tensor_from_f64(&data, &[4, 3], dtype, &device, &client).unwrap(),
                true,
            );
            let idx = Tensor::from_slice(&TOKEN_IDS, &[4], &device);
            let out = var_embedding_lookup(&weight, &idx, &client).expect("CUDA forward failed");
            let loss = var_sum(&out, &[0, 1], false, &client).expect("CUDA sum failed");
            let grads = backward(&loss, &client).expect("CUDA backward failed");
            let grad = grads
                .get(weight.id())
                .expect("CUDA weight gradient missing")
                .contiguous()
                .expect("CUDA contiguous failed");
            assert_tensor_allclose(
                &grad,
                &cpu_grad,
                dtype,
                "var_embedding_lookup backward CUDA vs CPU",
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|client, device| {
            let weight = Var::new(
                tensor_from_f64(&data, &[4, 3], dtype, &device, &client).unwrap(),
                true,
            );
            let idx = Tensor::from_slice(&TOKEN_IDS, &[4], &device);
            let out = var_embedding_lookup(&weight, &idx, &client).expect("WGPU forward failed");
            let loss = var_sum(&out, &[0, 1], false, &client).expect("WGPU sum failed");
            let grads = backward(&loss, &client).expect("WGPU backward failed");
            let grad = grads
                .get(weight.id())
                .expect("WGPU weight gradient missing")
                .contiguous()
                .expect("WGPU contiguous failed");
            assert_tensor_allclose(
                &grad,
                &cpu_grad,
                dtype,
                "var_embedding_lookup backward WGPU vs CPU",
            );
        });
    }
}
