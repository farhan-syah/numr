// Backend parity tests for reduction output shapes.
//
// Values are covered in reduce.rs; these pin the *shape* contract, which drifted
// per backend: reducing every dimension must give a scalar rather than `[1]`, and
// empty `dims` must mean "every dimension" with `keepdim` still honored. A `[1]`
// where CPU gives `[]` does not fail a value comparison, but it breaks
// broadcasting in autograd, so it needs its own assertions.

use numr::dtype::DType;
use numr::ops::{IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{create_cpu_client, is_dtype_supported};

/// (shape, dims, keepdim) cases where the reduced result is rank-reduced.
const CASES: &[(&[usize], &[usize], bool)] = &[
    // Every dimension reduced: must collapse to a scalar.
    (&[6], &[0], false),
    (&[2, 3], &[0, 1], false),
    (&[2, 3, 4], &[0, 1, 2], false),
    (&[6], &[0], true),
    (&[2, 3], &[0, 1], true),
    // Empty dims means every dimension.
    (&[6], &[], false),
    (&[2, 3], &[], false),
    (&[2, 3], &[], true),
    (&[2, 3, 4], &[], true),
    // Partial reductions, for contrast.
    (&[2, 3], &[0], false),
    (&[2, 3, 4], &[1], false),
    (&[2, 3, 4], &[0, 2], false),
];

fn data_for(shape: &[usize]) -> Vec<f64> {
    let n: usize = shape.iter().product();
    (0..n).map(|i| (i % 7) as f64 + 1.0).collect()
}

#[test]
fn test_reduce_output_shape_parity() {
    let dtype = DType::F32;

    for &(shape, dims, keepdim) in CASES {
        let data = data_for(shape);
        let label = format!("sum(shape={shape:?}, dims={dims:?}, keepdim={keepdim})");

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_t = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor failed: {e}"));
        let cpu_out = cpu_client
            .sum(&cpu_t, dims, keepdim)
            .unwrap_or_else(|e| panic!("CPU {label} failed: {e}"));
        let cpu_shape = cpu_out.shape().to_vec();
        let cpu_vals = cpu_out.contiguous().unwrap().to_vec::<f32>();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let t = tensor_from_f64(&data, shape, dtype, &device, &client).unwrap();
                let out = client
                    .sum(&t, dims, keepdim)
                    .unwrap_or_else(|e| panic!("CUDA {label} failed: {e}"));
                assert_eq!(out.shape(), cpu_shape.as_slice(), "CUDA shape: {label}");
                assert_eq!(
                    out.contiguous().unwrap().to_vec::<f32>(),
                    cpu_vals,
                    "CUDA values: {label}"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let t = tensor_from_f64(&data, shape, dtype, &device, &client).unwrap();
                let out = client
                    .sum(&t, dims, keepdim)
                    .unwrap_or_else(|e| panic!("WGPU {label} failed: {e}"));
                assert_eq!(out.shape(), cpu_shape.as_slice(), "WGPU shape: {label}");
                assert_eq!(
                    out.contiguous().unwrap().to_vec::<f32>(),
                    cpu_vals,
                    "WGPU values: {label}"
                );
            });
        }
    }
}

/// argmax/argmin shared the same hand-rolled shape logic as `sum`.
#[test]
fn test_argreduce_output_shape_parity() {
    let dtype = DType::F32;
    let shape: &[usize] = &[5];
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_t = tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_out = cpu_client.argmax(&cpu_t, 0, false).expect("CPU argmax");
    let cpu_shape = cpu_out.shape().to_vec();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|client, device| {
            let t = tensor_from_f64(&data, shape, dtype, &device, &client).unwrap();
            let out = client.argmax(&t, 0, false).expect("CUDA argmax");
            assert_eq!(
                out.shape(),
                cpu_shape.as_slice(),
                "argmax shape CUDA vs CPU"
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|client, device| {
            let t = tensor_from_f64(&data, shape, dtype, &device, &client).unwrap();
            let out = client.argmax(&t, 0, false).expect("WGPU argmax");
            assert_eq!(
                out.shape(),
                cpu_shape.as_slice(),
                "argmax shape WGPU vs CPU"
            );
        });
    }
}

/// A scalar loss must broadcast back to the input rank during backward. This is
/// the failure the shape divergence actually produced.
#[test]
fn test_scalar_loss_backward_parity() {
    use numr::autograd::var_ops::var_sum;
    use numr::autograd::{Var, backward};

    let dtype = DType::F32;
    let shapes: &[&[usize]] = &[&[6], &[2, 3]];

    for &shape in shapes {
        let data = data_for(shape);
        let dims: Vec<usize> = (0..shape.len()).collect();

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_v = Var::new(
            tensor_from_f64(&data, shape, dtype, &cpu_device, &cpu_client).unwrap(),
            true,
        );
        let cpu_loss = var_sum(&cpu_v, &dims, false, &cpu_client).expect("CPU sum");
        let cpu_grads = backward(&cpu_loss, &cpu_client).expect("CPU backward");
        let cpu_grad = cpu_grads
            .get(cpu_v.id())
            .expect("CPU grad")
            .contiguous()
            .unwrap()
            .to_vec::<f32>();
        assert!(
            cpu_grad.iter().all(|&g| g == 1.0),
            "CPU grad of sum is ones"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let v = Var::new(
                    tensor_from_f64(&data, shape, dtype, &device, &client).unwrap(),
                    true,
                );
                let loss = var_sum(&v, &dims, false, &client).expect("CUDA sum");
                let grads = backward(&loss, &client)
                    .unwrap_or_else(|e| panic!("CUDA backward for {shape:?}: {e}"));
                let g = grads.get(v.id()).expect("CUDA grad");
                assert_eq!(g.contiguous().unwrap().to_vec::<f32>(), cpu_grad);
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let v = Var::new(
                    tensor_from_f64(&data, shape, dtype, &device, &client).unwrap(),
                    true,
                );
                let loss = var_sum(&v, &dims, false, &client).expect("WGPU sum");
                let grads = backward(&loss, &client)
                    .unwrap_or_else(|e| panic!("WGPU backward for {shape:?}: {e}"));
                let g = grads.get(v.id()).expect("WGPU grad");
                assert_eq!(g.contiguous().unwrap().to_vec::<f32>(), cpu_grad);
            });
        }
    }
}
