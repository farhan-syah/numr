use numr::autograd::{
    Var, backward, var_abs, var_add_scalar, var_cast, var_cos, var_exp, var_log, var_mul_scalar,
    var_neg, var_recip, var_sin, var_sqrt, var_square, var_sum, var_tan, var_tanh,
};
use numr::error::Result;
use numr::prelude::{CpuDevice, CpuRuntime, DType, Runtime, Tensor};

const INPUTS: [f64; 3] = [0.25, 0.75, 1.25];
const SCALE: f64 = 0.4;
const SHIFT: f64 = 0.2;
const EPS: f64 = 1e-6;
const TOL: f64 = 1e-6;

fn scalar_loss<F>(inputs: &[f64], op: F) -> f64
where
    F: Fn(f64) -> f64 + Copy,
{
    inputs.iter().map(|x| op(SCALE * x + SHIFT)).sum()
}

fn finite_differences<F>(inputs: &[f64], op: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Copy,
{
    (0..inputs.len())
        .map(|idx| {
            let mut plus = inputs.to_vec();
            plus[idx] += EPS;
            let mut minus = inputs.to_vec();
            minus[idx] -= EPS;
            (scalar_loss(&plus, op) - scalar_loss(&minus, op)) / (2.0 * EPS)
        })
        .collect()
}

fn assert_vectors_close(test_name: &str, autograd: &[f64], finite_diff: &[f64], tolerance: f64) {
    eprintln!("{test_name}: autograd={autograd:?}, finite_diff={finite_diff:?}");

    assert_eq!(autograd.len(), finite_diff.len());
    for (idx, (actual, expected)) in autograd.iter().zip(finite_diff).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{test_name} gradient mismatch at index {idx}: autograd={actual}, finite_diff={expected}, diff={diff}, tolerance={tolerance}"
        );
    }
}

macro_rules! unary_reaches_leaf_after_upstream_op_test {
    ($test_name:ident, $var_op:ident, $scalar_op:expr) => {
        #[test]
        fn $test_name() -> Result<()> {
            let device = CpuDevice::new();
            let client = CpuRuntime::default_client(&device);
            let x = Var::new(
                Tensor::<CpuRuntime>::from_slice(&INPUTS, &[INPUTS.len()], &device),
                true,
            );

            let scaled = var_mul_scalar(&x, SCALE, &client)?;
            let shifted = var_add_scalar(&scaled, SHIFT, &client)?;
            let y = $var_op(&shifted, &client)?;
            let loss = var_sum(&y, &[], false, &client)?;
            let grads = backward(&loss, &client)?;
            let grad = match grads.get(x.id()) {
                Some(grad) => grad,
                None => panic!(
                    "{} did not propagate a gradient to the original leaf",
                    stringify!($test_name)
                ),
            };

            let autograd = grad.to_vec::<f64>();
            let finite_diff = finite_differences(&INPUTS, $scalar_op);
            assert_vectors_close(stringify!($test_name), &autograd, &finite_diff, TOL);
            Ok(())
        }
    };
}

unary_reaches_leaf_after_upstream_op_test!(
    neg_backward_reaches_leaf_after_upstream_op,
    var_neg,
    |x| -x
);
unary_reaches_leaf_after_upstream_op_test!(
    exp_backward_reaches_leaf_after_upstream_op,
    var_exp,
    f64::exp
);
unary_reaches_leaf_after_upstream_op_test!(
    log_backward_reaches_leaf_after_upstream_op,
    var_log,
    f64::ln
);
unary_reaches_leaf_after_upstream_op_test!(
    sqrt_backward_reaches_leaf_after_upstream_op,
    var_sqrt,
    f64::sqrt
);
unary_reaches_leaf_after_upstream_op_test!(
    tanh_backward_reaches_leaf_after_upstream_op,
    var_tanh,
    f64::tanh
);
unary_reaches_leaf_after_upstream_op_test!(
    abs_backward_reaches_leaf_after_upstream_op,
    var_abs,
    f64::abs
);
unary_reaches_leaf_after_upstream_op_test!(
    sin_backward_reaches_leaf_after_upstream_op,
    var_sin,
    f64::sin
);
unary_reaches_leaf_after_upstream_op_test!(
    cos_backward_reaches_leaf_after_upstream_op,
    var_cos,
    f64::cos
);
unary_reaches_leaf_after_upstream_op_test!(
    tan_backward_reaches_leaf_after_upstream_op,
    var_tan,
    f64::tan
);
unary_reaches_leaf_after_upstream_op_test!(
    recip_backward_reaches_leaf_after_upstream_op,
    var_recip,
    |x| 1.0 / x
);
unary_reaches_leaf_after_upstream_op_test!(
    square_backward_reaches_leaf_after_upstream_op,
    var_square,
    |x| x * x
);

#[test]
fn cast_backward_reaches_leaf_after_upstream_op() -> Result<()> {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let inputs = INPUTS.map(|x| x as f32);
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&inputs, &[inputs.len()], &device),
        true,
    );

    let scaled = var_mul_scalar(&x, SCALE, &client)?;
    let shifted = var_add_scalar(&scaled, SHIFT, &client)?;
    let y = var_cast(&shifted, DType::F64, &client)?;
    assert_eq!(y.tensor().dtype(), DType::F64);
    let loss = var_sum(&y, &[], false, &client)?;
    let grads = backward(&loss, &client)?;
    let grad = match grads.get(x.id()) {
        Some(grad) => grad,
        None => panic!("cast did not propagate a gradient to the original leaf"),
    };

    assert_eq!(grad.dtype(), DType::F32);
    let autograd: Vec<f64> = grad.to_vec::<f32>().into_iter().map(f64::from).collect();
    let finite_diff = finite_differences(&INPUTS, |x| x);
    assert_vectors_close(
        "cast_backward_reaches_leaf_after_upstream_op",
        &autograd,
        &finite_diff,
        1e-5,
    );
    Ok(())
}
