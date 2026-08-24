//! DType Audit for ML Workloads
//!
//! Tests F16, BF16, FP8E4M3, FP8E5M2 support across ML-critical operations.
//! All helpers are feature-gated so they only compile when the relevant dtype
//! features are enabled.

#[cfg(any(feature = "f16", feature = "fp8"))]
mod common;

#[cfg(any(feature = "f16", feature = "fp8"))]
use common::create_cpu_client;
#[cfg(any(feature = "f16", feature = "fp8"))]
use numr::dtype::DType;
#[cfg(any(feature = "f16", feature = "fp8"))]
use numr::error::Result;
#[cfg(any(feature = "f16", feature = "fp8"))]
use numr::ops::*;
#[cfg(any(feature = "f16", feature = "fp8"))]
use numr::runtime::cpu::CpuRuntime;
#[cfg(any(feature = "f16", feature = "fp8"))]
use numr::tensor::Tensor;

#[cfg(any(feature = "f16", feature = "fp8"))]
fn make_tensor(
    data: &[f32],
    shape: &[usize],
    dtype: DType,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    client: &impl TypeConversionOps<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let t = Tensor::try_from_slice(data, shape, device).unwrap();
    if dtype == DType::F32 {
        Ok(t)
    } else {
        client.cast(&t, dtype)
    }
}

#[cfg(any(feature = "f16", feature = "fp8"))]
macro_rules! audit_op {
    ($name:expr, $body:expr) => {{
        let result: Result<()> = (|| {
            $body;
            Ok(())
        })();
        match &result {
            Ok(()) => println!("  PASS: {}", $name),
            Err(e) => println!("  FAIL: {} - {}", $name, e),
        }
        result.is_ok()
    }};
}

#[cfg(any(feature = "f16", feature = "fp8"))]
fn audit_dtype(dtype: DType) {
    println!("\n=== Auditing {:?} ===", dtype);
    let (client, device) = create_cpu_client();
    let mut pass = 0u32;
    let mut fail = 0u32;

    macro_rules! tally {
        ($ok:expr) => {
            if $ok {
                pass += 1;
            } else {
                fail += 1;
            }
        };
    }

    // Cast F32 -> target
    let cast_ok = audit_op!("cast F32 -> target", {
        let t =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let _ = client.cast(&t, dtype)?;
    });
    tally!(cast_ok);

    if !cast_ok {
        println!("  SKIP remaining (cast failed)");
        println!("\n  Summary for {:?}: {} pass, {} fail", dtype, pass, fail);
        return;
    }

    let t1 = |d: &[f32], s: &[usize]| make_tensor(d, s, dtype, &device, &client);

    // Binary ops
    tally!(audit_op!("add", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let b = t1(&[5.0, 6.0, 7.0, 8.0], &[4])?;
        let _ = client.add(&a, &b)?;
    }));
    tally!(audit_op!("sub", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let b = t1(&[5.0, 6.0, 7.0, 8.0], &[4])?;
        let _ = client.sub(&a, &b)?;
    }));
    tally!(audit_op!("mul", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let b = t1(&[5.0, 6.0, 7.0, 8.0], &[4])?;
        let _ = client.mul(&a, &b)?;
    }));
    tally!(audit_op!("div", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let b = t1(&[5.0, 6.0, 7.0, 8.0], &[4])?;
        let _ = client.div(&a, &b)?;
    }));

    // Scalar ops
    tally!(audit_op!("mul_scalar", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let _ = client.mul_scalar(&a, 2.0)?;
    }));
    tally!(audit_op!("add_scalar", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let _ = client.add_scalar(&a, 1.0)?;
    }));

    // Unary ops
    tally!(audit_op!("exp", {
        let a = t1(&[0.0, 0.5, 1.0, 1.5], &[4])?;
        let _ = client.exp(&a)?;
    }));
    tally!(audit_op!("log", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let _ = client.log(&a)?;
    }));
    tally!(audit_op!("sqrt", {
        let a = t1(&[1.0, 4.0, 9.0, 16.0], &[4])?;
        let _ = client.sqrt(&a)?;
    }));
    tally!(audit_op!("tanh", {
        let a = t1(&[0.0, 0.5, 1.0, -1.0], &[4])?;
        let _ = client.tanh(&a)?;
    }));
    tally!(audit_op!("neg", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let _ = client.neg(&a)?;
    }));

    // Reduce ops (dims are usize, use last dim = 1 for [2,2])
    tally!(audit_op!("sum", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let _ = client.sum(&a, &[1], false)?;
    }));
    tally!(audit_op!("max", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let _ = client.max(&a, &[1], false)?;
    }));
    tally!(audit_op!("mean", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let _ = client.mean(&a, &[1], false)?;
    }));
    tally!(audit_op!("argmax", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let _ = client.argmax(&a, 1, false)?;
    }));

    // Matmul (disambiguate)
    tally!(audit_op!("matmul", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = t1(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let _ = MatmulOps::matmul(&client, &a, &b)?;
    }));

    // Activation ops
    tally!(audit_op!("softmax", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let _ = client.softmax(&a, -1)?;
    }));
    tally!(audit_op!("relu", {
        let a = t1(&[-1.0, 0.0, 1.0, 2.0], &[4])?;
        let _ = client.relu(&a)?;
    }));
    tally!(audit_op!("gelu", {
        let a = t1(&[-1.0, 0.0, 1.0, 2.0], &[4])?;
        let _ = client.gelu(&a)?;
    }));
    tally!(audit_op!("silu", {
        let a = t1(&[-1.0, 0.0, 1.0, 2.0], &[4])?;
        let _ = client.silu(&a)?;
    }));

    // Normalization ops (require weight/bias tensors)
    tally!(audit_op!("rms_norm", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let w = t1(&[1.0, 1.0, 1.0], &[3])?;
        let _ = client.rms_norm(&a, &w, 1e-5)?;
    }));
    tally!(audit_op!("layer_norm", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let w = t1(&[1.0, 1.0, 1.0], &[3])?;
        let b = t1(&[0.0, 0.0, 0.0], &[3])?;
        let _ = client.layer_norm(&a, &w, &b, 1e-5)?;
    }));

    // Cast back
    tally!(audit_op!("cast target -> F32", {
        let a = t1(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let _ = client.cast(&a, DType::F32)?;
    }));

    println!("\n  Summary for {:?}: {} pass, {} fail", dtype, pass, fail);
    if fail > 0 {
        panic!("{:?} has {} failures", dtype, fail);
    }
}

#[test]
#[cfg(feature = "f16")]
fn audit_f16() {
    audit_dtype(DType::F16);
}

#[test]
#[cfg(feature = "f16")]
fn audit_bf16() {
    audit_dtype(DType::BF16);
}

#[test]
#[cfg(feature = "fp8")]
fn audit_fp8e4m3() {
    audit_dtype(DType::FP8E4M3);
}

#[test]
#[cfg(feature = "fp8")]
fn audit_fp8e5m2() {
    audit_dtype(DType::FP8E5M2);
}
