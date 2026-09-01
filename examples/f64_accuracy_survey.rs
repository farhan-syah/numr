//! Survey f64 accuracy of every CPU SIMD transcendental against `std`.
//!
//! Tensors are sized past the SIMD threshold; short tensors take the scalar
//! tail, which is exact and hides a polynomial that is too low-degree for f64.

use numr::ops::UnaryOps;
use numr::prelude::*;

type Kernel = fn(&CpuClient, &Tensor<CpuRuntime>) -> numr::error::Result<Tensor<CpuRuntime>>;
/// Name, kernel under test, reference, and the closed interval to sweep.
type Case = (&'static str, Kernel, fn(f64) -> f64, f64, f64);

/// `f64::atanh` is the one function here Rust std does not delegate to libm:
/// it is `0.5 * ((2x)/(1-x)).ln_1p()`, which is not odd-symmetric and loses
/// accuracy without bound as `x -> -1`. Evaluating it on the positive side and
/// negating gives the accurate branch for both signs, so the survey measures
/// the kernel rather than the reference.
fn atanh_reference(x: f64) -> f64 {
    if x < 0.0 { -(-x).atanh() } else { x.atanh() }
}

fn main() {
    let device = CpuDevice::default();
    let client = CpuRuntime::default_client(&device);

    let cases: Vec<Case> = vec![
        ("sqrt", |c, t| c.sqrt(t), f64::sqrt, 1e-3, 100.0),
        ("cbrt", |c, t| c.cbrt(t), f64::cbrt, -100.0, 100.0),
        ("recip", |c, t| c.recip(t), |x| 1.0 / x, 0.1, 100.0),
        ("exp", |c, t| c.exp(t), f64::exp, -700.0, 700.0),
        ("exp2", |c, t| c.exp2(t), f64::exp2, -1000.0, 1000.0),
        ("expm1", |c, t| c.expm1(t), f64::exp_m1, -2.0, 2.0),
        ("log", |c, t| c.log(t), f64::ln, 1e-3, 1e3),
        ("log2", |c, t| c.log2(t), f64::log2, 1e-3, 1e3),
        ("log10", |c, t| c.log10(t), f64::log10, 1e-3, 1e3),
        ("log1p", |c, t| c.log1p(t), f64::ln_1p, -0.9, 10.0),
        ("sin", |c, t| c.sin(t), f64::sin, -20.0, 20.0),
        ("cos", |c, t| c.cos(t), f64::cos, -20.0, 20.0),
        ("tan", |c, t| c.tan(t), f64::tan, -1.4, 1.4),
        ("asin", |c, t| c.asin(t), f64::asin, -0.99, 0.99),
        ("acos", |c, t| c.acos(t), f64::acos, -0.99, 0.99),
        ("atan", |c, t| c.atan(t), f64::atan, -50.0, 50.0),
        ("sinh", |c, t| c.sinh(t), f64::sinh, -10.0, 10.0),
        ("cosh", |c, t| c.cosh(t), f64::cosh, -10.0, 10.0),
        ("tanh", |c, t| c.tanh(t), f64::tanh, -10.0, 10.0),
        ("asinh", |c, t| c.asinh(t), f64::asinh, -50.0, 50.0),
        ("acosh", |c, t| c.acosh(t), f64::acosh, 1.01, 50.0),
        ("atanh", |c, t| c.atanh(t), atanh_reference, -0.99, 0.99),
    ];

    const N: usize = 4096;
    println!("{:>7}  {:>11}  {:>9}  verdict", "op", "worst rel", "ulps");
    let mut broken = Vec::new();
    for (name, kernel, reference, lo, hi) in cases {
        let args: Vec<f64> = (0..N)
            .map(|i| lo + (hi - lo) * (i as f64) / (N as f64 - 1.0))
            .collect();
        let t = match Tensor::<CpuRuntime>::from_slice(&args, &[N], &device) {
            Ok(t) => t,
            Err(e) => {
                println!("{name:>7}  tensor build failed: {e}");
                continue;
            }
        };
        let got = match kernel(&client, &t) {
            Ok(r) => r.to_vec::<f64>(),
            Err(e) => {
                println!("{name:>7}  unsupported: {e}");
                continue;
            }
        };
        let mut worst = 0.0f64;
        let mut at = 0.0f64;
        let mut got_at = 0.0f64;
        let mut want_at = 0.0f64;
        for (i, &x) in args.iter().enumerate() {
            let want = reference(x);
            if want.is_finite() && want.abs() > 1e-300 && got[i].is_finite() {
                let rel = ((got[i] - want) / want).abs();
                if rel > worst {
                    worst = rel;
                    at = x;
                    got_at = got[i];
                    want_at = want;
                }
            }
        }
        let ulps = worst / f64::EPSILON;
        let verdict = if ulps > 64.0 {
            broken.push(name);
            "BROKEN"
        } else if ulps > 4.0 {
            "marginal"
        } else {
            "ok"
        };
        println!(
            "{name:>7}  {worst:>11.3e}  {ulps:>9.1}  {verdict:<9} at x={at:.4} got={got_at:.10e} want={want_at:.10e}"
        );
    }
    println!("\nf64 eps = {:.3e}", f64::EPSILON);
    println!("broken ({}): {}", broken.len(), broken.join(" "));
}
