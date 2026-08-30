//! CUDA scalar, `where` and fused-elementwise kernels must cover every dtype
//! the CPU backend covers, with the same numerical semantics.
//!
//! The CUDA path looks its kernel up by NAME — `{op}_scalar_{suffix}`,
//! `where_{suffix}`, `fused_mul_add_{suffix}` — so a dtype with no `.cu`
//! instantiation compiles fine and fails at launch with `named symbol not
//! found`. U32 and the narrow integers had no instantiation in any of the
//! three families, and `fused_elementwise.cu` had no integer row at all.
//!
//! Semantics come from the CPU reference (`src/runtime/cpu/kernels/scalar.rs`,
//! `binary_int.rs` and `fused_elementwise.rs`):
//!
//! * add, sub, rsub and mul WRAP on overflow.
//! * div by a zero scalar yields 0.
//! * pow is exact and saturating, and `pow_scalar_output_dtype` decides whether
//!   the output stays integral at all.
//! * a fused op answers exactly what the unfused sequence answers, wrapping at
//!   every step.
//!
//! Run: cargo test --features cuda --test cuda_scalar_ternary_fused_dtype_coverage

#![cfg(feature = "cuda")]

mod common;

use common::backend_lock::with_cuda_backend;
use common::create_cpu_client;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, ScalarOps, TypeConversionOps};
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// The operands the U32 scalar tests share.
///
/// Index 2 overflows under `mul`, index 4 goes below zero under `sub`, and
/// every element goes below zero under `rsub` by 3 except the last.
const U32_A: [u32; 5] = [10, 5, 4_000_000_000, 7, 1];

/// Run one tensor-scalar operation on CPU and on CUDA, asserting both equal
/// `expected`.
///
/// The CPU assertion pins the reference semantics; the CUDA assertion pins
/// parity with it. A CUDA-only check would pass a kernel that agrees with a
/// wrong expectation.
macro_rules! check_scalar {
    ($ty:ty, $method:ident, $a:expr, $scalar:expr, $expected:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));
        let shape = [$a.len()];

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_a = Tensor::<CpuRuntime>::from_slice($a, &shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU input failed: {e:?}"));
        let cpu_out = cpu_client
            .$method(&cpu_a, $scalar)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), $expected, "{label}: CPU reference");

        with_cuda_backend(|cuda_client, cuda_device| {
            let a = Tensor::<CudaRuntime>::from_slice($a, &shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
            let out = cuda_client
                .$method(&a, $scalar)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), $expected, "{label}: CUDA vs CPU");
        });
    }};
}

// ============================================================================
// U32 tensor-scalar arithmetic
// ============================================================================

#[test]
fn u32_add_scalar_matches_cpu() {
    check_scalar!(u32, add_scalar, &U32_A, 3.0, &[13, 8, 4_000_000_003, 10, 4]);
}

/// `1 - 3` wraps to `u32::MAX - 1`. A saturating kernel would answer 0.
#[test]
fn u32_sub_scalar_wraps_below_zero() {
    check_scalar!(
        u32,
        sub_scalar,
        &U32_A,
        3.0,
        &[7, 2, 3_999_999_997, 4, 4_294_967_294]
    );
}

/// `rsub` puts the scalar on the LEFT, so four of the five elements wrap.
#[test]
fn u32_rsub_scalar_wraps_below_zero() {
    check_scalar!(
        u32,
        rsub_scalar,
        &U32_A,
        3.0,
        &[4_294_967_289, 4_294_967_294, 294_967_299, 4_294_967_292, 2]
    );
}

/// `4_000_000_000 * 3` is 1.2e10, which wraps by 2^32 twice.
#[test]
fn u32_mul_scalar_wraps_past_u32_max() {
    check_scalar!(
        u32,
        mul_scalar,
        &U32_A,
        3.0,
        &[30, 15, 3_410_065_408, 21, 3]
    );
}

#[test]
fn u32_div_scalar_matches_cpu() {
    check_scalar!(u32, div_scalar, &U32_A, 3.0, &[3, 1, 1_333_333_333, 2, 0]);
}

/// A zero divisor yields 0 rather than trapping, matching `binary_int.rs`.
#[test]
fn u32_div_scalar_by_zero_yields_zero() {
    check_scalar!(u32, div_scalar, &U32_A, 0.0, &[0, 0, 0, 0, 0]);
}

/// pow SATURATES where the other arithmetic wraps: its result is an
/// accumulator. `4_000_000_000 ** 2` is 1.6e19, far past U32.
#[test]
fn u32_pow_scalar_saturates_on_overflow() {
    check_scalar!(u32, pow_scalar, &U32_A, 2.0, &[100, 25, u32::MAX, 49, 1]);
}

// ============================================================================
// pow_scalar output dtype
// ============================================================================

/// `pow_scalar_output_dtype` is the single authority on whether an integer
/// input keeps its dtype. Both backends must land on what it says, and CUDA
/// must produce the value the promoted dtype implies.
#[test]
fn pow_scalar_output_dtype_governs_both_backends() {
    let (cpu_client, cpu_device) = create_cpu_client();

    let ints: [i64; 4] = [4, 9, 1, 2];

    // A non-negative whole exponent keeps the integer dtype.
    // A whole, non-negative exponent keeps the integer dtype.
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&ints, &[4], &cpu_device)
        .expect("staging the CPU input must succeed");
    let cpu_cube = cpu_client
        .pow_scalar(&cpu_a, 3.0)
        .expect("the CPU pow_scalar must succeed");
    assert_eq!(cpu_cube.dtype(), DType::I64);
    assert_eq!(cpu_cube.to_vec::<i64>(), vec![64, 729, 1, 8]);

    // A fractional exponent has no integer result, so the output promotes.
    // A fractional exponent leaves the integers, so the result promotes to F64.
    let cpu_root = cpu_client
        .pow_scalar(&cpu_a, 0.5)
        .expect("the CPU pow_scalar must succeed");
    assert_eq!(cpu_root.dtype(), DType::F64);
    assert_eq!(cpu_root.to_vec::<f64>()[0], 2.0);

    with_cuda_backend(|client, device| {
        let a = Tensor::<CudaRuntime>::from_slice(&ints, &[4], &device)
            .expect("staging the CUDA input must succeed");

        let cube = client
            .pow_scalar(&a, 3.0)
            .expect("the CUDA pow_scalar must succeed");
        assert_eq!(cube.dtype(), DType::I64);
        assert_eq!(cube.to_vec::<i64>(), vec![64, 729, 1, 8]);

        let root = client
            .pow_scalar(&a, 0.5)
            .expect("the CUDA pow_scalar must succeed");
        assert_eq!(root.dtype(), DType::F64);
        assert_eq!(root.to_vec::<f64>()[0], 2.0);
    });
}

// ============================================================================
// where on U32
// ============================================================================

/// `where_u32` had no instantiation, so this failed at kernel lookup.
#[test]
fn u32_where_matches_cpu() {
    let cond: [u8; 4] = [1, 0, 1, 0];
    let x: [u32; 4] = [10, 20, 30, 40];
    let y: [u32; 4] = [100, 200, 300, 400];
    let expected: [u32; 4] = [10, 200, 30, 400];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_cond = Tensor::<CpuRuntime>::from_slice(&cond, &[4], &cpu_device)
        .expect("staging the CPU condition must succeed");
    let cpu_x = Tensor::<CpuRuntime>::from_slice(&x, &[4], &cpu_device)
        .expect("staging the CPU lhs must succeed");
    let cpu_y = Tensor::<CpuRuntime>::from_slice(&y, &[4], &cpu_device)
        .expect("staging the CPU rhs must succeed");
    let cpu_out = cpu_client
        .where_cond(&cpu_cond, &cpu_x, &cpu_y)
        .expect("the CPU where_cond must succeed");
    assert_eq!(cpu_out.to_vec::<u32>(), expected.to_vec());

    with_cuda_backend(|client, device| {
        let cond = Tensor::<CudaRuntime>::from_slice(&cond, &[4], &device)
            .expect("staging the CUDA condition must succeed");
        let x = Tensor::<CudaRuntime>::from_slice(&x, &[4], &device)
            .expect("staging the CUDA lhs must succeed");
        let y = Tensor::<CudaRuntime>::from_slice(&y, &[4], &device)
            .expect("staging the CUDA rhs must succeed");
        let out = client
            .where_cond(&cond, &x, &y)
            .expect("the CUDA where_cond must succeed");
        assert_eq!(out.to_vec::<u32>(), expected.to_vec());
    });
}

// ============================================================================
// Fused elementwise vs the unfused sequence
// ============================================================================

/// Run the three fused ops against one backend, asserting each equals both an
/// explicit literal and the unfused sequence it stands for.
///
/// Separate from [`check_fused`] because a `macro_rules!` cannot define another
/// one inside its own body: the inner `$` metavariables would be read by the
/// outer expansion.
macro_rules! check_fused_on {
    (
        $rt:ty, $client:expr, $device:expr, $tag:expr, $ty:ty,
        $a:expr, $b:expr, $c:expr,
        $mul_add:expr, $add_mul:expr, $mul_add_scalar:expr
    ) => {{
        let tag = $tag;
        let client = $client;
        let shape = [$a.len()];
        let (scale, bias) = (2.0f64, 1.0f64);

        let a = Tensor::<$rt>::from_slice($a, &shape, $device)
            .unwrap_or_else(|e| panic!("{tag}: staging a failed: {e:?}"));
        let b = Tensor::<$rt>::from_slice($b, &shape, $device)
            .unwrap_or_else(|e| panic!("{tag}: staging b failed: {e:?}"));
        let c = Tensor::<$rt>::from_slice($c, &shape, $device)
            .unwrap_or_else(|e| panic!("{tag}: staging c failed: {e:?}"));

        let fused = client
            .fused_mul_add(&a, &b, &c)
            .unwrap_or_else(|e| panic!("{tag}: fused_mul_add failed: {e:?}"));
        let unfused = client
            .mul(&a, &b)
            .and_then(|m| client.add(&m, &c))
            .unwrap_or_else(|e| panic!("{tag}: mul then add failed: {e:?}"));
        assert_eq!(
            fused.to_vec::<$ty>(),
            $mul_add.to_vec(),
            "{tag}: fused_mul_add"
        );
        assert_eq!(
            fused.to_vec::<$ty>(),
            unfused.to_vec::<$ty>(),
            "{tag}: fused_mul_add vs mul then add"
        );

        let fused = client
            .fused_add_mul(&a, &b, &c)
            .unwrap_or_else(|e| panic!("{tag}: fused_add_mul failed: {e:?}"));
        let unfused = client
            .add(&a, &b)
            .and_then(|s| client.mul(&s, &c))
            .unwrap_or_else(|e| panic!("{tag}: add then mul failed: {e:?}"));
        assert_eq!(
            fused.to_vec::<$ty>(),
            $add_mul.to_vec(),
            "{tag}: fused_add_mul"
        );
        assert_eq!(
            fused.to_vec::<$ty>(),
            unfused.to_vec::<$ty>(),
            "{tag}: fused_add_mul vs add then mul"
        );

        let fused = client
            .fused_mul_add_scalar(&a, scale, bias)
            .unwrap_or_else(|e| panic!("{tag}: fused_mul_add_scalar failed: {e:?}"));
        let unfused = client
            .mul_scalar(&a, scale)
            .and_then(|m| client.add_scalar(&m, bias))
            .unwrap_or_else(|e| panic!("{tag}: mul_scalar then add_scalar failed: {e:?}"));
        assert_eq!(
            fused.to_vec::<$ty>(),
            $mul_add_scalar.to_vec(),
            "{tag}: fused_mul_add_scalar"
        );
        assert_eq!(
            fused.to_vec::<$ty>(),
            unfused.to_vec::<$ty>(),
            "{tag}: fused_mul_add_scalar vs the unfused pair"
        );
    }};
}

/// The CPU assertions pin the reference semantics; the CUDA ones pin parity
/// with it.
macro_rules! check_fused {
    (
        $ty:ty, $a:expr, $b:expr, $c:expr,
        $mul_add:expr, $add_mul:expr, $mul_add_scalar:expr
    ) => {{
        let (cpu_client, cpu_device) = create_cpu_client();
        check_fused_on!(
            CpuRuntime,
            &cpu_client,
            &cpu_device,
            concat!("fused ", stringify!($ty), " cpu"),
            $ty,
            $a,
            $b,
            $c,
            $mul_add,
            $add_mul,
            $mul_add_scalar
        );

        with_cuda_backend(|cuda_client, cuda_device| {
            check_fused_on!(
                CudaRuntime,
                &cuda_client,
                &cuda_device,
                concat!("fused ", stringify!($ty), " cuda"),
                $ty,
                $a,
                $b,
                $c,
                $mul_add,
                $add_mul,
                $mul_add_scalar
            );
        });
    }};
}

/// Index 0 overflows at the first step of every one of the three ops, so a
/// kernel that saturates instead of wrapping fails here.
#[test]
fn i32_fused_ops_match_the_unfused_sequence() {
    check_fused!(
        i32,
        &[i32::MAX, 2, -3],
        &[2i32, 3, 4],
        &[1i32, 5, 6],
        [-1i32, 11, -6],
        [-2_147_483_647i32, 25, 6],
        [-1i32, 5, -5]
    );
}

#[test]
fn u32_fused_ops_match_the_unfused_sequence() {
    check_fused!(
        u32,
        &[u32::MAX, 2, 4_000_000_000],
        &[2u32, 3, 4],
        &[1u32, 5, 6],
        [4_294_967_295u32, 11, 3_115_098_118],
        [1u32, 25, 2_525_163_544],
        [4_294_967_295u32, 5, 3_705_032_705]
    );
}

// ============================================================================
// Whole-matrix kernel resolution
// ============================================================================

/// Every dtype the scalar, ternary and fused instantiation matrices claim to
/// cover. F16, BF16 and the FP8 types are feature-gated, so they are exercised
/// by the backend-parity suite rather than here.
const COVERED_DTYPES: [DType; 10] = [
    DType::F32,
    DType::F64,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
];

/// A missing instantiation fails here at module lookup, not months later in a
/// caller. The scalar is small and non-zero so the check is about kernel
/// resolution, not about values.
#[test]
fn every_dtype_resolves_a_kernel_for_every_op() {
    with_cuda_backend(|client, device| {
        let base = Tensor::<CudaRuntime>::from_slice(&[4.0f64, 3.0, 2.0, 1.0], &[4], &device)
            .expect("staging the base tensor must succeed");
        let base_cond = Tensor::<CudaRuntime>::from_slice(&[1u8, 0, 1, 0], &[4], &device)
            .expect("staging the condition must succeed");

        for &dtype in COVERED_DTYPES.iter() {
            let a = client
                .cast(&base, dtype)
                .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));

            for (name, out) in [
                ("add_scalar", client.add_scalar(&a, 2.0)),
                ("sub_scalar", client.sub_scalar(&a, 2.0)),
                ("rsub_scalar", client.rsub_scalar(&a, 2.0)),
                ("mul_scalar", client.mul_scalar(&a, 2.0)),
                ("div_scalar", client.div_scalar(&a, 2.0)),
                ("pow_scalar", client.pow_scalar(&a, 2.0)),
                (
                    "fused_mul_add_scalar",
                    client.fused_mul_add_scalar(&a, 2.0, 1.0),
                ),
                ("where_cond", client.where_cond(&base_cond, &a, &a)),
                ("fused_mul_add", client.fused_mul_add(&a, &a, &a)),
                ("fused_add_mul", client.fused_add_mul(&a, &a, &a)),
            ] {
                out.unwrap_or_else(|e| panic!("{name} on {dtype:?} failed: {e:?}"));
            }
        }
        client.synchronize();
    });
}
