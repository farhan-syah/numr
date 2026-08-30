// Backend parity tests for floor / ceil / round / round_ties_even / trunc on
// I32 and U32.
//
// Every integer is already its own nearest integer, so all five rounding ops
// are the identity on an integer dtype. CPU and CUDA already answer it, so
// WebGPU must too. `assert_parity_i32`/`assert_parity_u32` compare integers
// EXACTLY, so this pins the values, not a tolerance.
//
// Every test is `#[cfg(feature = "wgpu")]`, so the imports are too - otherwise a
// non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use numr::ops::UnaryOps;
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::{
    assert_parity_i32, assert_parity_u32, with_wgpu_backend_or_skip,
};
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

#[cfg(feature = "wgpu")]
#[test]
fn test_rounding_i32_parity() {
    let data = vec![i32::MIN, -1, 0, 1, i32::MAX];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device).expect("cpu tensor");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[5], &device).expect("wgpu tensor");
        for (name, cpu_out, wgpu_out) in [
            (
                "floor",
                cpu_client.floor(&a_cpu).expect("cpu floor i32"),
                client.floor(&a).expect("wgpu floor i32"),
            ),
            (
                "ceil",
                cpu_client.ceil(&a_cpu).expect("cpu ceil i32"),
                client.ceil(&a).expect("wgpu ceil i32"),
            ),
            (
                "round",
                cpu_client.round(&a_cpu).expect("cpu round i32"),
                client.round(&a).expect("wgpu round i32"),
            ),
            (
                "round_ties_even",
                cpu_client
                    .round_ties_even(&a_cpu)
                    .expect("cpu round_ties_even i32"),
                client
                    .round_ties_even(&a)
                    .expect("wgpu round_ties_even i32"),
            ),
            (
                "trunc",
                cpu_client.trunc(&a_cpu).expect("cpu trunc i32"),
                client.trunc(&a).expect("wgpu trunc i32"),
            ),
        ] {
            assert_parity_i32(
                &wgpu_out.to_vec::<i32>(),
                &cpu_out.to_vec::<i32>(),
                &format!("{name} i32"),
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_rounding_u32_parity() {
    let data = vec![0u32, 1, 2, 4_000_000_000, u32::MAX];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[5], &cpu_device).expect("cpu tensor");

    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&data, &[5], &device).expect("wgpu tensor");
        for (name, cpu_out, wgpu_out) in [
            (
                "floor",
                cpu_client.floor(&a_cpu).expect("cpu floor u32"),
                client.floor(&a).expect("wgpu floor u32"),
            ),
            (
                "ceil",
                cpu_client.ceil(&a_cpu).expect("cpu ceil u32"),
                client.ceil(&a).expect("wgpu ceil u32"),
            ),
            (
                "round",
                cpu_client.round(&a_cpu).expect("cpu round u32"),
                client.round(&a).expect("wgpu round u32"),
            ),
            (
                "round_ties_even",
                cpu_client
                    .round_ties_even(&a_cpu)
                    .expect("cpu round_ties_even u32"),
                client
                    .round_ties_even(&a)
                    .expect("wgpu round_ties_even u32"),
            ),
            (
                "trunc",
                cpu_client.trunc(&a_cpu).expect("cpu trunc u32"),
                client.trunc(&a).expect("wgpu trunc u32"),
            ),
        ] {
            assert_parity_u32(
                &wgpu_out.to_vec::<u32>(),
                &cpu_out.to_vec::<u32>(),
                &format!("{name} u32"),
            );
        }
    });
}
