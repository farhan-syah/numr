//! CUDA indexing must cover every dtype the CPU backend covers.
//!
//! The CPU path dispatches indexing through `dispatch_dtype!`, which spans
//! every `DType`. The CUDA path looks its kernel up by NAME —
//! `kernel_name("index_select", dtype)` — so a dtype with no `.cu`
//! instantiation compiles fine and then fails at launch with
//! `CUDA_ERROR_NOT_FOUND, "named symbol not found"`. Nothing catches that at
//! build time, and nothing caught it in review: `index_select` shipped for
//! f32/f64/f16/bf16/i32/i64/fp8 only, and the gap surfaced when a
//! block-quantized embedding gather indexed a U8 storage tensor on GPU.
//!
//! U8 is the caller that found it, but the fix and this test cover the whole
//! narrow-integer family so the next one does not rediscover it the same way.
//!
//! Run: cargo test --features cuda --test cuda_indexing_dtype_coverage

#![cfg(feature = "cuda")]

use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Gather rows `[2, 0, 3, 0]` out of a `[4, 3]` table and check every element.
///
/// Repeated and out-of-order indices are deliberate: a kernel that ignored the
/// index tensor and copied straight through would still pass an identity
/// gather.
fn check_dtype<T>(values: &[T], expected: &[T], label: &str)
where
    T: numr::dtype::Element + bytemuck::Pod + PartialEq + std::fmt::Debug,
{
    let device = CudaDevice::new(0);
    let client =
        <CudaRuntime as Runtime>::Client::new(device.clone()).expect("CUDA client must initialise");

    let table = Tensor::<CudaRuntime>::from_slice(values, &[4, 3], &device)
        .unwrap_or_else(|e| panic!("{label}: staging the table failed: {e:?}"));
    let idx = Tensor::<CudaRuntime>::from_slice(&[2i64, 0, 3, 0], &[4], &device)
        .unwrap_or_else(|e| panic!("{label}: staging the indices failed: {e:?}"));

    let out = client
        .index_select(&table, 0, &idx)
        .unwrap_or_else(|e| panic!("{label}: index_select failed: {e:?}"));

    assert_eq!(out.shape(), &[4, 3], "{label}: output shape");
    assert_eq!(out.to_vec::<T>(), expected, "{label}: gathered rows");
}

/// The dtype that actually broke: `QuantTensor` storage is U8, so gathering
/// embedding rows out of a GGUF table indexes U8 bytes.
#[test]
fn index_select_covers_u8() {
    let table: Vec<u8> = (0..12u8).collect();
    check_dtype(&table, &[6u8, 7, 8, 0, 1, 2, 9, 10, 11, 0, 1, 2], "u8");
}

#[test]
fn index_select_covers_i8() {
    let table: Vec<i8> = (0..12i8).map(|i| i - 6).collect();
    check_dtype(&table, &[0i8, 1, 2, -6, -5, -4, 3, 4, 5, -6, -5, -4], "i8");
}

#[test]
fn index_select_covers_u16() {
    let table: Vec<u16> = (0..12u16).map(|i| i * 1000).collect();
    check_dtype(
        &table,
        &[
            6000u16, 7000, 8000, 0, 1000, 2000, 9000, 10000, 11000, 0, 1000, 2000,
        ],
        "u16",
    );
}

#[test]
fn index_select_covers_u32() {
    let table: Vec<u32> = (0..12u32).map(|i| i * 100_000).collect();
    check_dtype(
        &table,
        &[
            600_000u32, 700_000, 800_000, 0, 100_000, 200_000, 900_000, 1_000_000, 1_100_000, 0,
            100_000, 200_000,
        ],
        "u32",
    );
}

#[test]
fn index_select_covers_u64() {
    let table: Vec<u64> = (0..12u64).map(|i| i * 10_000_000_000).collect();
    check_dtype(
        &table,
        &[
            60_000_000_000u64,
            70_000_000_000,
            80_000_000_000,
            0,
            10_000_000_000,
            20_000_000_000,
            90_000_000_000,
            100_000_000_000,
            110_000_000_000,
            0,
            10_000_000_000,
            20_000_000_000,
        ],
        "u64",
    );
}
