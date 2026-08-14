// Backend parity tests for NaN and signed-zero ordering in sorting ops.
//
// numr sorts by a single total order on every backend (see `Element::sort_cmp`):
// NaN compares greater than all non-NaN values, NaNs tie with each other, and
// -0.0 ties with +0.0 so ties break by input order.
//
// These live apart from sort.rs because NaN needs its own comparison: the shared
// assert_tensor_allclose treats NaN as a mismatch.

use numr::dtype::DType;
use numr::ops::SortingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{ToF64, create_cpu_client, is_dtype_supported, supported_dtypes};

/// Float dtypes worth exercising for NaN ordering.
///
/// FP8 is excluded: its 8-bit range cannot represent the distinct magnitudes
/// these cases rely on, so a mismatch would report precision, not ordering.
fn nan_dtypes() -> Vec<DType> {
    supported_dtypes("cpu")
        .into_iter()
        .filter(|d| d.is_float() && !matches!(d, DType::FP8E4M3 | DType::FP8E5M2))
        .collect()
}

fn readback_f64<R: Runtime<DType = DType>>(tensor: &Tensor<R>, dtype: DType) -> Vec<f64> {
    macro_rules! read {
        ($T:ty) => {
            tensor
                .to_vec::<$T>()
                .into_iter()
                .map(<$T as ToF64>::to_f64)
                .collect()
        };
    }
    match dtype {
        DType::F32 => read!(f32),
        DType::F64 => read!(f64),
        #[cfg(feature = "f16")]
        DType::F16 => read!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => read!(half::bf16),
        other => panic!("readback_f64: unsupported dtype {other:?}"),
    }
}

/// Read index output as i64 regardless of backend width.
///
/// WebGPU is a 32-bit-only backend, so its sorting ops emit I32 indices where
/// CPU and CUDA emit I64.
fn readback_indices<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Vec<i64> {
    match tensor.dtype() {
        DType::I64 => tensor.to_vec::<i64>(),
        DType::I32 => tensor.to_vec::<i32>().into_iter().map(i64::from).collect(),
        other => panic!("readback_indices: unexpected index dtype {other:?}"),
    }
}

/// Compare sorted values treating NaN as equal to NaN, everything else exactly.
///
/// Sorting only permutes its input, so values must match bit-for-bit rather than
/// within a tolerance; a tolerance here would hide a misplaced element.
fn assert_values_eq(actual: &[f64], expected: &[f64], msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let same = (a.is_nan() && e.is_nan()) || a == e;
        assert!(same, "{msg}: element {i} differs: {a} vs {e}");
    }
}

/// Compares one sorting call across backends for every NaN-capable float dtype.
///
/// The body runs once per backend and returns the values and indices to compare;
/// either may be empty when the operation under test does not produce it.
macro_rules! parity_case {
    ($label:expr, $data:expr, $shape:expr, |$client:ident, $tensor:ident| $body:expr) => {{
        let data: Vec<f64> = $data;
        let shape: Vec<usize> = $shape;

        for dtype in nan_dtypes() {
            let (cpu_client, cpu_device) = create_cpu_client();
            let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor failed for {dtype:?}: {e}"));
            let (cpu_values, cpu_indices) = {
                let $client = &cpu_client;
                let $tensor = &cpu_tensor;
                $body
            };

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| panic!("CUDA tensor failed for {dtype:?}: {e}"));
                    let (values, indices) = {
                        let $client = &cuda_client;
                        let $tensor = &t;
                        $body
                    };
                    assert_values_eq(
                        &values,
                        &cpu_values,
                        &format!("{} values CUDA vs CPU [{dtype:?}]", $label),
                    );
                    assert_eq!(
                        indices, cpu_indices,
                        "{} indices CUDA vs CPU [{dtype:?}]",
                        $label
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, wgpu_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| panic!("WebGPU tensor failed for {dtype:?}: {e}"));
                    let (values, indices) = {
                        let $client = &wgpu_client;
                        let $tensor = &t;
                        $body
                    };
                    assert_values_eq(
                        &values,
                        &cpu_values,
                        &format!("{} values WebGPU vs CPU [{dtype:?}]", $label),
                    );
                    assert_eq!(
                        indices, cpu_indices,
                        "{} indices WebGPU vs CPU [{dtype:?}]",
                        $label
                    );
                });
            }

            let _ = (&cpu_values, &cpu_indices);
        }
    }};
}

fn nan_mixed_input() -> Vec<f64> {
    vec![3.0, f64::NAN, 1.0, 2.0, f64::NAN, 0.0, -1.0, 4.0]
}

#[test]
fn test_sort_nan_ascending_parity() {
    parity_case!(
        "sort ascending with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, false).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            // NaN is the greatest value, so the non-NaN prefix must be ordered.
            assert_eq!(&values[..6], &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
            assert!(values[6].is_nan() && values[7].is_nan());
            (values, Vec::<i64>::new())
        }
    );
}

#[test]
fn test_sort_nan_descending_parity() {
    parity_case!(
        "sort descending with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, true).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            assert!(values[0].is_nan() && values[1].is_nan());
            assert_eq!(&values[2..], &[4.0, 3.0, 2.0, 1.0, 0.0, -1.0]);
            (values, Vec::<i64>::new())
        }
    );
}

#[test]
fn test_argsort_nan_parity() {
    parity_case!(
        "argsort with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let indices = client.argsort(tensor, 0, false).expect("argsort failed");
            let indices = readback_indices(&indices);
            // NaNs tie, so they keep input order: original positions 1 then 4.
            assert_eq!(indices, vec![6, 5, 2, 3, 0, 7, 1, 4]);
            (Vec::<f64>::new(), indices)
        }
    );
}

#[test]
fn test_sort_with_indices_nan_parity() {
    parity_case!(
        "sort_with_indices with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let (sorted, indices) = client
                .sort_with_indices(tensor, 0, false)
                .expect("sort_with_indices failed");
            let values = readback_f64(&sorted, tensor.dtype());
            (values, readback_indices(&indices))
        }
    );
}

/// NaN outranks infinity, and infinities must survive the padding used to reach
/// a power-of-two length on the GPU bitonic path.
#[test]
fn test_sort_nan_beats_infinity_parity() {
    parity_case!(
        "sort with NaN and infinities",
        vec![f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 0.0, 5.0],
        vec![5],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, false).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            assert_eq!(
                &values[..4],
                &[f64::NEG_INFINITY, 0.0, 5.0, f64::INFINITY],
                "infinities must not be dropped by padding"
            );
            assert!(values[4].is_nan());
            (values, Vec::<i64>::new())
        }
    );
}

#[test]
fn test_sort_infinity_descending_parity() {
    parity_case!(
        "sort descending with infinities",
        vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, 5.0, -5.0],
        vec![5],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, true).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            assert_eq!(
                values,
                vec![f64::INFINITY, 5.0, 0.0, -5.0, f64::NEG_INFINITY]
            );
            (values, Vec::<i64>::new())
        }
    );
}

#[test]
fn test_argsort_all_nan_parity() {
    parity_case!(
        "argsort all-NaN slice",
        vec![f64::NAN; 6],
        vec![6],
        |client, tensor| {
            let indices = client.argsort(tensor, 0, false).expect("argsort failed");
            let indices = readback_indices(&indices);
            // Every element ties, so input order is preserved.
            assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
            (Vec::<f64>::new(), indices)
        }
    );
}

/// -0.0 and +0.0 tie, so argsort must report input order rather than bit order.
#[test]
fn test_argsort_signed_zero_parity() {
    for descending in [false, true] {
        parity_case!(
            "argsort signed zeros",
            vec![0.0, -0.0, 0.0, -0.0],
            vec![4],
            |client, tensor| {
                let indices = client
                    .argsort(tensor, 0, descending)
                    .expect("argsort failed");
                let indices = readback_indices(&indices);
                assert_eq!(indices, vec![0, 1, 2, 3], "descending={descending}");
                (Vec::<f64>::new(), indices)
            }
        );
    }
}

/// Duplicate keys must keep input order in both directions. Descending was the
/// weak case: the GPU networks previously broke ties by descending index.
#[test]
fn test_argsort_duplicate_stability_parity() {
    for descending in [false, true] {
        parity_case!(
            "argsort duplicate stability",
            vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            vec![6],
            |client, tensor| {
                let indices = client
                    .argsort(tensor, 0, descending)
                    .expect("argsort failed");
                let indices = readback_indices(&indices);
                let expected = if descending {
                    vec![0, 2, 4, 1, 3, 5]
                } else {
                    vec![1, 3, 5, 0, 2, 4]
                };
                assert_eq!(indices, expected, "descending={descending}");
                (Vec::<f64>::new(), indices)
            }
        );
    }
}

/// NaN is the greatest value, so `largest` ranks it above every real element.
#[test]
fn test_topk_nan_largest_parity() {
    parity_case!(
        "topk largest with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let (values, indices) = client.topk(tensor, 3, 0, true, true).expect("topk failed");
            let values = readback_f64(&values, tensor.dtype());
            assert!(values[0].is_nan() && values[1].is_nan());
            assert_eq!(values[2], 4.0);
            (values, readback_indices(&indices))
        }
    );
}

/// `largest = false` sorts ascending, so NaN falls outside the k smallest.
#[test]
fn test_topk_nan_smallest_parity() {
    parity_case!(
        "topk smallest with NaN",
        nan_mixed_input(),
        vec![8],
        |client, tensor| {
            let (values, indices) = client.topk(tensor, 3, 0, false, true).expect("topk failed");
            let values = readback_f64(&values, tensor.dtype());
            assert_eq!(values, vec![-1.0, 0.0, 1.0]);
            (values, readback_indices(&indices))
        }
    );
}

/// searchsorted must use the same order the sequence was sorted by, which puts
/// the NaN run at the end.
#[test]
fn test_searchsorted_nan_parity() {
    let seq = vec![1.0, 2.0, 3.0, f64::NAN];
    let vals = vec![f64::NAN, 2.5, 0.5];

    for right in [false, true] {
        for dtype in nan_dtypes() {
            let (cpu_client, cpu_device) = create_cpu_client();
            let cpu_seq = tensor_from_f64(&seq, &[4], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor failed for {dtype:?}: {e}"));
            let cpu_vals = tensor_from_f64(&vals, &[3], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor failed for {dtype:?}: {e}"));
            let cpu_out = readback_indices(
                &cpu_client
                    .searchsorted(&cpu_seq, &cpu_vals, right)
                    .expect("CPU searchsorted failed"),
            );

            // NaN lands at the head of the trailing NaN run, or past it for `right`.
            let expected = if right { vec![4, 2, 0] } else { vec![3, 2, 0] };
            assert_eq!(cpu_out, expected, "CPU [{dtype:?}] right={right}");

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|client, device| {
                    let s = tensor_from_f64(&seq, &[4], dtype, &device, &client).unwrap();
                    let v = tensor_from_f64(&vals, &[3], dtype, &device, &client).unwrap();
                    let out = readback_indices(
                        &client.searchsorted(&s, &v, right).expect("searchsorted"),
                    );
                    assert_eq!(out, cpu_out, "searchsorted CUDA vs CPU [{dtype:?}]");
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|client, device| {
                    let s = tensor_from_f64(&seq, &[4], dtype, &device, &client).unwrap();
                    let v = tensor_from_f64(&vals, &[3], dtype, &device, &client).unwrap();
                    let out = readback_indices(
                        &client.searchsorted(&s, &v, right).expect("searchsorted"),
                    );
                    assert_eq!(out, cpu_out, "searchsorted WebGPU vs CPU [{dtype:?}]");
                });
            }
        }
    }
}

/// A non-power-of-two length forces the GPU padding path.
#[test]
fn test_sort_nan_non_power_of_two_parity() {
    let mut data = vec![f64::NAN, 7.0, f64::NAN];
    data.extend((0..60).map(|i| f64::from(60 - i)));
    let len = data.len();

    parity_case!(
        "sort NaN with padded length",
        data,
        vec![len],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, false).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            let nan_count = values.iter().filter(|v| v.is_nan()).count();
            assert_eq!(nan_count, 2, "NaNs must not be lost or duplicated");
            assert!(values[values.len() - 2].is_nan() && values[values.len() - 1].is_nan());
            (values, Vec::<i64>::new())
        }
    );
}

/// Inputs above the shared-memory tile size take a different WebGPU kernel, so
/// the ordering contract has to be checked there too.
#[test]
fn test_sort_nan_beyond_shared_memory_tile_parity() {
    // 1000 > the 512-element tile, and not a power of two, so padding is in play.
    let mut data: Vec<f64> = (0..1000).map(|i| ((i * 37) % 501) as f64 - 250.0).collect();
    for idx in [0usize, 1, 499, 500, 511, 512, 513, 998, 999] {
        data[idx] = f64::NAN;
    }
    let nan_count = data.iter().filter(|v| v.is_nan()).count();

    parity_case!(
        "sort ascending with NaN beyond tile",
        data.clone(),
        vec![1000],
        |client, tensor| {
            let sorted = client.sort(tensor, 0, false).expect("sort failed");
            let values = readback_f64(&sorted, tensor.dtype());
            assert_eq!(
                values.iter().filter(|v| v.is_nan()).count(),
                nan_count,
                "NaNs must not be lost or duplicated"
            );
            assert!(
                values[values.len() - nan_count..]
                    .iter()
                    .all(|v| v.is_nan()),
                "NaNs must occupy the tail"
            );
            let head = &values[..values.len() - nan_count];
            assert!(
                head.windows(2).all(|w| w[0] <= w[1]),
                "non-NaN prefix sorted"
            );
            (values, Vec::<i64>::new())
        }
    );
}

/// Stability above the tile size: duplicates must keep input order in both
/// directions, which the global path resolves with its own index tiebreak.
#[test]
fn test_argsort_duplicate_stability_beyond_tile_parity() {
    let data: Vec<f64> = (0..900).map(|i| (i % 3) as f64).collect();

    for descending in [false, true] {
        parity_case!(
            "argsort duplicate stability beyond tile",
            data.clone(),
            vec![900],
            |client, tensor| {
                let indices = client
                    .argsort(tensor, 0, descending)
                    .expect("argsort failed");
                let indices = readback_indices(&indices);
                // Equal keys form runs; within each run indices must ascend.
                let keys: Vec<f64> = indices.iter().map(|&i| data[i as usize]).collect();
                for w in keys.windows(2) {
                    if descending {
                        assert!(w[0] >= w[1], "keys ordered descending");
                    } else {
                        assert!(w[0] <= w[1], "keys ordered ascending");
                    }
                }
                for pair in indices.windows(2) {
                    if data[pair[0] as usize] == data[pair[1] as usize] {
                        assert!(
                            pair[0] < pair[1],
                            "ties keep input order (descending={descending})"
                        );
                    }
                }
                (Vec::<f64>::new(), indices)
            }
        );
    }
}
