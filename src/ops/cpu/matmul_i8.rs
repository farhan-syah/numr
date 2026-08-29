//! CPU I8 matmul: the one element type whose product does not fit its operands.
//!
//! `A @ B` on I8 is a quantized accumulation, so both the plain and the fused-
//! bias form allocate an I32 output and sum in i32 (see `ops/matmul_dtype.rs`
//! for the dtype rule the backends share). Keeping both forms here is what
//! stops the widening from being spelled twice in `matmul.rs`.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::cpu::kernels::{matmul_i8_to_i32_bias_kernel, matmul_i8_to_i32_kernel};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Run an I8 matmul, with an optional I32 bias, into a freshly allocated I32
/// tensor.
///
/// `a` and `b` must already be contiguous; `bias`, when given, is a contiguous
/// I32 vector of length `n` that seeds every row's accumulator.
///
/// The two forms share one function because they differ only in that seed: a
/// bias added after the store would be added to a narrowed value, which is a
/// different number wherever the accumulator leaves the output's range.
#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_i8_i32(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    bias: Option<&Tensor<CpuRuntime>>,
    out_shape: &[usize],
    a_batch_idx: &[usize],
    b_batch_idx: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<CpuRuntime>> {
    let out = Tensor::<CpuRuntime>::empty(out_shape, DType::I32, &client.device)?;

    // Addresses, not pointers: Rayon's closure bound needs a `Sync` capture, and
    // a raw pointer is not one. Every use below casts inside the closure.
    let a_addr = a.ptr();
    let b_addr = b.ptr();
    let out_addr = out.ptr();
    // The bias is [n], so every batch reads the same vector.
    let bias_addr = bias.map(|t| t.ptr());

    // Leading dimensions for contiguous row-major matrices.
    let (lda, ldb, ldc) = (k, n, n);
    let batch_size = a_batch_idx.len();

    // One batch element, as a closure so the serial and parallel paths below
    // cannot drift apart on the offset arithmetic.
    //
    // SAFETY: every pointer is derived from a contiguous tensor of the shape the
    // caller validated, and each batch writes a disjoint m*n block of `out`.
    let run_batch = |batch: usize| unsafe {
        let a_ptr = a_addr as *const i8;
        let b_ptr = b_addr as *const i8;
        let out_ptr = out_addr as *mut i32;

        let a_off = a_batch_idx[batch] * m * k;
        let b_off = b_batch_idx[batch] * k * n;
        let out_off = batch * m * n;

        match bias_addr {
            Some(bias_addr) => matmul_i8_to_i32_bias_kernel(
                a_ptr.add(a_off),
                b_ptr.add(b_off),
                bias_addr as *const i32,
                out_ptr.add(out_off),
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
            ),
            None => matmul_i8_to_i32_kernel(
                a_ptr.add(a_off),
                b_ptr.add(b_off),
                out_ptr.add(out_off),
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
            ),
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        if batch_size > 1 {
            let min_len = client.rayon_min_len();
            client.install_parallelism(|| {
                (0..batch_size)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .for_each(run_batch);
            });
        } else {
            for batch in 0..batch_size {
                run_batch(batch);
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for batch in 0..batch_size {
            run_batch(batch);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::error::Error;
    use crate::ops::MatmulOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    /// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]], plus [1,-2] per column.
    /// B is read column-wise: column 0 is [5,7], column 1 is [6,8].
    #[test]
    fn test_matmul_bias_i8_returns_i32() {
        let (c, dev) = client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1i8, 2, 3, 4], &[2, 2], &dev).expect("A");
        let b = Tensor::<CpuRuntime>::from_slice(&[5i8, 6, 7, 8], &[2, 2], &dev).expect("B");
        let bias = Tensor::<CpuRuntime>::from_slice(&[1i32, -2], &[2], &dev).expect("bias");

        let out = c.matmul_bias(&a, &b, &bias).expect("matmul_bias");
        assert_eq!(out.dtype(), DType::I32);
        assert_eq!(out.to_vec::<i32>(), vec![20i32, 20, 44, 48]);
    }

    /// 4 * 127 * 127 = 64_516, plus a bias of 1_000. Neither the product nor the
    /// sum fits I8, and both fit I32.
    #[test]
    fn test_matmul_bias_i8_accumulates_past_i8_range() {
        let (c, dev) = client();
        let a = Tensor::<CpuRuntime>::from_slice(&[127i8; 4], &[1, 4], &dev).expect("A");
        let b = Tensor::<CpuRuntime>::from_slice(&[127i8; 4], &[4, 1], &dev).expect("B");
        let bias = Tensor::<CpuRuntime>::from_slice(&[1_000i32], &[1], &dev).expect("bias");

        let out = c.matmul_bias(&a, &b, &bias).expect("matmul_bias");
        assert_eq!(out.dtype(), DType::I32);
        assert_eq!(out.to_vec::<i32>(), vec![65_516i32]);
    }

    /// Batched: slice 1 is slice 0 doubled, and the bias applies to both.
    #[test]
    fn test_matmul_bias_i8_batched_returns_i32() {
        let (c, dev) = client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1i8, 2, 3, 4, 5, 6, 7, 8], &[2, 2, 2], &dev)
            .expect("A");
        let b = Tensor::<CpuRuntime>::from_slice(&[1i8, 0, 0, 1, 2, 0, 0, 2], &[2, 2, 2], &dev)
            .expect("B");
        let bias = Tensor::<CpuRuntime>::from_slice(&[1i32, -2], &[2], &dev).expect("bias");

        let out = c.matmul_bias(&a, &b, &bias).expect("matmul_bias");
        assert_eq!(out.dtype(), DType::I32);
        // batch 0 = [[1,2],[3,4]] + [1,-2]; batch 1 = [[10,12],[14,16]] + [1,-2]
        assert_eq!(out.to_vec::<i32>(), vec![2i32, 0, 4, 2, 11, 10, 15, 14]);
    }

    /// An I8 bias is refused, and the message names the op, the dtype it got,
    /// and the dtype it wanted.
    #[test]
    fn test_matmul_bias_i8_rejects_i8_bias() {
        let (c, dev) = client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1i8, 2, 3, 4], &[2, 2], &dev).expect("A");
        let b = Tensor::<CpuRuntime>::from_slice(&[5i8, 6, 7, 8], &[2, 2], &dev).expect("B");
        let bias = Tensor::<CpuRuntime>::from_slice(&[1i8, -2], &[2], &dev).expect("bias");

        match c.matmul_bias(&a, &b, &bias) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "bias");
                assert!(reason.contains("matmul_bias"), "reason: {reason}");
                assert!(reason.contains("I32"), "reason: {reason}");
            }
            other => panic!("expected an I8 bias to be rejected, got {other:?}"),
        }
    }
}
