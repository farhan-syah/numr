//! Arbitrary-size FFT on CUDA via Bluestein (chirp-z).
//!
//! The Stockham kernels accept only power-of-two transform lengths. Bluestein
//! rewrites an N-point DFT as a cyclic convolution of length
//! `M = next_power_of_two(2N - 1)`, which they DO accept, so every `N >= 1`
//! becomes available without padding the signal (padding would change the
//! frequency grid, not just the cost).
//!
//! This exists because real audio sizes are not powers of two: NeuCodec's
//! vocoder iSTFT is `n_fft = 1920` and Kokoro's is 20. Without it those decode
//! paths are pinned to CPU.
//!
//! # Precision
//!
//! The chirp and kernel-spectrum tables come from
//! [`crate::algorithm::fft_bluestein`], which builds them in f64 on the host —
//! the same tables the CPU backend uses, so the two cannot disagree about the
//! chirp. The convolution itself runs in the caller's own dtype: `Complex64`
//! input convolves in f32. Forcing f64 there would cost ~32x on consumer cards
//! for accuracy the caller did not ask for, and the tables are where phase
//! precision is actually won.

use super::CudaRuntime;
use super::allocator::CudaAllocator;
use super::client::CudaClient;
use super::kernels;
use crate::algorithm::fft_bluestein::cached_tables;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};

pub use kernels::BluesteinInput;

/// Upload a host slice as raw bytes into a fresh device allocation.
fn upload<'a, T: bytemuck::Pod>(
    client: &'a CudaClient,
    host: &[T],
) -> Result<AllocGuard<'a, CudaAllocator>> {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let guard = AllocGuard::new(client.allocator(), bytes.len())?;
    CudaRuntime::copy_to_device(bytes, guard.ptr(), client.device())?;
    Ok(guard)
}

/// Out-of-place power-of-two Stockham transform, `src` -> `dst`.
///
/// Mirrors the small/large split used by [`super::fft`]: a single shared-memory
/// kernel below [`kernels::MAX_SHARED_MEM_FFT_SIZE`], otherwise `log2(m)`
/// ping-pong stages.
fn stockham(
    client: &CudaClient,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    m: usize,
    batch_size: usize,
    inverse: bool,
    scale: f64,
) -> Result<()> {
    let device = client.device();
    let bytes = batch_size * m * dtype.size_in_bytes();

    if m <= kernels::MAX_SHARED_MEM_FFT_SIZE {
        unsafe {
            kernels::launch_stockham_fft_batched(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                src_ptr,
                dst_ptr,
                m,
                batch_size,
                inverse,
                scale,
            )?;
        }
        return Ok(());
    }

    let temp_guard = AllocGuard::new(client.allocator(), bytes)?;
    CudaRuntime::copy_within_device(src_ptr, dst_ptr, bytes, device)?;

    let mut cur = dst_ptr;
    let mut other = temp_guard.ptr();
    let log_m = m.trailing_zeros() as usize;
    for stage in 0..log_m {
        unsafe {
            kernels::launch_stockham_fft_stage(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                cur,
                other,
                m,
                stage,
                batch_size,
                inverse,
            )?;
        }
        std::mem::swap(&mut cur, &mut other);
    }

    if (scale - 1.0).abs() > 1e-10 {
        unsafe {
            kernels::launch_scale_complex(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                cur,
                scale,
                batch_size * m,
            )?;
        }
    }

    if cur != dst_ptr {
        CudaRuntime::copy_within_device(cur, dst_ptr, bytes, device)?;
    }
    Ok(())
}

/// Run an arbitrary-size transform, returning the output allocation.
///
/// * `input_ptr` — `batch_size * n` elements, complex or real per `kind`
/// * `out_n` — `n` for a full transform, `n / 2 + 1` to keep only the Hermitian
///   half (`rfft`)
/// * `scale` — normalization applied on the final store
///
/// The returned guard owns `batch_size * out_n` complex elements.
#[allow(clippy::too_many_arguments)]
pub(super) fn bluestein_transform<'a>(
    client: &'a CudaClient,
    complex_dtype: DType,
    kind: BluesteinInput,
    input_ptr: u64,
    n: usize,
    out_n: usize,
    batch_size: usize,
    inverse: bool,
    scale: f64,
) -> Result<AllocGuard<'a, CudaAllocator>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "Bluestein transform requires N >= 1".to_string(),
        });
    }

    let tables = cached_tables(n, inverse)?;
    let m = tables.m;
    let device = client.device();
    let elem = complex_dtype.size_in_bytes();

    let (chirp_guard, kernel_guard) = match complex_dtype {
        DType::Complex64 => (
            upload(client, &tables.chirp_f32())?,
            upload(client, &tables.kernel_spectrum_f32())?,
        ),
        DType::Complex128 => (
            upload(client, &tables.chirp_f64())?,
            upload(client, &tables.kernel_spectrum_f64())?,
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: complex_dtype,
                op: "bluestein_transform",
            });
        }
    };

    let conv_bytes = batch_size * m * elem;
    let a_guard = AllocGuard::new(client.allocator(), conv_bytes)?;
    let spec_guard = AllocGuard::new(client.allocator(), conv_bytes)?;

    unsafe {
        kernels::launch_bluestein_premultiply(
            client.context(),
            client.stream(),
            device.index,
            complex_dtype,
            kind,
            input_ptr,
            chirp_guard.ptr(),
            a_guard.ptr(),
            n,
            m,
            batch_size,
        )?;
    }

    // Forward M-point transform of the chirped signal.
    stockham(
        client,
        complex_dtype,
        a_guard.ptr(),
        spec_guard.ptr(),
        m,
        batch_size,
        false,
        1.0,
    )?;

    unsafe {
        kernels::launch_bluestein_pointwise_mul(
            client.context(),
            client.stream(),
            device.index,
            complex_dtype,
            spec_guard.ptr(),
            kernel_guard.ptr(),
            m,
            batch_size,
        )?;
    }

    // Inverse M-point transform completes the cyclic convolution. The 1/M here
    // is the convolution's own normalization and is unrelated to `scale`, which
    // is the caller's FFT normalization and is applied at the final store.
    stockham(
        client,
        complex_dtype,
        spec_guard.ptr(),
        a_guard.ptr(),
        m,
        batch_size,
        true,
        1.0 / m as f64,
    )?;

    let out_guard = AllocGuard::new(client.allocator(), batch_size * out_n * elem)?;
    unsafe {
        kernels::launch_bluestein_postmultiply(
            client.context(),
            client.stream(),
            device.index,
            complex_dtype,
            a_guard.ptr(),
            chirp_guard.ptr(),
            out_guard.ptr(),
            m,
            out_n,
            batch_size,
            scale,
        )?;
    }

    client.synchronize();
    Ok(out_guard)
}
