//! Arbitrary-size FFT on WebGPU via Bluestein (chirp-z).
//!
//! The Stockham shader accepts only power-of-two transform lengths. Bluestein
//! reduces an N-point DFT to a cyclic convolution of length
//! `M = next_power_of_two(2N - 1)`, which it does accept, so every `N >= 1`
//! becomes available without padding the signal — padding would change the
//! frequency grid, not just the cost.
//!
//! Sizes this unblocks are not hypothetical: NeuCodec's vocoder iSTFT is
//! `n_fft = 1920` and Kokoro's is 20.
//!
//! Complex64 only, like the rest of the WebGPU FFT path — WGSL has no f64. The
//! chirp tables are still built in f64 on the host (see
//! [`crate::algorithm::fft_bluestein`]) and narrowed on upload, so the phase is
//! computed at full precision even though the convolution runs in f32.

use super::client::WgpuAllocator;
use super::client::get_buffer;
use super::shaders::fft as kernels;
use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::fft_bluestein::cached_tables;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};

/// Uniform block matching `BluesteinParams` in `fft_bluestein.wgsl`.
fn params_words(n: usize, m: usize, batch_size: usize, out_n: usize, scale: f32) -> [u32; 8] {
    [
        n as u32,
        m as u32,
        batch_size as u32,
        out_n as u32,
        scale.to_bits(),
        0,
        0,
        0,
    ]
}

/// Out-of-place power-of-two Stockham transform, `src` -> `dst`.
///
/// Mirrors the small/large split in [`super::fft`]: one workgroup-shared
/// dispatch below [`kernels::MAX_WORKGROUP_FFT_SIZE`], otherwise `log2(m)`
/// ping-pong stages.
fn stockham(
    client: &WgpuClient,
    src_ptr: u64,
    dst_ptr: u64,
    m: usize,
    batch_size: usize,
    inverse: bool,
    scale: f32,
) -> Result<()> {
    let device = client.device();
    let bytes = batch_size * m * DType::Complex64.size_in_bytes();
    let log_m = m.trailing_zeros();

    let params = client.create_uniform_buffer("bluestein_fft_params", 32);
    client.write_buffer(
        &params,
        &[
            m as u32,
            log_m,
            u32::from(inverse),
            scale.to_bits(),
            batch_size as u32,
            0,
            0,
            0,
        ],
    );

    let src = get_buffer(src_ptr)
        .ok_or_else(|| Error::Internal("Bluestein stockham src buffer missing".to_string()))?;
    let dst = get_buffer(dst_ptr)
        .ok_or_else(|| Error::Internal("Bluestein stockham dst buffer missing".to_string()))?;

    if m <= kernels::MAX_WORKGROUP_FFT_SIZE {
        kernels::launch_stockham_fft_batched(
            client.pipeline_cache(),
            &client.queue,
            &src,
            &dst,
            &params,
            m,
            batch_size,
        )?;
        return Ok(());
    }

    let temp_guard = AllocGuard::new(client.allocator(), bytes)?;
    let temp_ptr = temp_guard.ptr();
    WgpuRuntime::copy_within_device(src_ptr, temp_ptr, bytes, device)?;

    let mut use_temp_as_input = true;
    for stage in 0..log_m {
        client.write_buffer(
            &params,
            &[
                m as u32,
                stage,
                u32::from(inverse),
                1.0f32.to_bits(),
                batch_size as u32,
                0,
                0,
                0,
            ],
        );
        let (s_ptr, d_ptr) = if use_temp_as_input {
            (temp_ptr, dst_ptr)
        } else {
            (dst_ptr, temp_ptr)
        };
        let s = get_buffer(s_ptr)
            .ok_or_else(|| Error::Internal("Bluestein stage src missing".to_string()))?;
        let d = get_buffer(d_ptr)
            .ok_or_else(|| Error::Internal("Bluestein stage dst missing".to_string()))?;
        kernels::launch_stockham_fft_stage(
            client.pipeline_cache(),
            &client.queue,
            &s,
            &d,
            &params,
            m,
            batch_size,
        )?;
        use_temp_as_input = !use_temp_as_input;
    }

    // After log_m swaps the result sits in temp when log_m is odd.
    let result_ptr = if use_temp_as_input { temp_ptr } else { dst_ptr };
    if (scale - 1.0).abs() > 1e-10 {
        client.write_buffer(
            &params,
            &[
                (batch_size * m) as u32,
                0,
                0,
                scale.to_bits(),
                batch_size as u32,
                0,
                0,
                0,
            ],
        );
        let r = get_buffer(result_ptr)
            .ok_or_else(|| Error::Internal("Bluestein scale buffer missing".to_string()))?;
        kernels::launch_scale_complex(
            client.pipeline_cache(),
            &client.queue,
            &r,
            &r,
            &params,
            batch_size * m,
        )?;
    }
    if result_ptr != dst_ptr {
        WgpuRuntime::copy_within_device(result_ptr, dst_ptr, bytes, device)?;
    }
    let _ = dst;
    Ok(())
}

/// Run an arbitrary-size Complex64 transform, returning the output allocation.
///
/// * `input_ptr` — `batch_size * n` Complex64 elements (real input must be
///   packed to complex first, with `rfft_pack`)
/// * `out_n` — `n` for a full transform, `n / 2 + 1` to keep only the Hermitian
///   half
pub(super) fn bluestein_transform<'a>(
    client: &'a WgpuClient,
    input_ptr: u64,
    n: usize,
    out_n: usize,
    batch_size: usize,
    inverse: bool,
    scale: f32,
) -> Result<AllocGuard<'a, WgpuAllocator>> {
    let tables = cached_tables(n, inverse)?;
    let m = tables.m;
    let device = client.device();
    let elem = DType::Complex64.size_in_bytes();

    let chirp_host = tables.chirp_f32();
    let kernel_host = tables.kernel_spectrum_f32();
    let chirp_guard = AllocGuard::new(client.allocator(), chirp_host.len() * 4)?;
    let kernel_guard = AllocGuard::new(client.allocator(), kernel_host.len() * 4)?;
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&chirp_host), chirp_guard.ptr(), device)?;
    WgpuRuntime::copy_to_device(
        bytemuck::cast_slice(&kernel_host),
        kernel_guard.ptr(),
        device,
    )?;

    let conv_bytes = batch_size * m * elem;
    let a_guard = AllocGuard::new(client.allocator(), conv_bytes)?;
    let spec_guard = AllocGuard::new(client.allocator(), conv_bytes)?;
    let out_guard = AllocGuard::new(client.allocator(), batch_size * out_n * elem)?;

    let params = client.create_uniform_buffer("bluestein_params", 32);

    let fetch = |ptr: u64, what: &'static str| {
        get_buffer(ptr).ok_or_else(|| Error::Internal(format!("Bluestein {what} buffer missing")))
    };
    let input_buf = fetch(input_ptr, "input")?;
    let chirp_buf = fetch(chirp_guard.ptr(), "chirp")?;
    let kernel_buf = fetch(kernel_guard.ptr(), "kernel")?;
    let a_buf = fetch(a_guard.ptr(), "a")?;
    let spec_buf = fetch(spec_guard.ptr(), "spectrum")?;
    let out_buf = fetch(out_guard.ptr(), "out")?;

    client.write_buffer(&params, &params_words(n, m, batch_size, out_n, scale));
    kernels::launch_bluestein_stage(
        client.pipeline_cache(),
        &client.queue,
        "bluestein_premultiply",
        &input_buf,
        &chirp_buf,
        &a_buf,
        &params,
        m,
        batch_size,
    )?;

    stockham(
        client,
        a_guard.ptr(),
        spec_guard.ptr(),
        m,
        batch_size,
        false,
        1.0,
    )?;

    client.write_buffer(&params, &params_words(n, m, batch_size, out_n, scale));
    kernels::launch_bluestein_stage(
        client.pipeline_cache(),
        &client.queue,
        "bluestein_pointwise_mul",
        &spec_buf,
        &kernel_buf,
        &a_buf,
        &params,
        m,
        batch_size,
    )?;

    // The 1/M is the convolution's own normalization, unrelated to `scale`,
    // which is the caller's FFT normalization and lands in the postmultiply.
    stockham(
        client,
        a_guard.ptr(),
        spec_guard.ptr(),
        m,
        batch_size,
        true,
        1.0 / m as f32,
    )?;

    client.write_buffer(&params, &params_words(n, m, batch_size, out_n, scale));
    kernels::launch_bluestein_stage(
        client.pipeline_cache(),
        &client.queue,
        "bluestein_postmultiply",
        &spec_buf,
        &chirp_buf,
        &out_buf,
        &params,
        out_n,
        batch_size,
    )?;

    client.synchronize();
    Ok(out_guard)
}
