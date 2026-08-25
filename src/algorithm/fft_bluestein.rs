//! Host-side Bluestein (chirp-z) tables, shared by every backend.
//!
//! Bluestein rewrites an N-point DFT as a cyclic convolution of length
//! `M = (2N - 1).next_power_of_two()`, which any radix-2 kernel evaluates
//! directly. That makes every size `N >= 1` available without padding the
//! signal, which would change the frequency grid.
//!
//! Only the TABLES live here — the convolution itself is each backend's own
//! radix-2 kernel. The tables are the part where precision is won or lost, and
//! sharing them means a CPU and a CUDA transform of the same size cannot
//! disagree about the chirp. They are always built in f64 regardless of the
//! caller's dtype; a backend narrows only when it uploads.
//!
//! Building the tables costs one host-side M-point FFT, so a backend that
//! calls this per transform should cache by `(n, inverse)`.

use crate::dtype::Complex128;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock, Mutex};

/// Chirp sequence and convolution-kernel spectrum for one N-point transform.
///
/// With `w[k] = exp(sign * i * pi * k^2 / N)` (`sign = -1` forward, `+1` inverse):
///
/// ```text
/// a[j] = x[j] * w[j]                 (j < N, zero padded to M)
/// b[t] = conj(w[t]), b[M - t] = b[t] (1 <= t < N, zero elsewhere)
/// c    = ifft_M(fft_M(a) * fft_M(b))
/// X[k] = w[k] * c[k]
/// ```
#[derive(Debug, Clone)]
pub struct BluesteinTables {
    /// Transform length.
    pub n: usize,
    /// Convolution length, `(2N - 1).next_power_of_two()`.
    pub m: usize,
    /// `w[k]` for `k < N`.
    pub chirp: Vec<Complex128>,
    /// `fft_M(b)`, length M.
    pub kernel_spectrum: Vec<Complex128>,
}

/// Build the length-M convolution kernel `b` from a chirp.
///
/// `M >= 2N - 1` guarantees the head (`t < N`) and the mirrored tail
/// (`M - t >= N`) never collide, so no entry is written twice.
fn kernel_from_chirp(chirp: &[Complex128], m: usize) -> Vec<Complex128> {
    let n = chirp.len();
    let mut kernel = vec![Complex128::default(); m];
    kernel[0] = chirp[0].conj();
    for t in 1..n {
        let v = chirp[t].conj();
        kernel[t] = v;
        kernel[m - t] = v;
    }
    kernel
}

/// The chirp `w[k] = exp(sign * i * pi * k^2 / N)` for `k < n`.
///
/// The phase is `pi * (k^2 mod 2N) / N` with `k^2 mod 2N` accumulated in
/// integer arithmetic via `q_{k+1} = q_k + 2k + 1 (mod 2N)`. Evaluating
/// `pi * k * k / N` directly destroys all precision once `k^2` exceeds the f64
/// mantissa — at N = 1920 that happens well inside the table.
pub fn chirp_sequence(n: usize, inverse: bool) -> Vec<Complex128> {
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    let two_n = 2 * n;
    let mut chirp = Vec::with_capacity(n);
    let mut q = 0usize;
    for k in 0..n {
        let theta = sign * PI * (q as f64) / (n as f64);
        chirp.push(Complex128::new(theta.cos(), theta.sin()));
        q = (q + 2 * k + 1) % two_n;
    }
    chirp
}

impl BluesteinTables {
    /// Build tables for an N-point transform.
    ///
    /// `forward_fft_m` computes an unnormalized forward M-point FFT in place of
    /// the caller's own radix-2 kernel, so this module needs no backend types.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new<F>(n: usize, inverse: bool, forward_fft_m: F) -> Self
    where
        F: FnOnce(&[Complex128]) -> Vec<Complex128>,
    {
        assert!(n >= 1, "Bluestein FFT requires N >= 1");
        let m = (2 * n - 1).next_power_of_two();
        let chirp = chirp_sequence(n, inverse);
        let kernel_spectrum = forward_fft_m(&kernel_from_chirp(&chirp, m));
        debug_assert_eq!(kernel_spectrum.len(), m);
        Self {
            n,
            m,
            chirp,
            kernel_spectrum,
        }
    }

    /// `chirp` narrowed to interleaved f32 pairs, for upload to a Complex64 buffer.
    pub fn chirp_f32(&self) -> Vec<f32> {
        self.chirp
            .iter()
            .flat_map(|c| [c.re as f32, c.im as f32])
            .collect()
    }

    /// `kernel_spectrum` narrowed to interleaved f32 pairs.
    pub fn kernel_spectrum_f32(&self) -> Vec<f32> {
        self.kernel_spectrum
            .iter()
            .flat_map(|c| [c.re as f32, c.im as f32])
            .collect()
    }

    /// `chirp` as interleaved f64 pairs, for upload to a Complex128 buffer.
    pub fn chirp_f64(&self) -> Vec<f64> {
        self.chirp.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    /// `kernel_spectrum` as interleaved f64 pairs.
    pub fn kernel_spectrum_f64(&self) -> Vec<f64> {
        self.kernel_spectrum
            .iter()
            .flat_map(|c| [c.re, c.im])
            .collect()
    }
}

/// Cache of built tables, keyed by `(n, inverse)`.
///
/// Building tables costs one host M-point FFT. Only the HOST side is cached —
/// a backend uploads its own device buffers per call — because caching device
/// allocations would tie their lifetime to a static and outlive the allocator
/// that made them.
static TABLE_CACHE: LazyLock<Mutex<HashMap<(usize, bool), Arc<BluesteinTables>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Unnormalized forward FFT of a power-of-two host buffer, in f64.
///
/// Iterative radix-2 Cooley-Tukey. Self-contained rather than calling a
/// backend's kernel, so no backend depends on another being compiled in. It
/// runs once per `(n, inverse)` and the result is cached.
fn host_fft(x: &[Complex128]) -> Vec<Complex128> {
    let m = x.len();
    debug_assert!(m.is_power_of_two(), "Bluestein M must be a power of two");
    let mut a = x.to_vec();

    let bits = m.trailing_zeros();
    for i in 0..m {
        let j = ((i as u32).reverse_bits() >> (32 - bits)) as usize;
        if j > i {
            a.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= m {
        let ang = -2.0 * PI / (len as f64);
        for start in (0..m).step_by(len) {
            for k in 0..len / 2 {
                let theta = ang * (k as f64);
                let w = Complex128::new(theta.cos(), theta.sin());
                let u = a[start + k];
                let v = a[start + k + len / 2] * w;
                a[start + k] = u + v;
                a[start + k + len / 2] = u - v;
            }
        }
        len <<= 1;
    }
    a
}

/// Fetch or build the tables for an N-point transform.
///
/// Shared by every GPU backend so a CUDA and a WebGPU transform of the same
/// size use byte-identical chirps.
pub fn cached_tables(n: usize, inverse: bool) -> Result<Arc<BluesteinTables>> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "Bluestein transform requires N >= 1".to_string(),
        });
    }
    let mut cache = TABLE_CACHE
        .lock()
        .map_err(|_| Error::Internal("Bluestein table cache poisoned".to_string()))?;
    if let Some(t) = cache.get(&(n, inverse)) {
        return Ok(Arc::clone(t));
    }
    let tables = Arc::new(BluesteinTables::new(n, inverse, host_fft));
    cache.insert((n, inverse), Arc::clone(&tables));
    Ok(tables)
}

#[cfg(test)]
mod tests;
