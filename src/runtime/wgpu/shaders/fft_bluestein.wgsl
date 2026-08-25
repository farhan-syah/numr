// Bluestein (chirp-z) FFT stages for arbitrary transform sizes.
//
// stockham_fft.wgsl only accepts power-of-two N. Bluestein rewrites an N-point
// DFT as a cyclic convolution of length M = next_power_of_two(2N - 1), which it
// DOES accept. These are the pre/post stages around that convolution. The chirp
// and kernel-spectrum tables are built on the host in f64 and uploaded as f32
// (see numr::algorithm::fft_bluestein), so this backend and the CPU one cannot
// disagree about the chirp.
//
// WGSL has no f64, so everything here is Complex64 (vec2<f32>) only.
//
// All three entry points share ONE binding set, because WGSL requires a unique
// (group, binding) pair per module. Every stage is out-of-place, reading
// bs_in + bs_table and writing bs_out, so the caller ping-pongs two buffers and
// never aliases a read-write storage binding. There is no real-input variant:
// rfft packs real to complex with the existing rfft_pack shader first.

const WORKGROUP_SIZE: u32 = 256u;

struct BluesteinParams {
    n: u32,
    m: u32,
    batch_size: u32,
    out_n: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> bs_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> bs_table: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bs_out: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> bs_params: BluesteinParams;

fn bs_cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Stage 1: chirp premultiply into the zero-padded convolution buffer.
// Writes ALL batch_size * m elements, zeroing the tail past n, so the caller
// does not need to pre-clear the buffer.
@compute @workgroup_size(WORKGROUP_SIZE)
fn bluestein_premultiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let b = gid.y;
    let m = bs_params.m;
    let n = bs_params.n;

    if (k >= m || b >= bs_params.batch_size) {
        return;
    }

    let dst = b * m + k;
    if (k < n) {
        bs_out[dst] = bs_cmul(bs_in[b * n + k], bs_table[k]);
    } else {
        bs_out[dst] = vec2<f32>(0.0, 0.0);
    }
}

// Stage 2: pointwise multiply by the kernel spectrum. The kernel depends only
// on (N, direction), so one length-M table serves the whole batch.
@compute @workgroup_size(WORKGROUP_SIZE)
fn bluestein_pointwise_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let b = gid.y;
    let m = bs_params.m;

    if (k >= m || b >= bs_params.batch_size) {
        return;
    }

    let idx = b * m + k;
    bs_out[idx] = bs_cmul(bs_in[idx], bs_table[k]);
}

// Stage 3: chirp postmultiply, crop from m back to out_n, apply scale.
// out_n is n for a full transform and n/2 + 1 for rfft.
@compute @workgroup_size(WORKGROUP_SIZE)
fn bluestein_postmultiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let b = gid.y;
    let out_n = bs_params.out_n;

    if (k >= out_n || b >= bs_params.batch_size) {
        return;
    }

    let v = bs_cmul(bs_table[k], bs_in[b * bs_params.m + k]);
    bs_out[b * out_n + k] = v * bs_params.scale;
}
