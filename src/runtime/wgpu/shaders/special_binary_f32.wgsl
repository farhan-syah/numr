// Special binary functions for f32

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;
const SQRT_PI: f32 = 1.7724538509055159;
const EULER_GAMMA: f32 = 0.5772156649015329;
const LN_SQRT_2PI: f32 = 0.9189385332046727;
const LANCZOS_G: f32 = 7.0;
const MAX_ITER: i32 = 100;
const EPSILON: f32 = 1e-6;
const TINY: f32 = 1e-30;

struct SpecialBinaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> special_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> special_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> special_out: array<f32>;
@group(0) @binding(3) var<uniform> special_params: SpecialBinaryParams;

// ============================================================================
// Helper Functions (shared lgamma)
// ============================================================================

// Lanczos computation for positive x only (no recursion)
fn lgamma_positive(x: f32) -> f32 {
    // Lanczos coefficients (g=7, n=9)
    let c0 = 0.99999999999980993;
    let c1 = 676.5203681218851;
    let c2 = -1259.1392167224028;
    let c3 = 771.32342877765313;
    let c4 = -176.61502916214059;
    let c5 = 12.507343278686905;
    let c6 = -0.13857109526572012;
    let c7 = 9.9843695780195716e-6;
    let c8 = 1.5056327351493116e-7;

    let z = x - 1.0;
    var ag = c0;
    ag = ag + c1 / (z + 1.0);
    ag = ag + c2 / (z + 2.0);
    ag = ag + c3 / (z + 3.0);
    ag = ag + c4 / (z + 4.0);
    ag = ag + c5 / (z + 5.0);
    ag = ag + c6 / (z + 6.0);
    ag = ag + c7 / (z + 7.0);
    ag = ag + c8 / (z + 8.0);

    let t = z + LANCZOS_G + 0.5;
    return LN_SQRT_2PI + (z + 0.5) * log(t) - t + log(ag);
}

// Log-gamma using Lanczos approximation (non-recursive)
fn lgamma_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        // Use reflection formula for negative values
        if (x == floor(x)) {
            return 1e30; // Pole at non-positive integers
        }
        // lgamma(x) = log(pi / sin(pi*x)) - lgamma(1-x)
        // Since 1-x > 0 for x <= 0, we call lgamma_positive directly
        let sinpix = sin(PI * x);
        if (sinpix == 0.0) {
            return 1e30;
        }
        return log(PI / abs(sinpix)) - lgamma_positive(1.0 - x);
    }

    return lgamma_positive(x);
}

// Lower incomplete gamma series
fn gammainc_series(a: f32, x: f32) -> f32 {
    if (x == 0.0) {
        return 0.0;
    }

    var term = 1.0 / a;
    var sum = term;

    for (var n = 1; n < MAX_ITER; n = n + 1) {
        term = term * x / (a + f32(n));
        sum = sum + term;
        if (abs(term) < abs(sum) * EPSILON) {
            break;
        }
    }

    return exp(-x + a * log(x) - lgamma_impl(a)) * sum;
}

// Upper incomplete gamma continued fraction
fn gammaincc_cf(a: f32, x: f32) -> f32 {
    var f = 1e30;
    var c = 1e30;
    var d = 0.0;

    for (var n = 1; n < MAX_ITER; n = n + 1) {
        var an: f32;
        if (n % 2 == 1) {
            an = f32((n + 1) / 2);
        } else {
            an = a - f32(n / 2);
        }
        let bn = x + f32(n) - a;

        d = bn + an * d;
        if (abs(d) < TINY) {
            d = TINY;
        }
        c = bn + an / c;
        if (abs(c) < TINY) {
            c = TINY;
        }

        d = 1.0 / d;
        let delta = c * d;
        f = f * delta;

        if (abs(delta - 1.0) < EPSILON) {
            break;
        }
    }

    return exp(-x + a * log(x) - lgamma_impl(a)) / f;
}

fn gammainc_impl(a: f32, x: f32) -> f32 {
    if (x < 0.0 || a <= 0.0) {
        return bitcast<f32>(0x7FC00000u); // NaN
    }
    if (x == 0.0) {
        return 0.0;
    }
    if (x < a + 1.0) {
        return gammainc_series(a, x);
    }
    return 1.0 - gammaincc_cf(a, x);
}

fn gammaincc_impl(a: f32, x: f32) -> f32 {
    if (x < 0.0 || a <= 0.0) {
        return bitcast<f32>(0x7FC00000u); // NaN
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x < a + 1.0) {
        return 1.0 - gammainc_series(a, x);
    }
    return gammaincc_cf(a, x);
}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn beta_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        let a = special_a[idx];
        let b = special_b[idx];
        special_out[idx] = exp(lgamma_impl(a) + lgamma_impl(b) - lgamma_impl(a + b));
    }
}

@compute @workgroup_size(256)
fn gammainc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = gammainc_impl(special_a[idx], special_b[idx]);
    }
}

@compute @workgroup_size(256)
fn gammaincc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = gammaincc_impl(special_a[idx], special_b[idx]);
    }
}
