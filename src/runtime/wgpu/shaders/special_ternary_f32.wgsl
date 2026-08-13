// Special ternary functions for f32

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;
const SQRT_PI: f32 = 1.7724538509055159;
const EULER_GAMMA: f32 = 0.5772156649015329;
const LN_SQRT_2PI: f32 = 0.9189385332046727;
const LANCZOS_G: f32 = 7.0;
const MAX_ITER: i32 = 100;
const EPSILON: f32 = 1e-6;
const TINY: f32 = 1e-30;

struct SpecialTernaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> special_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> special_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> special_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> special_out: array<f32>;
@group(0) @binding(4) var<uniform> special_params: SpecialTernaryParams;

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

// Regularized incomplete beta using continued fraction
fn betainc_cf(a: f32, b: f32, x: f32) -> f32 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    var c = 1.0;
    var d = 1.0 - qab * x / qap;
    if (abs(d) < TINY) {
        d = TINY;
    }
    d = 1.0 / d;
    var h = d;

    for (var m = 1; m < MAX_ITER; m = m + 1) {
        let m2 = 2 * m;

        var aa = f32(m) * (b - f32(m)) * x / ((qam + f32(m2)) * (a + f32(m2)));
        d = 1.0 + aa * d;
        if (abs(d) < TINY) {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if (abs(c) < TINY) {
            c = TINY;
        }
        d = 1.0 / d;
        h = h * d * c;

        aa = -(a + f32(m)) * (qab + f32(m)) * x / ((a + f32(m2)) * (qap + f32(m2)));
        d = 1.0 + aa * d;
        if (abs(d) < TINY) {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if (abs(c) < TINY) {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h = h * delta;

        if (abs(delta - 1.0) < EPSILON) {
            break;
        }
    }

    let lnbeta = lgamma_impl(a) + lgamma_impl(b) - lgamma_impl(a + b);
    return exp(a * log(x) + b * log(1.0 - x) - lnbeta) * h / a;
}

fn betainc_impl(a: f32, b: f32, x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }

    // Use symmetry for better convergence (non-recursive version)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        // Compute directly without recursion using symmetry
        return 1.0 - betainc_cf(b, a, 1.0 - x);
    }

    return betainc_cf(a, b, x);
}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn betainc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = betainc_impl(special_a[idx], special_b[idx], special_x[idx]);
    }
}
