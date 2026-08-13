// Special functions for f32
// Algorithms: A&S for erf, Lanczos for gamma, asymptotic for digamma

const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;
const SQRT_PI: f32 = 1.7724538509055159;
const EULER_GAMMA: f32 = 0.5772156649015329;
const LN_SQRT_2PI: f32 = 0.9189385332046727;
const LANCZOS_G: f32 = 7.0;
const MAX_ITER: i32 = 100;
const EPSILON: f32 = 1e-6;
const TINY: f32 = 1e-30;

struct SpecialParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> special_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> special_out: array<f32>;
@group(0) @binding(2) var<uniform> special_params: SpecialParams;

// ============================================================================
// Helper Functions
// ============================================================================

// Error function using Abramowitz & Stegun approximation 7.1.26
fn erf_impl(x: f32) -> f32 {
    if (x == 0.0) {
        return 0.0;
    }

    let sgn = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);

    // Constants for A&S 7.1.26
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-ax * ax);

    return sgn * y;
}

// Complementary error function
fn erfc_impl(x: f32) -> f32 {
    return 1.0 - erf_impl(x);
}

// Inverse error function using rational approximation
fn erfinv_impl(x: f32) -> f32 {
    if (x <= -1.0) {
        return -1e30; // -inf approximation
    }
    if (x >= 1.0) {
        return 1e30; // +inf approximation
    }
    if (x == 0.0) {
        return 0.0;
    }

    let sgn = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);

    // Rational approximation for central region
    if (ax <= 0.7) {
        let x2 = ax * ax;
        let r = ax * ((((-0.140543331 * x2 + 0.914624893) * x2 - 1.645349621) * x2 + 0.886226899) /
                     ((((0.012229801 * x2 - 0.329097515) * x2 + 1.442710462) * x2 - 2.118377725) * x2 + 1.0));
        return sgn * r;
    }

    // Tail approximation
    let z = sqrt(-log((1.0 - ax) / 2.0));
    let r = (((1.641345311 * z + 3.429567803) * z - 1.624906493) * z - 1.970840454) /
            ((1.637067800 * z + 3.543889200) * z + 1.0);
    return sgn * r;
}

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

// Gamma function
fn gamma_impl(x: f32) -> f32 {
    if (x <= 0.0 && x == floor(x)) {
        return 1e30; // Pole
    }
    return exp(lgamma_impl(x));
}

// Digamma for positive x using asymptotic expansion (no recursion)
fn digamma_positive(x: f32) -> f32 {
    var result = 0.0;
    var xx = x;

    // Recurrence to shift to large x where asymptotic works
    while (xx < 6.0) {
        result = result - 1.0 / xx;
        xx = xx + 1.0;
    }

    // Asymptotic expansion
    let x2 = 1.0 / (xx * xx);
    result = result + log(xx) - 0.5 / xx;
    result = result - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0)));

    return result;
}

// Digamma function (non-recursive)
fn digamma_impl(x: f32) -> f32 {
    if (x <= 0.0 && x == floor(x)) {
        return 1e30; // Pole at non-positive integers
    }

    // Reflection formula for negative x (non-recursive)
    if (x < 0.0) {
        // For negative x, 1-x > 0, so we can call digamma_positive directly
        return digamma_positive(1.0 - x) - PI / tan(PI * x);
    }

    return digamma_positive(x);
}

// ============================================================================
// Compute Kernels
// ============================================================================

@compute @workgroup_size(256)
fn erf_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = erf_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn erfc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = erfc_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn erfinv_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = erfinv_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn gamma_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = gamma_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn lgamma_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = lgamma_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn digamma_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = digamma_impl(special_a[idx]);
    }
}

// ============================================================================
// Bessel Functions
// ============================================================================

// J0: Bessel function of the first kind, order 0 (Numerical Recipes style)
fn bessel_j0_impl(x: f32) -> f32 {
    let ax = abs(x);

    if (ax < 8.0) {
        let y = x * x;

        // Numerator polynomial
        let p1 = 57568490574.0;
        let p2 = -13362590354.0;
        let p3 = 651619640.7;
        let p4 = -11214424.18;
        let p5 = 77392.33017;
        let p6 = -184.9052456;

        // Denominator polynomial
        let q1 = 57568490411.0;
        let q2 = 1029532985.0;
        let q3 = 9494680.718;
        let q4 = 59272.64853;
        let q5 = 267.8532712;
        let q6 = 1.0;

        let num = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));

        return num / den;
    } else {
        // Asymptotic expansion
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398163; // ax - π/4

        let p1 = 1.0;
        let p2 = -0.1098628627e-2;
        let p3 = 0.2734510407e-4;
        let p4 = -0.2073370639e-5;
        let p5 = 0.2093887211e-6;

        let q1 = -0.1562499995e-1;
        let q2 = 0.1430488765e-3;
        let q3 = -0.6911147651e-5;
        let q4 = 0.7621095161e-6;
        let q5 = -0.934945152e-7;

        let p0 = p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)));
        let q0 = z * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5))));

        return sqrt(0.636619772 / ax) * (cos(xx) * p0 - sin(xx) * q0);
    }
}

// J1: Bessel function of the first kind, order 1
fn bessel_j1_impl(x: f32) -> f32 {
    let ax = abs(x);

    var result: f32;
    if (ax < 8.0) {
        let y = x * x;

        // Numerator polynomial
        let p1 = 72362614232.0;
        let p2 = -7895059235.0;
        let p3 = 242396853.1;
        let p4 = -2972611.439;
        let p5 = 15704.48260;
        let p6 = -30.16036606;

        // Denominator polynomial
        let q1 = 144725228442.0;
        let q2 = 2300535178.0;
        let q3 = 18583304.74;
        let q4 = 99447.43394;
        let q5 = 376.9991397;
        let q6 = 1.0;

        let num = x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));

        result = num / den;
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194490; // ax - 3π/4

        let p1 = 1.0;
        let p2 = 0.183105e-2;
        let p3 = -0.3516396496e-4;
        let p4 = 0.2457520174e-5;
        let p5 = -0.240337019e-6;

        let q1 = 0.04687499995;
        let q2 = -0.2002690873e-3;
        let q3 = 0.8449199096e-5;
        let q4 = -0.88228987e-6;
        let q5 = 0.105787412e-6;

        let p0 = p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)));
        let q0 = z * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5))));

        let sign = select(-1.0, 1.0, x >= 0.0);
        result = sign * sqrt(0.636619772 / ax) * (cos(xx) * p0 - sin(xx) * q0);
    }

    return result;
}

// Y0: Bessel function of the second kind, order 0 (Numerical Recipes style)
fn bessel_y0_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        return 1e30; // NaN approximation for WGSL
    }

    if (x < 8.0) {
        let y = x * x;

        // Numerator polynomial
        let p1 = -2957821389.0;
        let p2 = 7062834065.0;
        let p3 = -512359803.6;
        let p4 = 10879881.29;
        let p5 = -86327.92757;
        let p6 = 228.4622733;

        // Denominator polynomial
        let q1 = 40076544269.0;
        let q2 = 745249964.8;
        let q3 = 7189466.438;
        let q4 = 47447.26470;
        let q5 = 226.1030244;
        let q6 = 1.0;

        let num = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));

        return num / den + 0.636619772 * bessel_j0_impl(x) * log(x);
    } else {
        // Asymptotic expansion for x >= 8
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785398163; // x - pi/4

        // P0 polynomial (same as J0)
        let p1 = 1.0;
        let p2 = -0.1098628627e-2;
        let p3 = 0.2734510407e-4;
        let p4 = -0.2073370639e-5;
        let p5 = 0.2093887211e-6;

        // Q0 polynomial (same as J0)
        let q1 = -0.1562499995e-1;
        let q2 = 0.1430488765e-3;
        let q3 = -0.6911147651e-5;
        let q4 = 0.7621095161e-6;
        let q5 = -0.934945152e-7;

        let p0 = p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)));
        let q0 = z * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5))));

        return sqrt(0.636619772 / x) * (sin(xx) * p0 + cos(xx) * q0);
    }
}

// Y1: Bessel function of the second kind, order 1 (Numerical Recipes style)
fn bessel_y1_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        return 1e30; // NaN approximation
    }

    if (x < 8.0) {
        let y = x * x;

        // Numerator polynomial (Numerical Recipes coefficients)
        let p1 = -0.4900604943e13;
        let p2 = 0.1275274390e13;
        let p3 = -0.5153438139e11;
        let p4 = 0.7349264551e9;
        let p5 = -0.4237922726e7;
        let p6 = 0.8511937935e4;

        // Denominator polynomial
        let q1 = 0.2499580570e14;
        let q2 = 0.4244198890e12;
        let q3 = 0.3733650367e10;
        let q4 = 0.2245904002e8;
        let q5 = 0.1020426050e6;
        let q6 = 0.3549632885e3;
        let q7 = 1.0;

        let num = x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * (q6 + y * q7)))));

        return num / den + 0.636619772 * (bessel_j1_impl(x) * log(x) - 1.0 / x);
    } else {
        // Asymptotic expansion for x >= 8
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 2.356194490; // x - 3*pi/4

        // P1 polynomial (same as J1)
        let p1 = 1.0;
        let p2 = 0.183105e-2;
        let p3 = -0.3516396496e-4;
        let p4 = 0.2457520174e-5;
        let p5 = -0.240337019e-6;

        // Q1 polynomial (same as J1)
        let q1 = 0.04687499995;
        let q2 = -0.2002690873e-3;
        let q3 = 0.8449199096e-5;
        let q4 = -0.88228987e-6;
        let q5 = 0.105787412e-6;

        let p0 = p1 + y * (p2 + y * (p3 + y * (p4 + y * p5)));
        let q0 = z * (q1 + y * (q2 + y * (q3 + y * (q4 + y * q5))));

        return sqrt(0.636619772 / x) * (sin(xx) * p0 + cos(xx) * q0);
    }
}

// I0: Modified Bessel function of the first kind, order 0
fn bessel_i0_impl(x: f32) -> f32 {
    let ax = abs(x);

    if (ax <= 15.0) {
        // Power series
        let z = ax * ax;
        var sum = 1.0;
        var term = 1.0;

        for (var k = 1; k < 25; k++) {
            let kf = f32(k);
            term = term * z / (4.0 * kf * kf);
            sum = sum + term;
            if (abs(term) < abs(sum) * 1e-7) {
                break;
            }
        }

        return sum;
    } else {
        // Asymptotic expansion
        let z = 1.0 / ax;

        let p0 = 1.0;
        let p1 = 1.25e-01;
        let p2 = 7.03125e-02;
        let p3 = 7.324218750e-02;
        let p4 = 1.1215209960937500e-01;
        let p5 = 2.2710800170898438e-01;

        let poly = ((((p5 * z + p4) * z + p3) * z + p2) * z + p1) * z + p0;

        return exp(ax) / sqrt(2.0 * PI * ax) * poly;
    }
}

// I1: Modified Bessel function of the first kind, order 1
fn bessel_i1_impl(x: f32) -> f32 {
    let ax = abs(x);

    var result: f32;
    if (ax <= 15.0) {
        // Power series
        let z = ax * ax;
        var sum = 0.5;
        var term = 0.5;

        for (var k = 1; k < 25; k++) {
            let kf = f32(k);
            term = term * z / (4.0 * kf * (kf + 1.0));
            sum = sum + term;
            if (abs(term) < abs(sum) * 1e-7) {
                break;
            }
        }

        result = ax * sum;
    } else {
        // Asymptotic expansion
        let z = 1.0 / ax;

        let q0 = 1.0;
        let q1 = -3.75e-01;
        let q2 = -1.171875e-01;
        let q3 = -1.025390625e-01;
        let q4 = -1.4419555664062500e-01;
        let q5 = -2.7757644653320312e-01;

        let poly = ((((q5 * z + q4) * z + q3) * z + q2) * z + q1) * z + q0;

        result = exp(ax) / sqrt(2.0 * PI * ax) * poly;
    }

    // I1 is an odd function
    return select(-result, result, x >= 0.0);
}

// K0: Modified Bessel function of the second kind, order 0
fn bessel_k0_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        return 1e30; // NaN approximation
    }

    if (x <= 2.0) {
        let z = x * x / 4.0;
        let i0 = bessel_i0_impl(x);

        let p0 = -0.57721566;
        let p1 = 0.42278420;
        let p2 = 0.23069756;
        let p3 = 0.03488590;
        let p4 = 0.00262698;
        let p5 = 0.00010750;
        let p6 = 0.00000740;

        let poly = (((((p6 * z + p5) * z + p4) * z + p3) * z + p2) * z + p1) * z + p0;

        return -log(x / 2.0) * i0 + poly;
    } else {
        let z = 2.0 / x;

        let p0 = 1.25331414;
        let p1 = -0.07832358;
        let p2 = 0.02189568;
        let p3 = -0.01062446;
        let p4 = 0.00587872;
        let p5 = -0.00251540;
        let p6 = 0.00053208;

        let poly = (((((p6 * z + p5) * z + p4) * z + p3) * z + p2) * z + p1) * z + p0;

        return exp(-x) / sqrt(x) * poly;
    }
}

// K1: Modified Bessel function of the second kind, order 1
fn bessel_k1_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        return 1e30; // NaN approximation
    }

    if (x <= 2.0) {
        let z = x * x / 4.0;
        let i1 = bessel_i1_impl(x);

        let p0 = 1.0;
        let p1 = 0.15443144;
        let p2 = -0.67278579;
        let p3 = -0.18156897;
        let p4 = -0.01919402;
        let p5 = -0.00110404;
        let p6 = -0.00004686;

        let poly = x * (((((p6 * z + p5) * z + p4) * z + p3) * z + p2) * z + p1) * z + p0;

        return log(x / 2.0) * i1 + poly / x;
    } else {
        let z = 2.0 / x;

        let q0 = 1.25331414;
        let q1 = 0.23498619;
        let q2 = -0.03655620;
        let q3 = 0.01504268;
        let q4 = -0.00780353;
        let q5 = 0.00325614;
        let q6 = -0.00068245;

        let poly = (((((q6 * z + q5) * z + q4) * z + q3) * z + q2) * z + q1) * z + q0;

        return exp(-x) / sqrt(x) * poly;
    }
}

@compute @workgroup_size(256)
fn bessel_j0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_j0_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_j1_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_j1_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_y0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_y0_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_y1_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_y1_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_i0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_i0_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_i1_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_i1_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_k0_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_k0_impl(special_a[idx]);
    }
}

@compute @workgroup_size(256)
fn bessel_k1_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < special_params.numel) {
        special_out[idx] = bessel_k1_impl(special_a[idx]);
    }
}
