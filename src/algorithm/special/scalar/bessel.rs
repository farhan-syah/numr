//! Bessel function implementations (J0, J1, Y0, Y1, I0, I1, K0, K1)
//!
//! Uses Numerical Recipes style polynomial approximations with:
//! - Rational polynomial approximation for small arguments
//! - Asymptotic expansion for large arguments

use crate::algorithm::special::bessel_coefficients::*;

// ============================================================================
// Bessel Functions of the First Kind
// ============================================================================

/// Compute J₀(x) using polynomial approximation (Numerical Recipes style).
///
/// Uses power series for small |x|, asymptotic expansion for large |x|.
pub fn bessel_j0_scalar(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 8.0 {
        // Polynomial approximation for |x| < 8
        let y = x * x;

        let num = J0_SMALL_P[0]
            + y * (J0_SMALL_P[1]
                + y * (J0_SMALL_P[2]
                    + y * (J0_SMALL_P[3] + y * (J0_SMALL_P[4] + y * J0_SMALL_P[5]))));
        let den = J0_SMALL_Q[0]
            + y * (J0_SMALL_Q[1]
                + y * (J0_SMALL_Q[2]
                    + y * (J0_SMALL_Q[3] + y * (J0_SMALL_Q[4] + y * J0_SMALL_Q[5]))));

        num / den
    } else {
        // Asymptotic expansion for |x| >= 8
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - FRAC_PI_4;

        let p0 = J0_ASYMP_P[0]
            + y * (J0_ASYMP_P[1] + y * (J0_ASYMP_P[2] + y * (J0_ASYMP_P[3] + y * J0_ASYMP_P[4])));
        let q0 = z
            * (J0_ASYMP_Q[0]
                + y * (J0_ASYMP_Q[1]
                    + y * (J0_ASYMP_Q[2] + y * (J0_ASYMP_Q[3] + y * J0_ASYMP_Q[4]))));

        (TWO_OVER_PI / ax).sqrt() * (xx.cos() * p0 - xx.sin() * q0)
    }
}

/// Compute J₁(x) using polynomial approximation.
pub fn bessel_j1_scalar(x: f64) -> f64 {
    let ax = x.abs();

    let result = if ax < 8.0 {
        // Polynomial approximation for |x| < 8
        let y = x * x;

        let num = x
            * (J1_SMALL_P[0]
                + y * (J1_SMALL_P[1]
                    + y * (J1_SMALL_P[2]
                        + y * (J1_SMALL_P[3] + y * (J1_SMALL_P[4] + y * J1_SMALL_P[5])))));
        let den = J1_SMALL_Q[0]
            + y * (J1_SMALL_Q[1]
                + y * (J1_SMALL_Q[2]
                    + y * (J1_SMALL_Q[3] + y * (J1_SMALL_Q[4] + y * J1_SMALL_Q[5]))));

        num / den
    } else {
        // Asymptotic expansion for |x| >= 8
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - FRAC_3PI_4;

        let p0 = J1_ASYMP_P[0]
            + y * (J1_ASYMP_P[1] + y * (J1_ASYMP_P[2] + y * (J1_ASYMP_P[3] + y * J1_ASYMP_P[4])));
        let q0 = z
            * (J1_ASYMP_Q[0]
                + y * (J1_ASYMP_Q[1]
                    + y * (J1_ASYMP_Q[2] + y * (J1_ASYMP_Q[3] + y * J1_ASYMP_Q[4]))));

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (TWO_OVER_PI / ax).sqrt() * (xx.cos() * p0 - xx.sin() * q0)
    };

    result
}

// ============================================================================
// Bessel Functions of the Second Kind (Neumann Functions)
// ============================================================================

/// Compute Y₀(x) using polynomial approximation.
///
/// Y₀(x) is singular at x = 0, returns NaN for x ≤ 0.
pub fn bessel_y0_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x < 8.0 {
        let y = x * x;

        let num = Y0_SMALL_P[0]
            + y * (Y0_SMALL_P[1]
                + y * (Y0_SMALL_P[2]
                    + y * (Y0_SMALL_P[3] + y * (Y0_SMALL_P[4] + y * Y0_SMALL_P[5]))));
        let den = Y0_SMALL_Q[0]
            + y * (Y0_SMALL_Q[1]
                + y * (Y0_SMALL_Q[2]
                    + y * (Y0_SMALL_Q[3] + y * (Y0_SMALL_Q[4] + y * Y0_SMALL_Q[5]))));

        num / den + TWO_OVER_PI * bessel_j0_scalar(x) * x.ln()
    } else {
        // Asymptotic expansion for x >= 8
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - FRAC_PI_4;

        // Use same P0/Q0 polynomials as J0
        let p0 = J0_ASYMP_P[0]
            + y * (J0_ASYMP_P[1] + y * (J0_ASYMP_P[2] + y * (J0_ASYMP_P[3] + y * J0_ASYMP_P[4])));
        let q0 = z
            * (J0_ASYMP_Q[0]
                + y * (J0_ASYMP_Q[1]
                    + y * (J0_ASYMP_Q[2] + y * (J0_ASYMP_Q[3] + y * J0_ASYMP_Q[4]))));

        (TWO_OVER_PI / x).sqrt() * (xx.sin() * p0 + xx.cos() * q0)
    }
}

/// Compute Y₁(x) using polynomial approximation.
///
/// Y₁(x) is singular at x = 0, returns NaN for x ≤ 0.
pub fn bessel_y1_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x < 8.0 {
        let y = x * x;

        let num = x
            * (Y1_SMALL_P[0]
                + y * (Y1_SMALL_P[1]
                    + y * (Y1_SMALL_P[2]
                        + y * (Y1_SMALL_P[3] + y * (Y1_SMALL_P[4] + y * Y1_SMALL_P[5])))));
        let den = Y1_SMALL_Q[0]
            + y * (Y1_SMALL_Q[1]
                + y * (Y1_SMALL_Q[2]
                    + y * (Y1_SMALL_Q[3]
                        + y * (Y1_SMALL_Q[4] + y * (Y1_SMALL_Q[5] + y * Y1_SMALL_Q[6])))));

        num / den + TWO_OVER_PI * (bessel_j1_scalar(x) * x.ln() - 1.0 / x)
    } else {
        // Asymptotic expansion for x >= 8
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - FRAC_3PI_4;

        // Use same P1/Q1 polynomials as J1
        let p0 = J1_ASYMP_P[0]
            + y * (J1_ASYMP_P[1] + y * (J1_ASYMP_P[2] + y * (J1_ASYMP_P[3] + y * J1_ASYMP_P[4])));
        let q0 = z
            * (J1_ASYMP_Q[0]
                + y * (J1_ASYMP_Q[1]
                    + y * (J1_ASYMP_Q[2] + y * (J1_ASYMP_Q[3] + y * J1_ASYMP_Q[4]))));

        (TWO_OVER_PI / x).sqrt() * (xx.sin() * p0 + xx.cos() * q0)
    }
}

// ============================================================================
// Modified Bessel Functions of the First Kind
// ============================================================================

/// Compute I₀(x) using polynomial approximation.
///
/// I₀(x) is always positive and grows exponentially for large |x|.
pub fn bessel_i0_scalar(x: f64) -> f64 {
    let ax = x.abs();

    if ax <= 15.0 {
        // Power series approximation for moderate arguments
        let z = ax * ax;

        // Direct power series: I0(x) = sum_{k=0}^∞ (x/2)^{2k} / (k!)^2
        let mut sum = 1.0;
        let mut term = 1.0;

        for k in 1..30 {
            term *= z / (4.0 * (k as f64) * (k as f64));
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }

        sum
    } else {
        // Asymptotic expansion: I0(x) ≈ exp(x) / sqrt(2πx) * P(1/x)
        let z = 1.0 / ax;

        let poly = (((((I0_ASYMP[6] * z + I0_ASYMP[5]) * z + I0_ASYMP[4]) * z + I0_ASYMP[3]) * z
            + I0_ASYMP[2])
            * z
            + I0_ASYMP[1])
            * z
            + I0_ASYMP[0];

        // exp overflows past ln(f64::MAX) = 709.7827 while I0 stays finite up
        // to 713.9869, so forming exp(ax) first returns infinity over that
        // whole band. t = exp(ax/2) keeps every factor representable; ax/2 is
        // exact, and the product still overflows to infinity past 713.9869.
        let t = (0.5 * ax).exp();
        t / (2.0 * std::f64::consts::PI * ax).sqrt() * t * poly
    }
}

/// Compute I₁(x) using polynomial approximation.
///
/// I₁(x) is odd: I₁(-x) = -I₁(x).
pub fn bessel_i1_scalar(x: f64) -> f64 {
    let ax = x.abs();

    let result = if ax <= 15.0 {
        // Power series: I1(x) = (x/2) * sum_{k=0}^∞ (x/2)^{2k} / (k!(k+1)!)
        let z = ax * ax;
        let mut sum = 0.5;
        let mut term = 0.5;

        for k in 1..30 {
            term *= z / (4.0 * (k as f64) * ((k + 1) as f64));
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }

        ax * sum
    } else {
        // Asymptotic expansion: I1(x) ≈ exp(x) / sqrt(2πx) * Q(1/x)
        let z = 1.0 / ax;

        let poly = (((((I1_ASYMP[6] * z + I1_ASYMP[5]) * z + I1_ASYMP[4]) * z + I1_ASYMP[3]) * z
            + I1_ASYMP[2])
            * z
            + I1_ASYMP[1])
            * z
            + I1_ASYMP[0];

        // exp overflows past ln(f64::MAX) = 709.7827 while I1 stays finite up
        // to 713.9876, so forming exp(ax) first returns infinity over that
        // whole band. t = exp(ax/2) keeps every factor representable; ax/2 is
        // exact, and the product still overflows to infinity past 713.9876.
        let t = (0.5 * ax).exp();
        t / (2.0 * std::f64::consts::PI * ax).sqrt() * t * poly
    };

    // I1 is an odd function
    if x < 0.0 { -result } else { result }
}

// ============================================================================
// Modified Bessel Functions of the Second Kind
// ============================================================================

/// Compute K₀(x) using polynomial approximation.
///
/// K₀(x) is defined only for x > 0, singular at x = 0.
pub fn bessel_k0_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x <= 2.0 {
        // For small x: K0(x) = -ln(x/2) * I0(x) + polynomial
        let z = x * x / 4.0;

        let i0 = bessel_i0_scalar(x);
        let poly = (((((K0_SMALL[6] * z + K0_SMALL[5]) * z + K0_SMALL[4]) * z + K0_SMALL[3]) * z
            + K0_SMALL[2])
            * z
            + K0_SMALL[1])
            * z
            + K0_SMALL[0];

        -(x / 2.0).ln() * i0 + poly
    } else {
        // Asymptotic expansion: K0(x) ≈ sqrt(π/(2x)) * exp(-x) * P(1/x)
        let z = 2.0 / x;

        let poly = (((((K0_LARGE[6] * z + K0_LARGE[5]) * z + K0_LARGE[4]) * z + K0_LARGE[3]) * z
            + K0_LARGE[2])
            * z
            + K0_LARGE[1])
            * z
            + K0_LARGE[0];

        (-x).exp() / x.sqrt() * poly
    }
}

/// Compute K₁(x) using polynomial approximation.
///
/// K₁(x) is defined only for x > 0, singular at x = 0.
pub fn bessel_k1_scalar(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    if x <= 2.0 {
        // For small x: K1(x) = ln(x/2)*I1(x) + (1/x)*P(x^2/4)
        let y = x * x / 4.0;

        let i1 = bessel_i1_scalar(x);
        let poly = K1_SMALL[0]
            + y * (K1_SMALL[1]
                + y * (K1_SMALL[2]
                    + y * (K1_SMALL[3] + y * (K1_SMALL[4] + y * (K1_SMALL[5] + y * K1_SMALL[6])))));

        (x / 2.0).ln() * i1 + poly / x
    } else {
        // Asymptotic expansion for x > 2
        let y = 2.0 / x;

        let poly = K1_LARGE[0]
            + y * (K1_LARGE[1]
                + y * (K1_LARGE[2]
                    + y * (K1_LARGE[3] + y * (K1_LARGE[4] + y * (K1_LARGE[5] + y * K1_LARGE[6])))));

        (-x).exp() / x.sqrt() * poly
    }
}
