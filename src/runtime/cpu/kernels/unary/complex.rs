//! Complex number unary operation kernels
//!
//! Provides element-wise unary operations for Complex64 and Complex128 types,
//! operating on interleaved (re, im) pairs.

use crate::ops::UnaryOp;

/// Complex64 unary operations operating on (re, im) pairs stored as f32
/// `len` is the number of complex elements; memory has `2*len` f32 values
#[inline]
pub(super) unsafe fn unary_op_complex64(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let a_pairs = std::slice::from_raw_parts(a, len * 2);
    let out_pairs = std::slice::from_raw_parts_mut(out, len * 2);

    for i in 0..len {
        let re = a_pairs[2 * i] as f64;
        let im = a_pairs[2 * i + 1] as f64;
        let (ore, oim) = complex_unary_op(op, re, im);
        out_pairs[2 * i] = ore as f32;
        out_pairs[2 * i + 1] = oim as f32;
    }
}

/// Complex128 unary operations operating on (re, im) pairs stored as f64
#[inline]
pub(super) unsafe fn unary_op_complex128(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let a_pairs = std::slice::from_raw_parts(a, len * 2);
    let out_pairs = std::slice::from_raw_parts_mut(out, len * 2);

    for i in 0..len {
        let re = a_pairs[2 * i];
        let im = a_pairs[2 * i + 1];
        let (ore, oim) = complex_unary_op(op, re, im);
        out_pairs[2 * i] = ore;
        out_pairs[2 * i + 1] = oim;
    }
}

/// Perform a unary operation on a complex number (re, im) → (re', im')
#[inline]
fn complex_unary_op(op: UnaryOp, re: f64, im: f64) -> (f64, f64) {
    match op {
        UnaryOp::Neg => (-re, -im),
        UnaryOp::Abs => {
            // |z| as a real complex number
            ((re * re + im * im).sqrt(), 0.0)
        }
        UnaryOp::Square => {
            // (a+bi)^2 = a^2-b^2 + 2abi
            (re * re - im * im, 2.0 * re * im)
        }
        UnaryOp::Recip => {
            // 1/(a+bi) = (a-bi)/(a^2+b^2)
            let denom = re * re + im * im;
            if denom == 0.0 {
                (f64::NAN, f64::NAN)
            } else {
                (re / denom, -im / denom)
            }
        }
        UnaryOp::Exp => {
            // e^(a+bi) = e^a * (cos(b) + i*sin(b))
            let ea = re.exp();
            (ea * im.cos(), ea * im.sin())
        }
        UnaryOp::Log => {
            // log(a+bi) = log|z| + i*arg(z)
            let abs = (re * re + im * im).sqrt();
            (abs.ln(), im.atan2(re))
        }
        UnaryOp::Sqrt => {
            // sqrt(a+bi) using polar form
            let abs = (re * re + im * im).sqrt();
            let r = ((abs + re) / 2.0).sqrt();
            let i_val = ((abs - re) / 2.0).sqrt();
            if im >= 0.0 { (r, i_val) } else { (r, -i_val) }
        }
        UnaryOp::Sign => {
            let abs = (re * re + im * im).sqrt();
            if abs == 0.0 {
                (0.0, 0.0)
            } else {
                (re / abs, im / abs)
            }
        }
        // For operations that don't have natural complex extensions,
        // apply to magnitude and return as real
        _ => {
            let mag = (re * re + im * im).sqrt();
            let result = match op {
                UnaryOp::Floor => mag.floor(),
                UnaryOp::Ceil => mag.ceil(),
                UnaryOp::Round => mag.round(),
                UnaryOp::RoundTiesEven => mag.round_ties_even(),
                UnaryOp::Trunc => mag.trunc(),
                UnaryOp::Rsqrt => 1.0 / mag.sqrt(),
                UnaryOp::Cbrt => mag.cbrt(),
                UnaryOp::Sin => {
                    // sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
                    return (re.sin() * im.cosh(), re.cos() * im.sinh());
                }
                UnaryOp::Cos => {
                    // cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
                    return (re.cos() * im.cosh(), -re.sin() * im.sinh());
                }
                UnaryOp::Tan => {
                    // tan(z) = sin(z)/cos(z)
                    let (sr, si) = (re.sin() * im.cosh(), re.cos() * im.sinh());
                    let (cr, ci) = (re.cos() * im.cosh(), -re.sin() * im.sinh());
                    let denom = cr * cr + ci * ci;
                    if denom == 0.0 {
                        return (f64::NAN, f64::NAN);
                    }
                    return ((sr * cr + si * ci) / denom, (si * cr - sr * ci) / denom);
                }
                UnaryOp::Tanh => {
                    // tanh(z) = sinh(z)/cosh(z)
                    let (sr, si) = (re.sinh() * im.cos(), re.cosh() * im.sin());
                    let (cr, ci) = (re.cosh() * im.cos(), re.sinh() * im.sin());
                    let denom = cr * cr + ci * ci;
                    if denom == 0.0 {
                        return (f64::NAN, f64::NAN);
                    }
                    return ((sr * cr + si * ci) / denom, (si * cr - sr * ci) / denom);
                }
                UnaryOp::Sinh => {
                    return (re.sinh() * im.cos(), re.cosh() * im.sin());
                }
                UnaryOp::Cosh => {
                    return (re.cosh() * im.cos(), re.sinh() * im.sin());
                }
                UnaryOp::Exp2 => {
                    // 2^(a+bi) = 2^a * (cos(b*ln2) + i*sin(b*ln2))
                    let ln2 = std::f64::consts::LN_2;
                    let ea = (re * ln2).exp();
                    return (ea * (im * ln2).cos(), ea * (im * ln2).sin());
                }
                UnaryOp::Expm1 => {
                    // e^(a+bi) - 1
                    let ea = re.exp();
                    return (ea * im.cos() - 1.0, ea * im.sin());
                }
                UnaryOp::Log2 => {
                    // log2(z) = log(z) / ln(2)
                    let ln2 = std::f64::consts::LN_2;
                    let abs = (re * re + im * im).sqrt();
                    return (abs.ln() / ln2, im.atan2(re) / ln2);
                }
                UnaryOp::Log10 => {
                    // log10(z) = log(z) / ln(10)
                    let ln10 = std::f64::consts::LN_10;
                    let abs = (re * re + im * im).sqrt();
                    return (abs.ln() / ln10, im.atan2(re) / ln10);
                }
                UnaryOp::Log1p => {
                    // log(1 + z) = log((1+re) + im*i)
                    let new_re = 1.0 + re;
                    let abs = (new_re * new_re + im * im).sqrt();
                    return (abs.ln(), im.atan2(new_re));
                }
                UnaryOp::Asin => {
                    // asin(z) = -i * log(iz + sqrt(1 - z^2))
                    let (z2r, z2i) = (re * re - im * im, 2.0 * re * im);
                    let (sr, si) = (1.0 - z2r, -z2i);
                    let abs_s = (sr * sr + si * si).sqrt();
                    let (sqr, sqi) = (((abs_s + sr) / 2.0).sqrt(), ((abs_s - sr) / 2.0).sqrt());
                    let sqi = if si >= 0.0 { sqi } else { -sqi };
                    let (wr, wi) = (-im + sqr, re + sqi);
                    let abs_w = (wr * wr + wi * wi).sqrt();
                    return (wi.atan2(wr), -abs_w.ln());
                }
                UnaryOp::Acos => {
                    // acos(z) = pi/2 - asin(z)
                    let (z2r, z2i) = (re * re - im * im, 2.0 * re * im);
                    let (sr, si) = (1.0 - z2r, -z2i);
                    let abs_s = (sr * sr + si * si).sqrt();
                    let (sqr, sqi) = (((abs_s + sr) / 2.0).sqrt(), ((abs_s - sr) / 2.0).sqrt());
                    let sqi = if si >= 0.0 { sqi } else { -sqi };
                    let (wr, wi) = (-im + sqr, re + sqi);
                    let abs_w = (wr * wr + wi * wi).sqrt();
                    let asin_re = wi.atan2(wr);
                    let asin_im = -abs_w.ln();
                    return (std::f64::consts::FRAC_PI_2 - asin_re, -asin_im);
                }
                UnaryOp::Atan => {
                    // atan(z) = i/2 * log((1-iz)/(1+iz))
                    let (nr, ni) = (1.0 + im, -re);
                    let (dr, di) = (1.0 - im, re);
                    let dn = dr * dr + di * di;
                    if dn == 0.0 {
                        return (f64::NAN, f64::NAN);
                    }
                    let (qr, qi) = ((nr * dr + ni * di) / dn, (ni * dr - nr * di) / dn);
                    let abs_q = (qr * qr + qi * qi).sqrt();
                    return (-0.5 * qi.atan2(qr), 0.5 * abs_q.ln());
                }
                UnaryOp::Asinh => {
                    // asinh(z) = log(z + sqrt(z^2 + 1))
                    let (z2r, z2i) = (re * re - im * im + 1.0, 2.0 * re * im);
                    let abs_z2 = (z2r * z2r + z2i * z2i).sqrt();
                    let (sqr, sqi) = (((abs_z2 + z2r) / 2.0).sqrt(), ((abs_z2 - z2r) / 2.0).sqrt());
                    let sqi = if z2i >= 0.0 { sqi } else { -sqi };
                    let (wr, wi) = (re + sqr, im + sqi);
                    let abs_w = (wr * wr + wi * wi).sqrt();
                    return (abs_w.ln(), wi.atan2(wr));
                }
                UnaryOp::Acosh => {
                    // acosh(z) = log(z + sqrt(z^2 - 1))
                    let (z2r, z2i) = (re * re - im * im - 1.0, 2.0 * re * im);
                    let abs_z2 = (z2r * z2r + z2i * z2i).sqrt();
                    let (sqr, sqi) = (((abs_z2 + z2r) / 2.0).sqrt(), ((abs_z2 - z2r) / 2.0).sqrt());
                    let sqi = if z2i >= 0.0 { sqi } else { -sqi };
                    let (wr, wi) = (re + sqr, im + sqi);
                    let abs_w = (wr * wr + wi * wi).sqrt();
                    return (abs_w.ln(), wi.atan2(wr));
                }
                UnaryOp::Atanh => {
                    // atanh(z) = 1/2 * log((1+z)/(1-z))
                    let (nr, ni) = (1.0 + re, im);
                    let (dr, di) = (1.0 - re, -im);
                    let dn = dr * dr + di * di;
                    if dn == 0.0 {
                        return (f64::NAN, f64::NAN);
                    }
                    let (qr, qi) = ((nr * dr + ni * di) / dn, (ni * dr - nr * di) / dn);
                    let abs_q = (qr * qr + qi * qi).sqrt();
                    return (0.5 * abs_q.ln(), 0.5 * qi.atan2(qr));
                }
                // Rounding ops: no meaningful complex extension, return NaN
                _ => return (f64::NAN, f64::NAN),
            };
            (result, 0.0)
        }
    }
}
