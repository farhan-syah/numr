// F32 unary operations

const WORKGROUP_SIZE: u32 = 256u;

struct UnaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> unary_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<f32>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

@compute @workgroup_size(256)
fn neg_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = -unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn abs_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = abs(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn sqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = sqrt(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn exp_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = exp(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn log_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = log(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn sin_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = sin(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn cos_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = cos(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn tan_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = tan(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn atan_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = atan(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn tanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = tanh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn recip_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = 1.0 / unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn floor_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = floor(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn ceil_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = ceil(unary_a[idx]);
    }
}

// Ties away from zero, matching Rust's f32::round. The WGSL builtin round()
// breaks ties to even, so it cannot be used here.
@compute @workgroup_size(256)
fn round_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        unary_out[idx] = select(ceil(x - 0.5), floor(x + 0.5), x >= 0.0);
    }
}

// Ties to even. The WGSL builtin round() is specified as roundTiesToEven, which
// is exactly IEEE 754 roundTiesToEven and matches np.round / torch.round.
@compute @workgroup_size(256)
fn round_ties_even_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = round(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn trunc_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = trunc(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn rsqrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = inverseSqrt(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn cbrt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        unary_out[idx] = sign(x) * pow(abs(x), 1.0 / 3.0);
    }
}

@compute @workgroup_size(256)
fn exp2_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = exp2(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn expm1_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = exp(unary_a[idx]) - 1.0;
    }
}

@compute @workgroup_size(256)
fn log2_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = log2(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn log10_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = log(unary_a[idx]) * 0.4342944819032518;
    }
}

@compute @workgroup_size(256)
fn log1p_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = log(1.0 + unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn asin_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        let y = sqrt(max(0.0, 1.0 - x * x));
        unary_out[idx] = atan2(x, y);
    }
}

@compute @workgroup_size(256)
fn acos_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        let y = sqrt(max(0.0, 1.0 - x * x));
        unary_out[idx] = atan2(y, x);
    }
}

@compute @workgroup_size(256)
fn sinh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = sinh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn cosh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = cosh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn asinh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = asinh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn acosh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = acosh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn atanh_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = atanh(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn square_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        unary_out[idx] = x * x;
    }
}

@compute @workgroup_size(256)
fn sign_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = sign(unary_a[idx]);
    }
}

@compute @workgroup_size(256)
fn relu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = max(unary_a[idx], 0.0);
    }
}

@compute @workgroup_size(256)
fn sigmoid_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = 1.0 / (1.0 + exp(-unary_a[idx]));
    }
}

@compute @workgroup_size(256)
fn silu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        unary_out[idx] = x / (1.0 + exp(-x));
    }
}

@compute @workgroup_size(256)
fn gelu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        let c = 0.7978845608028654;
        unary_out[idx] = 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
    }
}

@compute @workgroup_size(256)
fn isnan_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        let bits = bitcast<u32>(f32(x));
        let exp = bits & 0x7f800000u;
        let mant = bits & 0x007fffffu;
        let is_nan = (exp == 0x7f800000u) && (mant != 0u);
        unary_out[idx] = select(0.0, 1.0, is_nan);
    }
}

@compute @workgroup_size(256)
fn isinf_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        let x = unary_a[idx];
        let bits = bitcast<u32>(f32(x));
        let exp = bits & 0x7f800000u;
        let mant = bits & 0x007fffffu;
        let is_inf = (exp == 0x7f800000u) && (mant == 0u);
        unary_out[idx] = select(0.0, 1.0, is_inf);
    }
}
