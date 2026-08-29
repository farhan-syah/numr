// U32 scalar operations

const WORKGROUP_SIZE: u32 = 256u;

struct ScalarParams {
    numel: u32,
    scalar: u32,
}

@group(0) @binding(0) var<storage, read_write> scalar_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> scalar_out: array<u32>;
@group(0) @binding(2) var<uniform> scalar_params: ScalarParams;

@compute @workgroup_size(256)
fn add_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] + scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn sub_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] - scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn rsub_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_params.scalar - scalar_a[idx];
    }
}

@compute @workgroup_size(256)
fn mul_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] * scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn div_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = scalar_a[idx] / scalar_params.scalar;
    }
}

@compute @workgroup_size(256)
fn pow_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < scalar_params.numel) {
        scalar_out[idx] = numr_ipow_u32(scalar_a[idx], scalar_params.scalar);
    }
}
