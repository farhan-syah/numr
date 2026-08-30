// U32 unary operations

const WORKGROUP_SIZE: u32 = 256u;

struct UnaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> unary_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<u32>;
@group(0) @binding(2) var<uniform> unary_params: UnaryParams;

// `-x` does not compile for a u32 in WGSL. `0u - x` does, and it is defined
// wrapping there, which is the element-wise integer convention this crate uses
// (see runtime/cpu/kernels/wide_acc.rs). So neg(1u) is u32::MAX.
@compute @workgroup_size(256)
fn neg_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = 0u - unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn abs_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}

// `sign` on an unsigned dtype has no negative branch: 0 for 0, 1 otherwise.
// This matches the CPU and CUDA backends.
@compute @workgroup_size(256)
fn sign_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        if (unary_a[idx] == 0u) {
            unary_out[idx] = 0u;
        } else {
            unary_out[idx] = 1u;
        }
    }
}

// floor/ceil/round/round_ties_even/trunc are the identity on an integer: every
// U32 value is already its own nearest integer. This matches CPU and CUDA.
@compute @workgroup_size(256)
fn floor_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn ceil_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn round_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn round_ties_even_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}

@compute @workgroup_size(256)
fn trunc_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < unary_params.numel) {
        unary_out[idx] = unary_a[idx];
    }
}
