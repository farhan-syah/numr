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
