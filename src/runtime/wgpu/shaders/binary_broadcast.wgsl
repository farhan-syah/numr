// F32 broadcast binary operations

const WORKGROUP_SIZE: u32 = 256u;

struct BroadcastBinaryParams {
    numel: u32,
    ndim: u32,
}

@group(0) @binding(0) var<storage, read_write> broadcast_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> broadcast_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> broadcast_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> broadcast_a_strides: array<u32>;
@group(0) @binding(4) var<storage, read_write> broadcast_b_strides: array<u32>;
@group(0) @binding(5) var<storage, read_write> broadcast_out_strides: array<u32>;
@group(0) @binding(6) var<uniform> broadcast_params: BroadcastBinaryParams;

@compute @workgroup_size(256)
fn broadcast_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = broadcast_a[a_offset] + broadcast_b[b_offset];
}

@compute @workgroup_size(256)
fn broadcast_sub_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = broadcast_a[a_offset] - broadcast_b[b_offset];
}

@compute @workgroup_size(256)
fn broadcast_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = broadcast_a[a_offset] * broadcast_b[b_offset];
}

@compute @workgroup_size(256)
fn broadcast_div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = broadcast_a[a_offset] / broadcast_b[b_offset];
}

@compute @workgroup_size(256)
fn broadcast_max_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = max(broadcast_a[a_offset], broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_min_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = min(broadcast_a[a_offset], broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_pow_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = pow(broadcast_a[a_offset], broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_eq_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] == broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_ne_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] != broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_lt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] < broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_le_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] <= broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_gt_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] > broadcast_b[b_offset]);
}

@compute @workgroup_size(256)
fn broadcast_ge_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broadcast_params.numel) {
        return;
    }

    var remaining = idx;
    var a_offset: u32 = 0u;
    var b_offset: u32 = 0u;

    for (var d: u32 = 0u; d < broadcast_params.ndim; d = d + 1u) {
        let stride = broadcast_out_strides[d];
        let coord = remaining / stride;
        remaining = remaining % stride;

        a_offset = a_offset + coord * broadcast_a_strides[d];
        b_offset = b_offset + coord * broadcast_b_strides[d];
    }

    broadcast_out[idx] = select(0.0, 1.0, broadcast_a[a_offset] >= broadcast_b[b_offset]);
}
