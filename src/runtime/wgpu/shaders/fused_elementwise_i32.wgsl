// Fused elementwise WGSL shaders (I32)
// fused_mul_add: out = a * b + c
// fused_add_mul: out = (a + b) * c
//
// The two arithmetic operators are the plain ones, exactly as binary_i32.wgsl
// writes them, because an integer fused op must answer what the unfused
// sequence answers element for element - including at the wrap boundary, where
// CPU's `binary_int_fused_elem` wraps at each step rather than saturating once
// (runtime/cpu/kernels/binary_int.rs). There is no `fma` here: an fma would
// change nothing for integers, and the f32 shader's rounding argument does not
// apply.

struct TernaryParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> tern_a: array<i32>;
@group(0) @binding(1) var<storage, read_write> tern_b: array<i32>;
@group(0) @binding(2) var<storage, read_write> tern_c: array<i32>;
@group(0) @binding(3) var<storage, read_write> tern_out: array<i32>;
@group(0) @binding(4) var<uniform> tern_params: TernaryParams;

@compute @workgroup_size(256)
fn fused_mul_add_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < tern_params.numel) {
        tern_out[idx] = tern_a[idx] * tern_b[idx] + tern_c[idx];
    }
}

@compute @workgroup_size(256)
fn fused_add_mul_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < tern_params.numel) {
        tern_out[idx] = (tern_a[idx] + tern_b[idx]) * tern_c[idx];
    }
}
