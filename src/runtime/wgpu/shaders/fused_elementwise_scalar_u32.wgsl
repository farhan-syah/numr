// Fused elementwise scalar WGSL shader (U32)
// fused_mul_add_scalar: out = a * scale + bias
//
// `scale` and `bias` arrive already converted to the element type, the same
// re-encoding `ScalarParams` does, because this shader and the f32 one read the
// same two 4-byte fields under different types. CPU converts the two scalars
// once and then wraps at each step (`fused_mul_add_scalar_kernel` in
// runtime/cpu/kernels/fused_elementwise.rs), which is what the plain operators
// below reproduce.

struct ScalarFmaParams {
    numel: u32,
    scale: u32,
    bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> sfma_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> sfma_out: array<u32>;
@group(0) @binding(2) var<uniform> sfma_params: ScalarFmaParams;

@compute @workgroup_size(256)
fn fused_mul_add_scalar_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < sfma_params.numel) {
        sfma_out[idx] = sfma_a[idx] * sfma_params.scale + sfma_params.bias;
    }
}
