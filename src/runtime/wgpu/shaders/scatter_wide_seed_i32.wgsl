// Widen a I32 tensor into the 64-bit scatter accumulator.
//
// Concatenated after int_saturate.wgsl. Runs before scatter_wide_i32.wgsl's
// atomic adds: the accumulator starts from the seeded destination, which is the
// original values when include_self is set and the reduction's identity
// otherwise, so the seed is one contribution like any other.
//
// The accumulator holds two u32 limbs per element, low limb first.

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterWideParams {
    n: u32,
    divide: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> sw_seed_src: array<i32>;
@group(0) @binding(1) var<storage, read_write> sw_seed_acc: array<u32>;
@group(0) @binding(2) var<uniform> sw_seed_params: ScatterWideParams;

@compute @workgroup_size(256)
fn scatter_wide_seed_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sw_seed_params.n) {
        return;
    }
    let v = sw_seed_src[idx];
    sw_seed_acc[idx * 2u] = bitcast<u32>(v);
    sw_seed_acc[idx * 2u + 1u] = select(0u, 0xffffffffu, v < 0);
}
