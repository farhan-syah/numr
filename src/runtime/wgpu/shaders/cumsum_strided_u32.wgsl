// Strided cumulative sum shader for u32.
//
// Same per-step `numr_u32_sat_add` as cumsum_u32.wgsl - see its header
// comment. int_saturate.wgsl is prepended to this module too.

struct CumsumStridedParams {
    scan_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CumsumStridedParams;

@compute @workgroup_size(256)
fn cumsum_strided_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {
        return;
    }

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    var acc: u32 = 0u;
    for (var s: u32 = 0u; s < params.scan_size; s = s + 1u) {
        let offset = outer_idx * params.scan_size * params.inner_size + s * params.inner_size + inner_idx;
        acc = numr_u32_sat_add(acc, input[offset]);
        output[offset] = acc;
    }
}
