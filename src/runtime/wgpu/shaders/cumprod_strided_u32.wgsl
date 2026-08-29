// Strided cumulative product shader for u32.
//
// Same magnitude-plus-saturation state as cumprod_u32.wgsl - see its header
// comment. int_saturate.wgsl is prepended to this module too. The
// `idx / inner_size` below sits outside the loop, so the loop itself stays
// division-free.

struct CumprodStridedParams {
    scan_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CumprodStridedParams;

@compute @workgroup_size(256)
fn cumprod_strided_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_inner = params.outer_size * params.inner_size;
    if (idx >= total_inner) {
        return;
    }

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;

    var mag: u32 = 1u;
    var zero_seen = false;
    var saturated = false;

    for (var s: u32 = 0u; s < params.scan_size; s = s + 1u) {
        let offset = outer_idx * params.scan_size * params.inner_size + s * params.inner_size + inner_idx;
        let v = input[offset];
        if (!zero_seen) {
            if (v == 0u) {
                zero_seen = true;
            } else if (!saturated) {
                if (numr_u32_mul_overflows(mag, v)) {
                    saturated = true;
                } else {
                    mag = mag * v;
                }
            }
        }
        output[offset] = numr_u32_product(zero_seen, saturated, mag);
    }
}
