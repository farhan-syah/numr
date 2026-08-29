// Cumulative product shader for u32.
//
// Tracks the exact magnitude and a saturation flag rather than multiplying in
// u32, so the output is the true product clamped to u32 - see the "Integer
// cumprod" section of int_saturate.wgsl, which is prepended to this module.

struct CumprodParams {
    scan_size: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CumprodParams;

@compute @workgroup_size(256)
fn cumprod_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.scan_size;
    var mag: u32 = 1u;
    var zero_seen = false;
    var saturated = false;

    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {
        let v = input[base + i];
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
        output[base + i] = numr_u32_product(zero_seen, saturated, mag);
    }
}
