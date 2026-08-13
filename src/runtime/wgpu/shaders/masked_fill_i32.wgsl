// masked_fill operations for i32

const WORKGROUP_SIZE: u32 = 256u;

struct MaskedFillParams {
    numel: u32,
    fill_value: f32,
}

@group(0) @binding(0) var<storage, read_write> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<uniform> params: MaskedFillParams;

@compute @workgroup_size(256)
fn masked_fill_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numel) {
        return;
    }

    if (mask[idx] != 0u) {
        output[idx] = i32(params.fill_value);
    } else {
        output[idx] = input[idx];
    }
}
