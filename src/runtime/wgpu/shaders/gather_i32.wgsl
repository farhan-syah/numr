// Gather operations for i32

const WORKGROUP_SIZE: u32 = 256u;
const MAX_DIMS: u32 = 4u;

struct GatherParams {
    ndim: u32,
    dim: u32,
    total_elements: u32,
    _padding: u32,
    // Shape and strides packed: [input_shape[0..4], input_strides[0..4], output_shape[0..4], output_strides[0..4]]
    input_shape: vec4<u32>,
    input_strides: vec4<u32>,
    output_shape: vec4<u32>,
    output_strides: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<uniform> params: GatherParams;

fn get_shape(arr: vec4<u32>, d: u32) -> u32 {
    if (d == 0u) { return arr.x; }
    else if (d == 1u) { return arr.y; }
    else if (d == 2u) { return arr.z; }
    else { return arr.w; }
}

@compute @workgroup_size(256)
fn gather_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    var remaining = idx;
    var src_offset: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        let out_stride = get_shape(params.output_strides, d);
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;

        if (d == params.dim) {
            let index_val = indices[idx];
            let dim_size = get_shape(params.input_shape, d);
            if (index_val < 0 || u32(index_val) >= dim_size) {
                output[idx] = 0;
                return;
            }
            src_offset = src_offset + u32(index_val) * get_shape(params.input_strides, d);
        } else {
            src_offset = src_offset + coord * get_shape(params.input_strides, d);
        }
    }

    output[idx] = input[src_offset];
}
