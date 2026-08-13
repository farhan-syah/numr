// Scatter operations for f32

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterParams {
    ndim: u32,
    dim: u32,
    src_total: u32,
    _padding: u32,
    output_shape: vec4<u32>,
    output_strides: vec4<u32>,
    src_shape: vec4<u32>,
    src_strides: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: ScatterParams;

fn get_shape(arr: vec4<u32>, d: u32) -> u32 {
    if (d == 0u) { return arr.x; }
    else if (d == 1u) { return arr.y; }
    else if (d == 2u) { return arr.z; }
    else { return arr.w; }
}

@compute @workgroup_size(256)
fn scatter_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.src_total) {
        return;
    }

    var remaining = idx;
    var dst_offset: u32 = 0u;

    for (var d: u32 = 0u; d < params.ndim; d = d + 1u) {
        let src_stride = get_shape(params.src_strides, d);
        let coord = remaining / src_stride;
        remaining = remaining % src_stride;

        if (d == params.dim) {
            let index_val = indices[idx];
            let dim_size = get_shape(params.output_shape, d);
            if (index_val < 0 || u32(index_val) >= dim_size) {
                return;
            }
            dst_offset = dst_offset + u32(index_val) * get_shape(params.output_strides, d);
        } else {
            dst_offset = dst_offset + coord * get_shape(params.output_strides, d);
        }
    }

    output[dst_offset] = src[idx];
}

// Copy kernel for initializing output from input
@group(0) @binding(0) var<storage, read_write> copy_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<f32>;

struct CopyParams {
    numel: u32,
}

@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(256)
fn copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < copy_params.numel) {
        copy_dst[idx] = copy_src[idx];
    }
}
