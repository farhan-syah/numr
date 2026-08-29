// scatter_reduce_prod for u32: one thread per DESTINATION element.
//
// Concatenated after int_saturate.wgsl, whose magnitude-and-sign cumprod
// helpers this file builds on. The signed twin, scatter_reduce_prod_i32.wgsl,
// carries the reasoning; the unsigned case is the same scan without a sign to
// track.

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Every storage binding is declared read_write, including the two this kernel
// only reads. The pipeline cache builds this layout with
// `num_readonly_storage: 0`, and wgpu requires a shader variable's access mode
// to match its layout entry exactly - a `read` variable against a
// `Storage { read_only: false }` entry fails pipeline creation with a bare
// validation error, which then invalidates the device for every later test.
@group(0) @binding(0) var<storage, read_write> scatter_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<u32>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_prod_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim_size = scatter_params.dim_size;
    let inner_size = scatter_params.inner_size;
    let src_dim_size = scatter_params.src_dim_size;
    let total = scatter_params.outer_size * dim_size * inner_size;
    if (idx >= total) {
        return;
    }

    let inner = idx % inner_size;
    let d = (idx / inner_size) % dim_size;
    let outer = idx / (dim_size * inner_size);

    let seed = scatter_dst[idx];
    var zero_seen = seed == 0u;
    var saturated = false;
    var mag = seed;

    let lane_base = outer * src_dim_size * inner_size + inner;

    for (var s = 0u; s < src_dim_size; s = s + 1u) {
        let src_idx = lane_base + s * inner_size;
        let index_val = scatter_indices[src_idx];
        if (index_val < 0 || u32(index_val) >= dim_size) {
            continue;
        }
        if (u32(index_val) != d) {
            continue;
        }
        let v = scatter_src[src_idx];
        if (v == 0u) {
            zero_seen = true;
        }
        if (!saturated) {
            if (numr_u32_mul_overflows(mag, v)) {
                saturated = true;
            } else {
                mag = mag * v;
            }
        }
    }

    scatter_dst[idx] = numr_u32_product(zero_seen, saturated, mag);
}
