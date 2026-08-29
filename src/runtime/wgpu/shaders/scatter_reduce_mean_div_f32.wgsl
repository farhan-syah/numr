// scatter_reduce_mean_div for f32

const WORKGROUP_SIZE: u32 = 256u;

struct MeanDivParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> mean_sum: array<f32>;
@group(0) @binding(1) var<storage, read_write> mean_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> mean_output: array<f32>;
@group(0) @binding(3) var<uniform> mean_params: MeanDivParams;

@compute @workgroup_size(256)
fn scatter_reduce_mean_div_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= mean_params.n) {
        return;
    }

    let c = mean_count[idx];
    if (c > 0u) {
        mean_output[idx] = mean_sum[idx] / f32(c);
    } else {
        // Nobody scattered here, so the slot keeps its seed rather than becoming
        // zero: with include_self that seed is the destination's own value, and
        // CPU's scatter_reduce returns it untouched.
        mean_output[idx] = mean_sum[idx];
    }
}
