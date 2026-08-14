// ConvTranspose1d shader for f32
// Input layout:  (N, C_in, L)
// Weight layout: (C_in, C_out/groups, K)   <-- input channels lead, unlike conv1d
// Output layout: (N, C_out, L_out)
//
// Gather formulation: one invocation per output element, searching for the
// inputs that scatter into it. The scatter form would require atomics, which
// WGSL does not offer for f32 and which would also cost reproducibility.
//
// Input j reaches output t via tap k when t = j*stride - padding + k*dilation,
// so j = (t + padding - k*dilation) / stride, valid only when it divides evenly.

const WORKGROUP_SIZE: u32 = 256u;

struct Conv1dParams {
    batch: u32,
    c_in: u32,
    length: u32,
    c_out: u32,
    kernel_size: u32,
    output_length: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> ct1d_input: array<f32>;
@group(0) @binding(1) var<storage, read> ct1d_weight: array<f32>;
@group(0) @binding(2) var<storage, read> ct1d_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> ct1d_output: array<f32>;
@group(0) @binding(4) var<uniform> ct1d_params: Conv1dParams;

@compute @workgroup_size(256)
fn conv_transpose1d_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = ct1d_params.batch * ct1d_params.c_out * ct1d_params.output_length;
    if (idx >= total) { return; }

    let ox = idx % ct1d_params.output_length;
    let oc = (idx / ct1d_params.output_length) % ct1d_params.c_out;
    let b = idx / (ct1d_params.c_out * ct1d_params.output_length);

    let c_in_per_group = ct1d_params.c_in / ct1d_params.groups;
    let c_out_per_group = ct1d_params.c_out / ct1d_params.groups;
    let g = oc / c_out_per_group;
    let oc_local = oc % c_out_per_group;
    let c_in_start = g * c_in_per_group;

    var sum: f32 = 0.0;

    for (var kx: u32 = 0u; kx < ct1d_params.kernel_size; kx = kx + 1u) {
        let shifted = i32(ox + ct1d_params.padding) - i32(kx * ct1d_params.dilation);
        if (shifted < 0) { continue; }
        if (u32(shifted) % ct1d_params.stride != 0u) { continue; }
        let j = u32(shifted) / ct1d_params.stride;
        if (j >= ct1d_params.length) { continue; }

        for (var ic: u32 = 0u; ic < c_in_per_group; ic = ic + 1u) {
            let c_in_idx = c_in_start + ic;
            let input_idx = b * ct1d_params.c_in * ct1d_params.length
                          + c_in_idx * ct1d_params.length + j;
            let weight_idx = c_in_idx * c_out_per_group * ct1d_params.kernel_size
                           + oc_local * ct1d_params.kernel_size + kx;
            sum = sum + ct1d_input[input_idx] * ct1d_weight[weight_idx];
        }
    }

    if (ct1d_params.has_bias != 0u) {
        sum = sum + ct1d_bias[oc];
    }

    ct1d_output[idx] = sum;
}
