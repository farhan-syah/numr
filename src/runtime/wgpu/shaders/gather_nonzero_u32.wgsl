// gather_nonzero operations for u32

const WORKGROUP_SIZE: u32 = 256u;

struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> indices_output: array<i32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> count_params: CountParams;

@compute @workgroup_size(256)
fn gather_nonzero_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let numel = count_params.numel;
    var idx = global_id.x;

    while (idx < numel) {
        if (input[idx] != 0u) {
            let out_idx = atomicAdd(&counter[0], 1u);
            indices_output[out_idx] = i32(idx);
        }
        idx = idx + WORKGROUP_SIZE * 256u;
    }
}
