// count_nonzero operations for u32

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_count: array<u32, 256>;

struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> count_output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> count_params: CountParams;

@compute @workgroup_size(256)
fn count_nonzero_u32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let numel = count_params.numel;

    // Each thread counts its elements
    var local_count: u32 = 0u;
    var idx = global_id.x;
    while (idx < numel) {
        if (input[idx] != 0u) {
            local_count = local_count + 1u;
        }
        idx = idx + WORKGROUP_SIZE * 256u; // stride by total threads
    }

    shared_count[tid] = local_count;
    workgroupBarrier();

    // Tree reduction
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_count[tid] = shared_count[tid] + shared_count[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 adds to global counter
    if (tid == 0u) {
        atomicAdd(&count_output[0], shared_count[0]);
    }
}
