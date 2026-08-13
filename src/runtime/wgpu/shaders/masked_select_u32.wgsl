// masked_select operations for u32

const WORKGROUP_SIZE: u32 = 256u;

// Phase 1: Count masked elements
struct CountParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> count_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> count_result: atomic<u32>;
@group(0) @binding(2) var<uniform> count_params: CountParams;

var<workgroup> shared_count: atomic<u32>;

@compute @workgroup_size(256)
fn masked_count(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x == 0u) {
        atomicStore(&shared_count, 0u);
    }
    workgroupBarrier();

    var local_count: u32 = 0u;
    var i = gid.x;
    while (i < count_params.numel) {
        if (count_mask[i] != 0u) {
            local_count = local_count + 1u;
        }
        i = i + 256u * 256u; // Grid stride
    }

    atomicAdd(&shared_count, local_count);
    workgroupBarrier();

    if (lid.x == 0u) {
        atomicAdd(&count_result, atomicLoad(&shared_count));
    }
}

// Phase 2: Compute prefix sum (sequential - for small arrays)
struct PrefixSumParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> prefix_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> prefix_sum: array<u32>;
@group(0) @binding(2) var<uniform> prefix_params: PrefixSumParams;

@compute @workgroup_size(1)
fn masked_prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < prefix_params.numel; i = i + 1u) {
        prefix_sum[i] = sum;
        if (prefix_mask[i] != 0u) {
            sum = sum + 1u;
        }
    }
}

// Phase 3: Gather selected elements
struct SelectParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> select_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> select_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> select_prefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> select_output: array<u32>;
@group(0) @binding(4) var<uniform> select_params: SelectParams;

@compute @workgroup_size(256)
fn masked_select_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= select_params.numel) {
        return;
    }

    if (select_mask[idx] != 0u) {
        let out_idx = select_prefix[idx];
        select_output[out_idx] = select_input[idx];
    }
}
