// Index reductions for U32.
// Entry points: argmax_u32, argmin_u32

var<workgroup> argmax_shared_val: array<u32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> argreduce_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> argreduce_output: array<u32>;
@group(0) @binding(2) var<uniform> argreduce_params: ArgReduceParams;

@compute @workgroup_size(256)
fn argmax_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= argreduce_params.numel_out) { return; }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: u32 = 0u;
    var max_idx: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        let val = argreduce_input[base_offset + i * inner_size];
        if (val > max_val) { max_val = val; max_idx = i; }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = max_val;
    argmax_shared_idx[tid] = max_idx;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] > argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) { argreduce_output[output_idx] = argmax_shared_idx[0]; }
}

@compute @workgroup_size(256)
fn argmin_u32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;
    if (output_idx >= argreduce_params.numel_out) { return; }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;
    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: u32 = 4294967295u;
    var min_idx: u32 = 0u;
    var i: u32 = tid;
    while (i < reduce_size) {
        let val = argreduce_input[base_offset + i * inner_size];
        if (val < min_val) { min_val = val; min_idx = i; }
        i = i + WORKGROUP_SIZE;
    }

    argmax_shared_val[tid] = min_val;
    argmax_shared_idx[tid] = min_idx;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (argmax_shared_val[tid + s] < argmax_shared_val[tid]) {
                argmax_shared_val[tid] = argmax_shared_val[tid + s];
                argmax_shared_idx[tid] = argmax_shared_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) { argreduce_output[output_idx] = argmax_shared_idx[0]; }
}
