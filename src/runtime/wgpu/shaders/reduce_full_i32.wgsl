// Whole-tensor reductions for I32.
// Entry points: full_reduce_max_i32, full_reduce_min_i32
//
// There is deliberately no full-reduce sum or prod here. Those accumulate,
// so they must run in the wide accumulator that `reduce_int_acc_i32.wgsl`
// carries and narrow exactly once; a two-pass whole-tensor reduction would
// have to store its partials in the element type and saturate each one,
// which is not the same answer. `native_reduce_op` collapses the reduced
// dims and calls the single-dim kernel instead.

struct FullReduceParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> full_reduce_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> full_reduce_output: array<i32>;
@group(0) @binding(2) var<uniform> full_reduce_params: FullReduceParams;


@compute @workgroup_size(256)
fn full_reduce_max_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var max_val: i32 = (-2147483647i - 1i);
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { max_val = max(max_val, full_reduce_input[i]); i = i + stride; }

    reduce_shared[tid] = max_val;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}

@compute @workgroup_size(256)
fn full_reduce_min_i32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var min_val: i32 = 2147483647i;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;
    while (i < numel) { min_val = min(min_val, full_reduce_input[i]); i = i + stride; }

    reduce_shared[tid] = min_val;
    workgroupBarrier();
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) { reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) { full_reduce_output[wid] = reduce_shared[0]; }
}
