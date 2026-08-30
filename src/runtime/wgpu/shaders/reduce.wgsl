// Reduction operations. F32 only.
// Entry points: reduce_sum_f32, reduce_mean_f32, reduce_max_f32, reduce_min_f32,
//               reduce_prod_f32, reduce_any_f32, reduce_all_f32,
//               full_reduce_sum_f32, full_reduce_max_f32, full_reduce_min_f32, full_reduce_prod_f32,
//               argmax_f32, argmin_f32, softmax_f32

// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for parallel reduction
var<workgroup> reduce_shared: array<f32, 256>;

// ============================================================================
// Reduction Parameters
// ============================================================================

struct ReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<f32>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

// ============================================================================
// Sum Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_sum_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Mean Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_mean_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        sum = sum + reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0] / f32(reduce_size);
    }
}

// ============================================================================
// Max Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_max_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        max_val = max(max_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Min Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_min_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: f32 = bitcast<f32>(0x7f800000u); // +inf: min's identity
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        min_val = min(min_val, reduce_input[input_idx]);
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Product Reduction
// ============================================================================

@compute @workgroup_size(256)
fn reduce_prod_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var prod: f32 = 1.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        prod = prod * reduce_input[input_idx];
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = prod;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] * reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Any Reduction (returns 1.0 if any element is non-zero, 0.0 otherwise)
// ============================================================================

@compute @workgroup_size(256)
fn reduce_any_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var found_nonzero: f32 = 0.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        if (reduce_input[input_idx] != 0.0) {
            found_nonzero = 1.0;
        }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = found_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// All Reduction (returns 1.0 if all elements are non-zero, 0.0 otherwise)
// ============================================================================

@compute @workgroup_size(256)
fn reduce_all_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= reduce_params.numel_out) {
        return;
    }

    let reduce_size = reduce_params.reduce_size;
    let inner_size = reduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var all_nonzero: f32 = 1.0;
    var i: u32 = tid;
    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        if (reduce_input[input_idx] == 0.0) {
            all_nonzero = 0.0;
        }
        i = i + WORKGROUP_SIZE;
    }

    reduce_shared[tid] = all_nonzero;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        reduce_output[output_idx] = reduce_shared[0];
    }
}

// ============================================================================
// Full Reduction (reduce all elements to single value)
// ============================================================================

struct FullReduceParams {
    numel: u32,
}

@group(0) @binding(0) var<storage, read_write> full_reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> full_reduce_output: array<f32>;
@group(0) @binding(2) var<uniform> full_reduce_params: FullReduceParams;

@compute @workgroup_size(256)
fn full_reduce_sum_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var sum: f32 = 0.0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        sum = sum + full_reduce_input[i];
        i = i + stride;
    }

    reduce_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] + reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_max_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        max_val = max(max_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = max(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_min_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) group_id: vec3<u32>,
                       @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var min_val: f32 = bitcast<f32>(0x7f800000u); // +inf: min's identity
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        min_val = min(min_val, full_reduce_input[i]);
        i = i + stride;
    }

    reduce_shared[tid] = min_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = min(reduce_shared[tid], reduce_shared[tid + s]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

@compute @workgroup_size(256)
fn full_reduce_prod_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) group_id: vec3<u32>,
                        @builtin(num_workgroups) num_groups: vec3<u32>) {
    let tid = local_id.x;
    let wid = group_id.x;
    let numel = full_reduce_params.numel;

    var prod: f32 = 1.0;
    var i: u32 = wid * WORKGROUP_SIZE + tid;
    let stride = num_groups.x * WORKGROUP_SIZE;

    while (i < numel) {
        prod = prod * full_reduce_input[i];
        i = i + stride;
    }

    reduce_shared[tid] = prod;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            reduce_shared[tid] = reduce_shared[tid] * reduce_shared[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        full_reduce_output[wid] = reduce_shared[0];
    }
}

// ============================================================================
// Argmax / Argmin (returns index of max/min value)
// ============================================================================

var<workgroup> argmax_shared_val: array<f32, 256>;
var<workgroup> argmax_shared_idx: array<u32, 256>;

struct ArgReduceParams {
    reduce_size: u32,
    outer_size: u32,
    inner_size: u32,
    numel_out: u32,
}

@group(0) @binding(0) var<storage, read_write> argreduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> argreduce_output: array<u32>;
@group(0) @binding(2) var<uniform> argreduce_params: ArgReduceParams;

@compute @workgroup_size(256)
fn argmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= argreduce_params.numel_out) {
        return;
    }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var max_idx: u32 = 0u;
    var i: u32 = tid;

    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        let val = argreduce_input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
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

    if (tid == 0u) {
        argreduce_output[output_idx] = argmax_shared_idx[0];
    }
}

@compute @workgroup_size(256)
fn argmin_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let output_idx = group_id.x;

    if (output_idx >= argreduce_params.numel_out) {
        return;
    }

    let reduce_size = argreduce_params.reduce_size;
    let inner_size = argreduce_params.inner_size;

    let outer = output_idx / inner_size;
    let inner = output_idx % inner_size;
    let base_offset = outer * reduce_size * inner_size + inner;

    var min_val: f32 = bitcast<f32>(0x7f800000u); // +inf: min's identity
    var min_idx: u32 = 0u;
    var i: u32 = tid;

    while (i < reduce_size) {
        let input_idx = base_offset + i * inner_size;
        let val = argreduce_input[input_idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
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

    if (tid == 0u) {
        argreduce_output[output_idx] = argmax_shared_idx[0];
    }
}

// ============================================================================
// Softmax (numerically stable)
// ============================================================================

struct SoftmaxParams {
    batch_size: u32,
    dim_size: u32,
}

@group(0) @binding(0) var<storage, read_write> softmax_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmax_output: array<f32>;
@group(0) @binding(2) var<uniform> softmax_params: SoftmaxParams;

var<workgroup> softmax_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= softmax_params.batch_size) {
        return;
    }

    let dim_size = softmax_params.dim_size;
    let base_offset = batch_idx * dim_size;

    // Step 1: Find max for numerical stability
    var max_val: f32 = bitcast<f32>(0xff800000u); // -inf: max's identity
    var i: u32 = tid;
    while (i < dim_size) {
        max_val = max(max_val, softmax_input[base_offset + i]);
        i = i + WORKGROUP_SIZE;
    }

    softmax_shared[tid] = max_val;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            softmax_shared[tid] = max(softmax_shared[tid], softmax_shared[tid + s]);
        }
        workgroupBarrier();
    }

    let global_max = softmax_shared[0];
    workgroupBarrier();

    // Step 2: Compute sum of exp(x - max)
    var sum: f32 = 0.0;
    i = tid;
    while (i < dim_size) {
        sum = sum + exp(softmax_input[base_offset + i] - global_max);
        i = i + WORKGROUP_SIZE;
    }

    softmax_shared[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            softmax_shared[tid] = softmax_shared[tid] + softmax_shared[tid + s];
        }
        workgroupBarrier();
    }

    let global_sum = softmax_shared[0];
    workgroupBarrier();

    // Step 3: Compute output = exp(x - max) / sum
    i = tid;
    while (i < dim_size) {
        softmax_output[base_offset + i] = exp(softmax_input[base_offset + i] - global_max) / global_sum;
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Softmax Backward
// d_input = output * (grad - dot), where dot = sum(grad * output)
// Uses same SoftmaxParams (batch_size, dim_size)
// Bindings: 0=grad(read), 1=output(read), 2=d_input(write), 3=params
// ============================================================================

@group(0) @binding(0) var<storage, read> sbwd_grad: array<f32>;
@group(0) @binding(1) var<storage, read> sbwd_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> sbwd_d_input: array<f32>;
@group(0) @binding(3) var<uniform> sbwd_params: SoftmaxParams;

var<workgroup> sbwd_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= sbwd_params.batch_size) {
        return;
    }

    let dim_size = sbwd_params.dim_size;
    let base_offset = batch_idx * dim_size;

    // Pass 1: dot = sum(grad * output)
    var dot: f32 = 0.0;
    var i: u32 = tid;
    while (i < dim_size) {
        dot = dot + sbwd_grad[base_offset + i] * sbwd_output[base_offset + i];
        i = i + WORKGROUP_SIZE;
    }

    sbwd_shared[tid] = dot;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sbwd_shared[tid] = sbwd_shared[tid] + sbwd_shared[tid + s];
        }
        workgroupBarrier();
    }

    let global_dot = sbwd_shared[0];
    workgroupBarrier();

    // Pass 2: d_input = output * (grad - dot)
    i = tid;
    while (i < dim_size) {
        let idx = base_offset + i;
        sbwd_d_input[idx] = sbwd_output[idx] * (sbwd_grad[idx] - global_dot);
        i = i + WORKGROUP_SIZE;
    }
}
