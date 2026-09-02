// Normalization operations. F32 only.
// Entry points: rms_norm_f32, layer_norm_f32, layer_norm_no_bias_f32, group_norm_f32
//
// Welford's online algorithm is used for LayerNorm and GroupNorm to compute
// mean and variance in a single pass with numerical stability. Each thread
// accumulates its own (count, mean, M2) triple, then a tree reduction merges
// accumulators across the workgroup using the parallel Welford merge formula:
//   delta    = mean_b - mean_a
//   mean_ab  = mean_a + delta * count_b / (count_a + count_b)
//   M2_ab    = M2_a + M2_b + delta^2 * count_a * count_b / (count_a + count_b)
//
// Shared memory is sized to WORKGROUP_SIZE (256). All workgroup_size attributes
// and shared memory array sizes MUST be kept in sync with this constant.

// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> norm_shared: array<f32, 256>;

// ============================================================================
// RMS Normalization
// ============================================================================
// rms_norm(x, weight, eps) = x / sqrt(mean(x^2) + eps) * weight
// Applied to last dimension

struct RmsNormParams {
    batch_size: u32,    // Product of all dims except the last
    hidden_size: u32,   // Size of the last dimension
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> rms_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rms_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> rms_output: array<f32>;
@group(0) @binding(3) var<uniform> rms_params: RmsNormParams;

@compute @workgroup_size(256)
fn rms_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= rms_params.batch_size) {
        return;
    }

    let hidden_size = rms_params.hidden_size;
    let eps = rms_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Step 1: Compute sum of squares
    var sum_sq: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let val = rms_input[base_offset + i];
        sum_sq = sum_sq + val * val;
        i = i + WORKGROUP_SIZE;
    }

    norm_shared[tid] = sum_sq;
    workgroupBarrier();

    // Reduce to get total sum of squares
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            norm_shared[tid] = norm_shared[tid] + norm_shared[tid + s];
        }
        workgroupBarrier();
    }

    // Compute RMS: sqrt(mean(x^2) + eps)
    let rms = sqrt(norm_shared[0] / f32(hidden_size) + eps);
    workgroupBarrier();

    // Step 2: Normalize and apply weight
    i = tid;
    while (i < hidden_size) {
        rms_output[base_offset + i] = rms_input[base_offset + i] / rms * rms_weight[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Layer Normalization
// ============================================================================
// layer_norm(x, weight, bias, eps) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
// Applied to last dimension

struct LayerNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> ln_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> ln_output: array<f32>;
@group(0) @binding(4) var<uniform> ln_params: LayerNormParams;

// Welford shared memory: count, mean, M2 per thread
var<workgroup> ln_shared_count: array<f32, 256>;
var<workgroup> ln_shared_mean: array<f32, 256>;
var<workgroup> ln_shared_m2: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_params.batch_size) {
        return;
    }

    let hidden_size = ln_params.hidden_size;
    let eps = ln_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so step 3 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and f32 is all this backend has.
    let ref_val = ln_input[base_offset];

    // Step 1: Per-thread Welford accumulation (single pass over input)
    var count: f32 = 0.0;
    var mean: f32 = 0.0;
    var m2: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let x = ln_input[base_offset + i] - ref_val;
        count = count + 1.0;
        let delta = x - mean;
        mean = mean + delta / count;
        m2 = m2 + delta * (x - mean);
        i = i + WORKGROUP_SIZE;
    }

    ln_shared_count[tid] = count;
    ln_shared_mean[tid] = mean;
    ln_shared_m2[tid] = m2;
    workgroupBarrier();

    // Step 2: Tree reduction with Welford merge
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let count_a = ln_shared_count[tid];
            let mean_a = ln_shared_mean[tid];
            let m2_a = ln_shared_m2[tid];
            let count_b = ln_shared_count[tid + s];
            let mean_b = ln_shared_mean[tid + s];
            let m2_b = ln_shared_m2[tid + s];

            let merged_count = count_a + count_b;
            if (merged_count > 0.0) {
                let delta = mean_b - mean_a;
                let merged_mean = mean_a + delta * count_b / merged_count;
                let merged_m2 = m2_a + m2_b + delta * delta * count_a * count_b / merged_count;
                ln_shared_count[tid] = merged_count;
                ln_shared_mean[tid] = merged_mean;
                ln_shared_m2[tid] = merged_m2;
            }
        }
        workgroupBarrier();
    }

    let shifted_mean = ln_shared_mean[0];
    let variance = ln_shared_m2[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply affine transformation (second pass over input)
    i = tid;
    while (i < hidden_size) {
        let normalized = (ln_input[base_offset + i] - ref_val - shifted_mean) * inv_std;
        ln_output[base_offset + i] = normalized * ln_weight[i] + ln_bias[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Layer Normalization without bias
// ============================================================================

@group(0) @binding(0) var<storage, read_write> ln_nb_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_nb_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_nb_output: array<f32>;
@group(0) @binding(3) var<uniform> ln_nb_params: LayerNormParams;

@compute @workgroup_size(256)
fn layer_norm_no_bias_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= ln_nb_params.batch_size) {
        return;
    }

    let hidden_size = ln_nb_params.hidden_size;
    let eps = ln_nb_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so step 3 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and f32 is all this backend has.
    let ref_val = ln_nb_input[base_offset];

    // Step 1: Per-thread Welford accumulation (single pass)
    var count: f32 = 0.0;
    var mean: f32 = 0.0;
    var m2: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let x = ln_nb_input[base_offset + i] - ref_val;
        count = count + 1.0;
        let delta = x - mean;
        mean = mean + delta / count;
        m2 = m2 + delta * (x - mean);
        i = i + WORKGROUP_SIZE;
    }

    // Reuse layer_norm shared memory for reduction
    ln_shared_count[tid] = count;
    ln_shared_mean[tid] = mean;
    ln_shared_m2[tid] = m2;
    workgroupBarrier();

    // Step 2: Tree reduction with Welford merge
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let count_a = ln_shared_count[tid];
            let mean_a = ln_shared_mean[tid];
            let m2_a = ln_shared_m2[tid];
            let count_b = ln_shared_count[tid + s];
            let mean_b = ln_shared_mean[tid + s];
            let m2_b = ln_shared_m2[tid + s];

            let merged_count = count_a + count_b;
            if (merged_count > 0.0) {
                let delta = mean_b - mean_a;
                ln_shared_count[tid] = merged_count;
                ln_shared_mean[tid] = mean_a + delta * count_b / merged_count;
                ln_shared_m2[tid] = m2_a + m2_b + delta * delta * count_a * count_b / merged_count;
            }
        }
        workgroupBarrier();
    }

    let shifted_mean = ln_shared_mean[0];
    let variance = ln_shared_m2[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply weight only (second pass)
    i = tid;
    while (i < hidden_size) {
        let normalized = (ln_nb_input[base_offset + i] - ref_val - shifted_mean) * inv_std;
        ln_nb_output[base_offset + i] = normalized * ln_nb_weight[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Group Normalization
// ============================================================================
// group_norm(x, weight, bias, num_groups) normalizes over groups of channels

struct GroupNormParams {
    batch_size: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    channels_per_group: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> gn_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> gn_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> gn_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> gn_output: array<f32>;
@group(0) @binding(4) var<uniform> gn_params: GroupNormParams;

var<workgroup> gn_shared_count: array<f32, 256>;
var<workgroup> gn_shared_mean: array<f32, 256>;
var<workgroup> gn_shared_m2: array<f32, 256>;

@compute @workgroup_size(256)
fn group_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let bg_id = group_id.x;  // batch_id * num_groups + group_id

    let batch_size = gn_params.batch_size;
    let channels = gn_params.channels;
    let spatial = gn_params.spatial;
    let num_groups = gn_params.num_groups;
    let channels_per_group = gn_params.channels_per_group;
    let eps = gn_params.eps;

    if (bg_id >= batch_size * num_groups) {
        return;
    }

    let batch_id = bg_id / num_groups;
    let group_id_val = bg_id % num_groups;
    let c_start = group_id_val * channels_per_group;
    let group_size = channels_per_group * spatial;

    let batch_offset = batch_id * channels * spatial;
    let group_offset = batch_offset + c_start * spatial;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic, and f32 is all this
    // backend has.
    let ref_val = gn_input[group_offset];

    // Step 1: Per-thread Welford accumulation (single pass)
    var count: f32 = 0.0;
    var mean: f32 = 0.0;
    var m2: f32 = 0.0;
    var i: u32 = tid;
    while (i < group_size) {
        let c_offset = i / spatial;
        let s_offset = i % spatial;
        let idx = group_offset + c_offset * spatial + s_offset;
        let x = gn_input[idx] - ref_val;
        count = count + 1.0;
        let delta = x - mean;
        mean = mean + delta / count;
        m2 = m2 + delta * (x - mean);
        i = i + WORKGROUP_SIZE;
    }

    gn_shared_count[tid] = count;
    gn_shared_mean[tid] = mean;
    gn_shared_m2[tid] = m2;
    workgroupBarrier();

    // Step 2: Tree reduction with Welford merge
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let count_a = gn_shared_count[tid];
            let mean_a = gn_shared_mean[tid];
            let m2_a = gn_shared_m2[tid];
            let count_b = gn_shared_count[tid + s];
            let mean_b = gn_shared_mean[tid + s];
            let m2_b = gn_shared_m2[tid + s];

            let merged_count = count_a + count_b;
            if (merged_count > 0.0) {
                let delta = mean_b - mean_a;
                gn_shared_count[tid] = merged_count;
                gn_shared_mean[tid] = mean_a + delta * count_b / merged_count;
                gn_shared_m2[tid] = m2_a + m2_b + delta * delta * count_a * count_b / merged_count;
            }
        }
        workgroupBarrier();
    }

    let shifted_mean = gn_shared_mean[0];
    let variance = gn_shared_m2[0] / f32(group_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply per-channel weight and bias (second pass)
    i = tid;
    while (i < group_size) {
        let c_offset = i / spatial;
        let s_offset = i % spatial;
        let idx = group_offset + c_offset * spatial + s_offset;
        let channel = c_start + c_offset;
        let normalized = (gn_input[idx] - ref_val - shifted_mean) * inv_std;
        gn_output[idx] = normalized * gn_weight[channel] + gn_bias[channel];
        i = i + WORKGROUP_SIZE;
    }
}
