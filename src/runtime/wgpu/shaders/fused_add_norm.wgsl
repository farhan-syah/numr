// Fused add + normalization operations. F32 only.
// Entry points:
//   - fused_add_rms_norm_f32: Add residual, then RMS normalize
//   - fused_add_layer_norm_f32: Add residual, then layer normalize
//   - fused_add_rms_norm_bwd_f32: Backward pass for fused add RMS norm
//   - fused_add_layer_norm_bwd_f32: Backward pass for fused add layer norm
//   - reduce_sum_rows_f32: Reduce d_weight/d_bias scratch buffers across batch dimension

// ============================================================================
// Workgroup Configuration
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// ============================================================================
// RMS Normalization Structs
// ============================================================================

struct RmsNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

struct LayerNormParams {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

struct ReduceSumParams {
    batch_size: u32,
    hidden_size: u32,
}

// ============================================================================
// Fused Add + RMS Norm (Forward)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> farn_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> farn_residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> farn_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> farn_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> farn_pre_norm: array<f32>;
@group(0) @binding(5) var<uniform> farn_params: RmsNormParams;

var<workgroup> farn_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn fused_add_rms_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= farn_params.batch_size) {
        return;
    }

    let hidden_size = farn_params.hidden_size;
    let eps = farn_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Step 1: Add input + residual -> pre_norm, compute sum of squares
    var sum_sq: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let pre_val = farn_input[base_offset + i] + farn_residual[base_offset + i];
        farn_pre_norm[base_offset + i] = pre_val;
        sum_sq = sum_sq + pre_val * pre_val;
        i = i + WORKGROUP_SIZE;
    }

    farn_shared[tid] = sum_sq;
    workgroupBarrier();

    // Reduce to get total sum of squares
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            farn_shared[tid] = farn_shared[tid] + farn_shared[tid + s];
        }
        workgroupBarrier();
    }

    // Compute RMS: sqrt(mean(x^2) + eps)
    let rms = sqrt(farn_shared[0] / f32(hidden_size) + eps);
    workgroupBarrier();

    // Step 2: Normalize and apply weight
    i = tid;
    while (i < hidden_size) {
        farn_output[base_offset + i] = farn_pre_norm[base_offset + i] / rms * farn_weight[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Fused Add + Layer Norm (Forward)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> faln_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> faln_residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> faln_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> faln_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> faln_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> faln_pre_norm: array<f32>;
@group(0) @binding(6) var<uniform> faln_params: LayerNormParams;

var<workgroup> faln_shared_mean: array<f32, 256>;
var<workgroup> faln_shared_var: array<f32, 256>;

@compute @workgroup_size(256)
fn fused_add_layer_norm_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                            @builtin(local_invocation_id) local_id: vec3<u32>,
                            @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= faln_params.batch_size) {
        return;
    }

    let hidden_size = faln_params.hidden_size;
    let eps = faln_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation is where the mantissa
    // goes. Mean, variance and the normalized value are all invariant under the
    // shift in exact arithmetic, and f32 is all this backend has.
    let ref_val = faln_input[base_offset] + faln_residual[base_offset];

    // Step 1: Add input + residual -> pre_norm, compute sum for mean
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let pre_val = faln_input[base_offset + i] + faln_residual[base_offset + i];
        faln_pre_norm[base_offset + i] = pre_val;
        sum = sum + (pre_val - ref_val);
        i = i + WORKGROUP_SIZE;
    }

    faln_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            faln_shared_mean[tid] = faln_shared_mean[tid] + faln_shared_mean[tid + s];
        }
        workgroupBarrier();
    }

    let shifted_mean = faln_shared_mean[0] / f32(hidden_size);
    workgroupBarrier();

    // Step 2: Compute variance
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < hidden_size) {
        let diff = (faln_pre_norm[base_offset + i] - ref_val) - shifted_mean;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }

    faln_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            faln_shared_var[tid] = faln_shared_var[tid] + faln_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let variance = faln_shared_var[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply affine transformation
    i = tid;
    while (i < hidden_size) {
        let normalized = (faln_pre_norm[base_offset + i] - ref_val - shifted_mean) * inv_std;
        faln_output[base_offset + i] = normalized * faln_weight[i] + faln_bias[i];
        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Fused Add + RMS Norm (Backward)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> farnb_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> farnb_pre_norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> farnb_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> farnb_d_input_residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> farnb_d_weight_scratch: array<f32>;
@group(0) @binding(5) var<uniform> farnb_params: RmsNormParams;

var<workgroup> farnb_shared_sum_sq: array<f32, 256>;
var<workgroup> farnb_shared_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn fused_add_rms_norm_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                              @builtin(local_invocation_id) local_id: vec3<u32>,
                              @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= farnb_params.batch_size) {
        return;
    }

    let hidden_size = farnb_params.hidden_size;
    let eps = farnb_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Phase 1: Compute sum_sq and dot(grad, weight, pre_norm)
    var sum_sq: f32 = 0.0;
    var dot: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        let pre_val = farnb_pre_norm[base_offset + i];
        sum_sq = sum_sq + pre_val * pre_val;
        dot = dot + farnb_grad[base_offset + i] * farnb_weight[i] * pre_val;
        i = i + WORKGROUP_SIZE;
    }

    farnb_shared_sum_sq[tid] = sum_sq;
    farnb_shared_dot[tid] = dot;
    workgroupBarrier();

    // Reduce both sums
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            farnb_shared_sum_sq[tid] = farnb_shared_sum_sq[tid] + farnb_shared_sum_sq[tid + s];
            farnb_shared_dot[tid] = farnb_shared_dot[tid] + farnb_shared_dot[tid + s];
        }
        workgroupBarrier();
    }

    let total_sum_sq = farnb_shared_sum_sq[0];
    let total_dot = farnb_shared_dot[0];
    let rms = sqrt(total_sum_sq / f32(hidden_size) + eps);
    let inv_rms = 1.0 / rms;
    let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    let coeff = total_dot * inv_rms_cubed / f32(hidden_size);
    workgroupBarrier();

    // Phase 2: Compute d_input_residual and accumulate d_weight
    i = tid;
    while (i < hidden_size) {
        // d_input_residual = (grad * weight - pre_norm * coeff) * inv_rms
        farnb_d_input_residual[base_offset + i] =
            (farnb_grad[base_offset + i] * farnb_weight[i] - farnb_pre_norm[base_offset + i] * coeff) * inv_rms;

        // d_weight contribution: sum(grad * pre_norm / rms) per element
        // Each workgroup writes its per-row contribution to scratch
        farnb_d_weight_scratch[base_offset + i] = farnb_grad[base_offset + i] * farnb_pre_norm[base_offset + i] * inv_rms;

        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Fused Add + Layer Norm (Backward)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> falnb_grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> falnb_pre_norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> falnb_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> falnb_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> falnb_d_input_residual: array<f32>;
@group(0) @binding(5) var<storage, read_write> falnb_d_weight_scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> falnb_d_bias_scratch: array<f32>;
@group(0) @binding(7) var<uniform> falnb_params: LayerNormParams;

var<workgroup> falnb_shared_mean: array<f32, 256>;
var<workgroup> falnb_shared_var: array<f32, 256>;

@compute @workgroup_size(256)
fn fused_add_layer_norm_bwd_f32(@builtin(global_invocation_id) global_id: vec3<u32>,
                                @builtin(local_invocation_id) local_id: vec3<u32>,
                                @builtin(workgroup_id) group_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = group_id.x;

    if (batch_idx >= falnb_params.batch_size) {
        return;
    }

    let hidden_size = falnb_params.hidden_size;
    let eps = falnb_params.eps;
    let base_offset = batch_idx * hidden_size;

    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance rather than consuming a saved
    // statistic, so it carries the forward pass's cancellation exactly. Mean,
    // variance and the normalized value are all invariant under the shift in
    // exact arithmetic. `mean_gs` needs no shift: it averages `grad * weight`.
    let ref_val = falnb_pre_norm[base_offset];

    // Phase 1: Compute mean of pre_norm
    var sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < hidden_size) {
        sum = sum + (falnb_pre_norm[base_offset + i] - ref_val);
        i = i + WORKGROUP_SIZE;
    }

    falnb_shared_mean[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            falnb_shared_mean[tid] = falnb_shared_mean[tid] + falnb_shared_mean[tid + s];
        }
        workgroupBarrier();
    }

    let shifted_mean = falnb_shared_mean[0] / f32(hidden_size);
    workgroupBarrier();

    // Phase 2: Compute variance
    var var_sum: f32 = 0.0;
    i = tid;
    while (i < hidden_size) {
        let diff = (falnb_pre_norm[base_offset + i] - ref_val) - shifted_mean;
        var_sum = var_sum + diff * diff;
        i = i + WORKGROUP_SIZE;
    }

    falnb_shared_var[tid] = var_sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            falnb_shared_var[tid] = falnb_shared_var[tid] + falnb_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let variance = falnb_shared_var[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + eps);

    // Compute grad_scaled = grad * weight sums
    var sum_gs: f32 = 0.0;
    var sum_gs_n: f32 = 0.0;
    i = tid;
    while (i < hidden_size) {
        let normalized = (falnb_pre_norm[base_offset + i] - ref_val - shifted_mean) * inv_std;
        let gs = falnb_grad[base_offset + i] * falnb_weight[i];
        sum_gs = sum_gs + gs;
        sum_gs_n = sum_gs_n + gs * normalized;
        i = i + WORKGROUP_SIZE;
    }

    falnb_shared_mean[tid] = sum_gs;
    falnb_shared_var[tid] = sum_gs_n;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            falnb_shared_mean[tid] = falnb_shared_mean[tid] + falnb_shared_mean[tid + s];
            falnb_shared_var[tid] = falnb_shared_var[tid] + falnb_shared_var[tid + s];
        }
        workgroupBarrier();
    }

    let total_sum_gs = falnb_shared_mean[0];
    let total_sum_gs_n = falnb_shared_var[0];
    workgroupBarrier();

    // Phase 3: Compute d_input_residual, d_weight_scratch, d_bias_scratch
    i = tid;
    while (i < hidden_size) {
        let normalized = (falnb_pre_norm[base_offset + i] - ref_val - shifted_mean) * inv_std;

        // d_input_residual = inv_std * (grad*weight - mean_gs - normalized * mean_gs_n)
        let mean_gs_val = total_sum_gs / f32(hidden_size);
        let mean_gs_n_val = total_sum_gs_n / f32(hidden_size);
        let gs = falnb_grad[base_offset + i] * falnb_weight[i];
        falnb_d_input_residual[base_offset + i] = inv_std *
            (gs - mean_gs_val - normalized * mean_gs_n_val);

        // d_weight: sum(grad * normalized) per element
        falnb_d_weight_scratch[base_offset + i] = falnb_grad[base_offset + i] * normalized;

        // d_bias: sum(grad) per element
        falnb_d_bias_scratch[base_offset + i] = falnb_grad[base_offset + i];

        i = i + WORKGROUP_SIZE;
    }
}

// ============================================================================
// Reduce Sum Rows (helper for backward)
// ============================================================================

@group(0) @binding(0) var<storage, read_write> rsr_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> rsr_output: array<f32>;
@group(0) @binding(2) var<uniform> rsr_params: ReduceSumParams;

@compute @workgroup_size(256)
fn reduce_sum_rows_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= rsr_params.hidden_size) {
        return;
    }

    var sum: f32 = 0.0;
    for (var b: u32 = 0u; b < rsr_params.batch_size; b = b + 1u) {
        sum = sum + rsr_input[b * rsr_params.hidden_size + i];
    }
    rsr_output[i] = sum;
}
