// scatter_reduce_prod for i32: one thread per DESTINATION element.
//
// Concatenated after int_saturate.wgsl, whose magnitude-and-sign cumprod
// helpers this file builds on.
//
// An integer scatter product accumulates, and accumulators saturate rather than
// wrap (runtime/cpu/kernels/wide_acc.rs). A 32-bit atomic on the element type
// cannot deliver that: the clamped value would have to double as its own
// saturation state, so a running product of exactly i32::MAX followed by a
// negative factor reports i32::MIN where the true answer, -i32::MAX, is
// representable.
//
// So this kernel owns the destination element instead of the source element,
// exactly as scatter_reduce_int_impl does in
// runtime/cuda/kernels/scatter_reduce.cu. Each thread scans the source
// positions in its own lane, needs no atomic at all, and keeps the running
// product as magnitude plus sign parity: multiplying by 0 pins the true product
// at 0 forever, and every other factor has magnitude at least 1, so a magnitude
// that has left the range can never come back. That state is exact for every
// representable product and clamps to the correctly-signed bound otherwise,
// matching the i128 accumulator in
// runtime/cpu/kernels/scatter_reduce_int.rs narrowed by from_i128_saturating.
//
// Division and modulo appear only outside the loop: an integer divide inside a
// WGSL loop fails NVIDIA's shader compiler with "NVVM compilation failed".

const WORKGROUP_SIZE: u32 = 256u;

struct ScatterReduceParams {
    dim: u32,
    outer_size: u32,
    dim_size: u32,
    inner_size: u32,
    src_dim_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Every storage binding is declared read_write, including the two this kernel
// only reads. The pipeline cache builds this layout with
// `num_readonly_storage: 0`, and wgpu requires a shader variable's access mode
// to match its layout entry exactly - a `read` variable against a
// `Storage { read_only: false }` entry fails pipeline creation with a bare
// validation error, which then invalidates the device for every later test.
@group(0) @binding(0) var<storage, read_write> scatter_src: array<i32>;
@group(0) @binding(1) var<storage, read_write> scatter_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> scatter_dst: array<i32>;
@group(0) @binding(3) var<uniform> scatter_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_reduce_prod_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim_size = scatter_params.dim_size;
    let inner_size = scatter_params.inner_size;
    let src_dim_size = scatter_params.src_dim_size;
    let total = scatter_params.outer_size * dim_size * inner_size;
    if (idx >= total) {
        return;
    }

    let inner = idx % inner_size;
    let d = (idx / inner_size) % dim_size;
    let outer = idx / (dim_size * inner_size);

    // The destination already holds the reduction's first contribution: a copy
    // of the original tensor when include_self is set, the identity 1
    // otherwise. Both are just factors, so one code path covers them.
    let seed = scatter_dst[idx];
    var zero_seen = seed == 0;
    var negative = seed < 0;
    var saturated = false;
    var mag = numr_i32_magnitude(seed);

    // Only source elements in this destination's own (outer, inner) lane can
    // land here, so the scan is over src_dim_size, not the whole source.
    let lane_base = outer * src_dim_size * inner_size + inner;

    for (var s = 0u; s < src_dim_size; s = s + 1u) {
        let src_idx = lane_base + s * inner_size;
        let index_val = scatter_indices[src_idx];
        if (index_val < 0 || u32(index_val) >= dim_size) {
            continue;
        }
        if (u32(index_val) != d) {
            continue;
        }
        let v = scatter_src[src_idx];
        if (v == 0) {
            zero_seen = true;
        }
        // `select` rather than a bool `!=`, which naga accepts but which is
        // easy to misread as an integer comparison here.
        negative = select(negative, !negative, v < 0);
        let v_mag = numr_i32_magnitude(v);
        if (!saturated) {
            if (numr_u32_mul_overflows(mag, v_mag)) {
                saturated = true;
            } else {
                mag = mag * v_mag;
            }
        }
    }

    scatter_dst[idx] = numr_i32_product(zero_seen, saturated, negative, mag);
}
