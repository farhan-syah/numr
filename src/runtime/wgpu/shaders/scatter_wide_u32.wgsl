// Wide-accumulator scatter sum and its narrowing epilogue, for U32.
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl and
// int_wide_div.wgsl, whose 64-bit helpers this file builds on.
//
// An integer scatter reduction accumulates, so it follows the accumulator half
// of the convention in runtime/cpu/kernels/wide_acc.rs: run the total wider than
// one element, narrow exactly once, saturate at the narrow. A 32-bit atomic on
// the element type cannot do that - it wraps on the way past the range and
// reports the wrong sign, and `mean` would then divide a total that is already
// wrong. So the destination is accumulated as two u32 limbs and narrowed by the
// finalize kernel below. CPU does the same arithmetic in i128
// (`scatter_reduce_int_kernel` in runtime/cpu/kernels/scatter_reduce_int.rs).

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

struct ScatterWideParams {
    n: u32,
    divide: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> sw_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> sw_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> sw_acc: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> sw_params: ScatterReduceParams;

@compute @workgroup_size(256)
fn scatter_wide_sum_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = sw_params.outer_size * sw_params.src_dim_size * sw_params.inner_size;
    if (idx >= total) {
        return;
    }

    let inner = idx % sw_params.inner_size;
    let outer = idx / (sw_params.src_dim_size * sw_params.inner_size);

    let index_val = sw_indices[idx];
    if (index_val < 0 || u32(index_val) >= sw_params.dim_size) {
        return;
    }

    let dst_idx = outer * sw_params.dim_size * sw_params.inner_size + u32(index_val) * sw_params.inner_size + inner;

    let v = sw_src[idx];
    let lo = v;
    let hi = 0u;

    // The two limbs are added separately, which stays correct without a 64-bit
    // atomic: the low add's carry is exactly its unsigned wraparound, and adding
    // that carry into the high limb is itself an independent atomic add.
    let old_lo = atomicAdd(&sw_acc[dst_idx * 2u], lo);
    var carry = 0u;
    if (old_lo + lo < old_lo) {
        carry = 1u;
    }
    atomicAdd(&sw_acc[dst_idx * 2u + 1u], hi + carry);
}

@group(0) @binding(0) var<storage, read_write> sw_fin_acc: array<u32>;
@group(0) @binding(1) var<storage, read_write> sw_fin_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> sw_fin_out: array<u32>;
@group(0) @binding(3) var<uniform> sw_fin_params: ScatterWideParams;

@compute @workgroup_size(256)
fn scatter_wide_finalize_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sw_fin_params.n) {
        return;
    }

    var acc = NumrI64(sw_fin_acc[idx * 2u], sw_fin_acc[idx * 2u + 1u]);
    let count = sw_fin_count[idx];

    // Mean divides once, here, by the number of contributions. A slot nobody
    // scattered into keeps its seed instead of becoming zero, which is what CPU
    // returns for count 0.
    if (sw_fin_params.divide != 0u && count > 0u) {
        acc = numr_u64_div_u32(acc, count);
    }

    sw_fin_out[idx] = numr_u64_to_u32_sat(acc);
}
