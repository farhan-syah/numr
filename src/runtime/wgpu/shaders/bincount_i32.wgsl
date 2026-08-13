// Unweighted bincount

const WORKGROUP_SIZE: u32 = 256u;

struct BincountParams {
    n: u32,
    minlength: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> bincount_input: array<i32>;
@group(0) @binding(1) var<storage, read_write> bincount_output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> bincount_params: BincountParams;

@compute @workgroup_size(256)
fn bincount_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bincount_params.n) {
        return;
    }

    let value = bincount_input[idx];
    if (value < 0 || u32(value) >= bincount_params.minlength) {
        return;
    }

    atomicAdd(&bincount_output[u32(value)], 1u);
}
