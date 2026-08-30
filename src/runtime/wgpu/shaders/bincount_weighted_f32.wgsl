// Weighted bincount for f32

const WORKGROUP_SIZE: u32 = 256u;

struct BincountParams {
    n: u32,
    minlength: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> bincount_input: array<i32>;
@group(0) @binding(1) var<storage, read> bincount_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bincount_output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> bincount_params: BincountParams;

@compute @workgroup_size(256)
fn bincount_weighted_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bincount_params.n) {
        return;
    }

    let value = bincount_input[idx];
    if (value < 0 || u32(value) >= bincount_params.minlength) {
        return;
    }

    let bin = u32(value);
    let weight = bincount_weights[idx];

    // WGSL has no float atomics: the bin is stored as the f32 bit pattern in an
    // atomic<u32> and updated by compare-and-swap. Adding the bit patterns with
    // atomicAdd would be integer addition of IEEE encodings, not addition of the
    // values they encode, so the read-add-write must be done explicitly and the
    // swap retried whenever another invocation won the race.
    var old_bits = atomicLoad(&bincount_output[bin]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + weight);
        let cas = atomicCompareExchangeWeak(&bincount_output[bin], old_bits, new_bits);
        if (cas.exchanged) {
            break;
        }
        old_bits = cas.old_value;
    }
}
