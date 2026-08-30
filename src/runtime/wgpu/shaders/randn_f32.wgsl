// Randn operation for f32

// PCG hash function for random number generation
// Based on PCG Random Number Generation by Melissa O'Neill
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Initialize PCG state from a full 64-bit seed (low/high words) and index.
// WGSL has no u64, so the seed arrives as two u32 words. Both words are
// folded into the state before the index is mixed in, so two seeds that
// share their low word but differ in the high word produce different
// streams — a plain XOR of the two words would not guarantee that.
fn pcg_init(seed_lo: u32, seed_hi: u32, idx: u32) -> u32 {
    let seed_state = pcg_hash(seed_lo ^ pcg_hash(seed_hi));
    return pcg_hash(seed_state ^ pcg_hash(idx));
}

// Generate uniform float in [0, 1)
fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    // f32(state) rounds up to 2^32 at the top of the u32 range, which would
    // return exactly 1.0 against a documented [0, 1). Clamp to the largest f32
    // below 1.0 -- the shader twin of `DType::largest_value_below_one`.
    return min(f32(*state) / 4294967296.0, 0.99999994);
}

// Box-Muller transform for normal distribution
// Generates one normal value, requires two uniform values
fn box_muller(u1: f32, u2: f32) -> f32 {
    let u1_safe = max(u1, 0.0000001);  // Avoid log(0)
    let r = sqrt(-2.0 * log(u1_safe));
    let theta = 6.28318530718 * u2;  // 2 * PI
    return r * cos(theta);
}

const WORKGROUP_SIZE: u32 = 256u;

struct RandnParams {
    numel: u32,
    seed: u32,
    seed_hi: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> randn_out: array<f32>;
@group(0) @binding(1) var<uniform> randn_params: RandnParams;

@compute @workgroup_size(256)
fn randn_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < randn_params.numel) {
        // Use two uniform random values for Box-Muller
        var state = pcg_init(randn_params.seed, randn_params.seed_hi, idx);
        let u1 = pcg_uniform(&state);
        let u2 = pcg_uniform(&state);
        let value = box_muller(u1, u2);
        randn_out[idx] = f32(value);
    }
}
