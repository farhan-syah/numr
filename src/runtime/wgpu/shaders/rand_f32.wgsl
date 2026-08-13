// Rand operation for f32

// PCG hash function for random number generation
// Based on PCG Random Number Generation by Melissa O'Neill
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Initialize PCG state from seed and index
fn pcg_init(seed: u32, idx: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(idx));
}

// Generate uniform float in [0, 1)
fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / 4294967296.0;  // Divide by 2^32
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

struct RandParams {
    numel: u32,
    seed: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> rand_out: array<f32>;
@group(0) @binding(1) var<uniform> rand_params: RandParams;

@compute @workgroup_size(256)
fn rand_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < rand_params.numel) {
        var state = pcg_init(rand_params.seed, idx);
        let value = pcg_uniform(&state);
        rand_out[idx] = f32(value);
    }
}
