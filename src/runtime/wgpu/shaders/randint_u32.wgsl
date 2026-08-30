// Randint operation for u32 (unsigned)

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

struct RandintParams {
    numel: u32,
    low: u32,           // Low bound as unsigned integer
    range: u32,         // high - low
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> randint_out: array<u32>;
@group(0) @binding(1) var<uniform> randint_params: RandintParams;

@compute @workgroup_size(256)
fn randint_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < randint_params.numel) {
        var state = pcg_init(randint_params.seed, idx);
        let r = pcg_hash(state);
        // Pure unsigned arithmetic - no overflow for valid inputs
        let offset = r % randint_params.range;
        randint_out[idx] = randint_params.low + offset;
    }
}
