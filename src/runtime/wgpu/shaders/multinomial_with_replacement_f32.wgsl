// multinomial_with_replacement operation for f32

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

struct MultinomialParams {
    num_distributions: u32,
    num_categories: u32,
    num_samples: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> multinomial_out: array<i32>;
@group(0) @binding(2) var<uniform> multinomial_params: MultinomialParams;

@compute @workgroup_size(256)
fn multinomial_with_replacement_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = multinomial_params.num_distributions * multinomial_params.num_samples;
    if (idx >= total) {
        return;
    }

    let dist = idx / multinomial_params.num_samples;
    let sample = idx % multinomial_params.num_samples;

    // Initialize RNG for this thread
    var state = pcg_init(multinomial_params.seed, idx);

    // Get pointer to this distribution's probabilities
    let prob_offset = dist * multinomial_params.num_categories;

    // Compute sum of probabilities for normalization
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {
        sum = sum + probs[prob_offset + i];
    }

    // Generate uniform random value
    let u = pcg_uniform(&state);

    // Linear search using CDF (on-the-fly computation)
    // Find smallest index where cumsum/sum >= u
    var cumsum: f32 = 0.0;
    var result: u32 = multinomial_params.num_categories - 1u;  // Default to last category
    for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {
        cumsum = cumsum + probs[prob_offset + i];
        if (cumsum / sum >= u) {
            result = i;
            break;
        }
    }

    multinomial_out[dist * multinomial_params.num_samples + sample] = i32(result);
}
