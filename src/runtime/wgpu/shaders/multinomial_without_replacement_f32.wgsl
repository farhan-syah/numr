// multinomial_without_replacement operation for f32

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
const MAX_CATEGORIES: u32 = 1024u;  // Maximum supported categories

struct MultinomialParams {
    num_distributions: u32,
    num_categories: u32,
    num_samples: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<storage, read_write> multinomial_out: array<i32>;
@group(0) @binding(2) var<uniform> multinomial_params: MultinomialParams;

var<workgroup> shared_probs: array<f32, MAX_CATEGORIES>;

@compute @workgroup_size(256)
fn multinomial_without_replacement_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let dist = gid.x / WORKGROUP_SIZE;
    if (dist >= multinomial_params.num_distributions) {
        return;
    }

    // Copy probabilities to shared memory (each thread copies some elements)
    let prob_offset = dist * multinomial_params.num_categories;
    let elements_per_thread = (multinomial_params.num_categories + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    for (var i: u32 = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = lid.x * elements_per_thread + i;
        if (idx < multinomial_params.num_categories) {
            shared_probs[idx] = probs[prob_offset + idx];
        }
    }

    workgroupBarrier();

    // Only thread 0 does the sequential sampling
    if (lid.x != 0u) {
        return;
    }

    // Initialize RNG
    var state = pcg_init(multinomial_params.seed, dist);

    // Sample without replacement
    for (var s: u32 = 0u; s < multinomial_params.num_samples; s = s + 1u) {
        // Compute sum of remaining probabilities
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {
            sum = sum + shared_probs[i];
        }

        // Generate uniform random value
        let u = pcg_uniform(&state);

        // Linear search using CDF
        var cumsum: f32 = 0.0;
        var result: u32 = multinomial_params.num_categories - 1u;
        for (var i: u32 = 0u; i < multinomial_params.num_categories; i = i + 1u) {
            cumsum = cumsum + shared_probs[i];
            if (cumsum / sum >= u) {
                result = i;
                break;
            }
        }

        multinomial_out[dist * multinomial_params.num_samples + s] = i32(result);

        // Zero out selected category
        shared_probs[result] = 0.0;
    }
}
