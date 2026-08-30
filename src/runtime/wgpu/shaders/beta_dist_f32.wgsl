// Beta distribution sampling for f32

// PCG hash function for random number generation
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_init(seed: u32, idx: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(idx));
}

fn pcg_uniform(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    // f32(state) rounds up to 2^32 at the top of the u32 range, which would
    // return exactly 1.0 against a documented [0, 1). Clamp to the largest f32
    // below 1.0 -- the shader twin of `DType::largest_value_below_one`.
    return min(f32(*state) / 4294967296.0, 0.99999994);
}

// Box-Muller for normal distribution
fn sample_normal(state: ptr<function, u32>) -> f32 {
    let u1 = max(pcg_uniform(state), 0.0000001);
    let u2 = pcg_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Gamma via Marsaglia-Tsang method
fn sample_gamma_mt(state: ptr<function, u32>, shape: f32, scale: f32) -> f32 {
    var alpha = shape;
    var boost = 1.0;

    // Handle shape < 1 by boosting
    if alpha < 1.0 {
        boost = pow(pcg_uniform(state), 1.0 / alpha);
        alpha = alpha + 1.0;
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / sqrt(9.0 * d);

    // Rejection sampling
    for (var i = 0u; i < 100u; i = i + 1u) {
        var x: f32;
        var v: f32;

        // Generate valid v
        for (var j = 0u; j < 100u; j = j + 1u) {
            x = sample_normal(state);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }

        v = v * v * v;
        let u = pcg_uniform(state);
        let x2 = x * x;

        // Accept/reject
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v * boost * scale;
        }
        if log(u) < 0.5 * x2 + d * (1.0 - v + log(v)) {
            return d * v * boost * scale;
        }
    }

    // Fallback (should rarely reach)
    return d * boost * scale;
}

const WORKGROUP_SIZE: u32 = 256u;

struct BetaParams {
    numel: u32,
    seed: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: BetaParams;

@compute @workgroup_size(256)
fn beta_dist_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < params.numel {
        var state = pcg_init(params.seed, idx);
        let x = sample_gamma_mt(&state, params.alpha, 1.0);
        let y = sample_gamma_mt(&state, params.beta, 1.0);
        out[idx] = f32(x / (x + y));
    }
}
