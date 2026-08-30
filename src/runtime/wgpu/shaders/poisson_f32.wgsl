// Poisson distribution sampling for f32

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

const WORKGROUP_SIZE: u32 = 256u;

struct PoissonParams {
    numel: u32,
    seed: u32,
    lambda: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: PoissonParams;

@compute @workgroup_size(256)
fn poisson_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < params.numel {
        var state = pcg_init(params.seed, idx);

        // Knuth's algorithm for small lambda
        if params.lambda < 30.0 {
            let L = exp(-params.lambda);
            var k = 0u;
            var p = 1.0;

            for (var i = 0u; i < 1000u; i = i + 1u) {
                p = p * pcg_uniform(&state);
                if p <= L {
                    break;
                }
                k = k + 1u;
            }
            out[idx] = f32(f32(k));
        } else {
            // Normal approximation for large lambda
            let z = sample_normal(&state);
            let result = max(0.0, round(params.lambda + sqrt(params.lambda) * z));
            out[idx] = f32(result);
        }
    }
}
