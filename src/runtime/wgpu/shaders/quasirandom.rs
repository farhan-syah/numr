//! Quasi-random sequence generation WGSL kernel launchers
//!
//! Provides launchers for quasi-random sequences:
//! - Sobol sequence (Gray code-based low-discrepancy)
//! - Halton sequence (van der Corput in prime bases)
//! - Latin Hypercube Sampling (stratified random sampling)

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Sobol Sequence
// ============================================================================

const SOBOL_WGSL: &str = r#"
// Direction vectors are passed via storage buffer.
// This supports all 21,201 dimensions from Joe & Kuo (2008).
// Each dimension has 32 direction vectors.

struct SobolParams {
    n_points: u32,
    dimension: u32,
    skip: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read_write> direction_vectors: array<u32>;
@group(0) @binding(2) var<uniform> params: SobolParams;

@compute @workgroup_size(256)
fn sobol_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) { return; }

    let point_index = params.skip + idx;

    // Gray code
    let gray = point_index ^ (point_index >> 1u);

    for (var d = 0u; d < params.dimension; d++) {
        // Get direction vectors for this dimension
        let v_offset = d * 32u;

        // Compute Sobol point using direction vectors
        var x = 0u;
        for (var bit = 0u; bit < 32u; bit++) {
            if ((gray & (1u << bit)) != 0u) {
                x = x ^ direction_vectors[v_offset + bit];
            }
        }

        // Convert to float in [0, 1)
        // f32(x) rounds up to 2^32 at the top of the u32 range, which would
        // return exactly 1.0 against a documented [0, 1). Clamp to the largest
        // f32 below 1.0 -- the shader twin of `DType::largest_value_below_one`.
        output[idx * params.dimension + d] = min(f32(x) / 4294967296.0, 0.99999994);
    }
}
"#;

/// Launches the Sobol sequence generator shader.
///
/// Generates low-discrepancy quasi-random sequences using Sobol direction numbers.
/// Useful for numerical integration and Monte Carlo methods.
///
/// Supports all 21,201 dimensions from Joe & Kuo (2008).
///
/// # Arguments
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - Command queue for GPU execution
/// * `out` - Output buffer for generated samples
/// * `direction_vectors` - Pre-computed direction vectors buffer `[dimension][32]`
/// * `params` - Parameters buffer (dimension, offset)
/// * `n_points` - Number of points to generate
/// * `dtype` - Data type (must be floating-point)
pub fn launch_sobol(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    direction_vectors: &Buffer,
    params: &Buffer,
    n_points: usize,
    dtype: DType,
) -> Result<()> {
    if n_points == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "sobol")?;

    let name = "sobol_f32";
    let module = cache.get_or_create_module(name, SOBOL_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, direction_vectors, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sobol"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sobol"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Dispatch based on n_points
        pass.dispatch_workgroups(workgroup_count(n_points), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Halton Sequence
// ============================================================================

const HALTON_WGSL: &str = r#"
// First 100 prime numbers
const PRIMES: array<u32, 100> = array(
    2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u, 67u, 71u,
    73u, 79u, 83u, 89u, 97u, 101u, 103u, 107u, 109u, 113u, 127u, 131u, 137u, 139u, 149u, 151u, 157u, 163u, 167u, 173u,
    179u, 181u, 191u, 193u, 197u, 199u, 211u, 223u, 227u, 229u, 233u, 239u, 241u, 251u, 257u, 263u, 269u, 271u, 277u, 281u,
    283u, 293u, 307u, 311u, 313u, 317u, 331u, 337u, 347u, 349u, 353u, 359u, 367u, 373u, 379u, 383u, 389u, 397u, 401u, 409u,
    419u, 421u, 431u, 433u, 439u, 443u, 449u, 457u, 461u, 463u, 467u, 479u, 487u, 491u, 499u, 503u, 509u, 521u, 523u, 541u
);

struct HaltonParams {
    n_points: u32,
    dimension: u32,
    skip: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: HaltonParams;

fn van_der_corput_f32(index: u32, base: u32) -> f32 {
    var result = 0.0;
    var f = 1.0 / f32(base);
    var i = index;
    while (i > 0u) {
        result += f * f32(i % base);
        i = i / base;
        f = f / f32(base);
    }
    // The exact sum is below 1, but the f32 accumulation can round up onto it.
    return min(result, 0.99999994);
}

@compute @workgroup_size(256)
fn halton_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) { return; }

    let point_index = params.skip + idx;

    for (var d = 0u; d < params.dimension; d++) {
        let base = PRIMES[d];
        output[idx * params.dimension + d] = van_der_corput_f32(point_index, base);
    }
}
"#;

/// Launches the Halton sequence generator shader.
///
/// Generates low-discrepancy quasi-random sequences using the Halton sequence.
/// Based on van der Corput sequences with different prime bases.
///
/// # Arguments
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - Command queue for GPU execution
/// * `out` - Output buffer for generated samples
/// * `params` - Parameters buffer (dimension, n_points, skip)
/// * `total_elements` - Total number of elements to generate
/// * `dtype` - Data type (must be floating-point)
pub fn launch_halton(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "halton")?;

    let name = "halton_f32";
    let module = cache.get_or_create_module(name, HALTON_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("halton"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("halton"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Latin Hypercube Sampling
// ============================================================================

const LATIN_HYPERCUBE_WGSL: &str = r#"
struct LatinHypercubeParams {
    n_samples: u32,
    dimension: u32,
    seed: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: LatinHypercubeParams;

// PCG hash (Melissa O'Neill), the same mixer the other WebGPU RNG shaders use.
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Counter-based draw. The stream is a function of BOTH the base key and the
// draw index, so two bases never share a trajectory the way two seeds of a
// plain xorshift do -- they only enter one cycle at different offsets.
fn rng_next(base: u32, counter_ptr: ptr<function, u32>) -> u32 {
    let c = *counter_ptr;
    *counter_ptr = c + 1u;
    return pcg_hash(base ^ pcg_hash(c ^ 0x9E3779B9u));
}

// Uniform in [0, 1). The shift to 24 bits keeps every value exactly
// representable in f32: f32(x) for a full 32-bit x rounds up to 2^32 at the top
// of the range, which would return exactly 1.0.
fn rng_uniform(base: u32, counter_ptr: ptr<function, u32>) -> f32 {
    return f32(rng_next(base, counter_ptr) >> 8u) / 16777216.0;
}

// Key for one dimension's permutation stream, and a distinct key per thread for
// the within-stratum offsets. Both hash the seed rather than adding to it, so a
// seed that moves by 1 between calls moves the whole permutation.
fn permutation_key(seed: u32, dim: u32) -> u32 {
    return pcg_hash(seed ^ pcg_hash(dim ^ 0x2545F491u));
}

fn offset_key(seed: u32, dim: u32, lane: u32) -> u32 {
    return pcg_hash(permutation_key(seed, dim) ^ pcg_hash(lane ^ 0xB5297A4Du));
}

@compute @workgroup_size(256)
fn latin_hypercube_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // One workgroup per dimension; the 256 threads inside it split the samples.
    // This MUST read the workgroup and local ids separately: a global id would
    // conflate the two, so every thread but the first would fall out of the
    // dimension bound and whole strata would be left unwritten.
    let dim = wid.x;
    let n = params.n_samples;
    let stride = params.dimension;

    // Never taken: the launcher dispatches exactly one workgroup per dimension.
    // It is a guard, not control flow the barrier below may sit inside, so the
    // barrier stays at the top level of the function and is reached by every
    // invocation of the workgroup.
    let dim_in_range = dim < params.dimension;

    // Pass 1: lane 0 builds a genuine Fisher-Yates permutation of [0, n) for
    // THIS dimension. The shuffle is inherently sequential, and one lane doing
    // O(n) work per dimension is the same total work the CPU backend does.
    //
    // The permutation is staged in the output column itself, so no scratch
    // buffer and no host round trip is needed. Column `dim` is written by this
    // workgroup alone, so the staging cannot collide with another dimension.
    // Stratum indices round-trip exactly through f32 below 2^24 samples.
    if (dim_in_range && lid.x == 0u) {
        for (var i = 0u; i < n; i++) {
            output[i * stride + dim] = f32(i);
        }

        var counter = 0u;
        let key = permutation_key(params.seed, dim);
        var i = n;
        loop {
            if (i <= 1u) { break; }
            i -= 1u;

            // Uniform j in [0, i] by masked rejection. NO integer division or
            // modulo: `%` inside a loop miscompiles on this NVIDIA driver. i is
            // at least 1 here, so firstLeadingBit is defined, and the mask is
            // the smallest 2^k - 1 that covers i, giving at most one rejection
            // per draw on average.
            let mask = (2u << firstLeadingBit(i)) - 1u;
            var j = rng_next(key, &counter) & mask;
            var tries = 0u;
            loop {
                if (j <= i || tries >= 32u) { break; }
                j = rng_next(key, &counter) & mask;
                tries += 1u;
            }
            // Unreachable short of a 2^-32 run of rejections; keeps the loop
            // total and the index in range whatever the draws do.
            if (j > i) { j = i; }

            let a = i * stride + dim;
            let b = j * stride + dim;
            let tmp = output[a];
            output[a] = output[b];
            output[b] = tmp;
        }
    }
    // storageBarrier, NOT workgroupBarrier: the permutation is staged in a
    // storage buffer, and workgroupBarrier orders `var<workgroup>` accesses
    // only. Both are execution barriers for the workgroup; only this one also
    // makes lane 0's writes visible to the other lanes.
    storageBarrier();
    if (!dim_in_range) { return; }

    // Pass 2: every lane turns its share of the staged strata into samples.
    let inv_n = 1.0 / f32(n);
    var counter = 0u;
    let key = offset_key(params.seed, dim, lid.x);

    // Strided over the workgroup so the samples are covered whatever n_samples
    // is, including the tail when it is not a multiple of 256.
    for (var i = lid.x; i < n; i += 256u) {
        let idx = i * stride + dim;
        let interval = u32(output[idx]);
        let lower = f32(interval) * inv_n;
        let value = lower + rng_uniform(key, &counter) * inv_n;

        // The sum can round UP onto the stratum's upper boundary, which would
        // put two samples in one stratum and, for the last stratum, return
        // exactly 1.0 against a documented [0, 1). Clamp to the largest f32
        // below the boundary; for the last stratum that value is 0x1.fffffep-1,
        // the f32 twin of `DType::largest_value_below_one`.
        let upper = f32(interval + 1u) * inv_n;
        let upper_excl = bitcast<f32>(bitcast<u32>(upper) - 1u);
        output[idx] = min(value, upper_excl);
    }
}
"#;

/// Launches the Latin Hypercube Sampling (LHS) generator shader.
///
/// Generates stratified samples using Latin Hypercube Sampling: each dimension
/// gets its OWN Fisher-Yates permutation of the N strata, drawn on the device
/// from the seed in `params`, so the dimensions are independent and successive
/// calls draw different permutations.
///
/// # Arguments
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - Command queue for GPU execution
/// * `out` - Output buffer for generated samples
/// * `params` - Parameters buffer (dimension, n_samples, seed)
/// * `total_workgroups` - Total number of workgroups to dispatch
/// * `dtype` - Data type (must be floating-point)
pub fn launch_latin_hypercube(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params: &Buffer,
    total_workgroups: usize,
    dtype: DType,
) -> Result<()> {
    if total_workgroups == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "latin_hypercube")?;

    let name = "latin_hypercube_f32";
    let module = cache.get_or_create_module(name, LATIN_HYPERCUBE_WGSL);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("latin_hypercube"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("latin_hypercube"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per dimension
        pass.dispatch_workgroups(total_workgroups as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
