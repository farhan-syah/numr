// embedding_lookup operation for i32
// Industry-standard embedding table lookup used in neural networks.
// Each thread handles one index lookup and copies the full embedding row.

const WORKGROUP_SIZE: u32 = 256u;

struct EmbeddingLookupParams {
    num_indices: u32,
    vocab_size: u32,
    embedding_dim: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> embeddings: array<i32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<uniform> params: EmbeddingLookupParams;

@compute @workgroup_size(256)
fn embedding_lookup_i32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_indices) {
        return;
    }

    let index_val = indices[idx];

    // Check bounds
    if (index_val < 0 || u32(index_val) >= params.vocab_size) {
        // Out of bounds - fill with zeros
        let out_start = idx * params.embedding_dim;
        for (var i: u32 = 0u; i < params.embedding_dim; i = i + 1u) {
            output[out_start + i] = 0;
        }
        return;
    }

    // Copy the entire embedding row to output
    let emb_start = u32(index_val) * params.embedding_dim;
    let out_start = idx * params.embedding_dim;
    for (var i: u32 = 0u; i < params.embedding_dim; i = i + 1u) {
        output[out_start + i] = embeddings[emb_start + i];
    }
}
