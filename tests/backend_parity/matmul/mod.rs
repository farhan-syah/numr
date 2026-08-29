// Backend parity tests for MatmulOps trait.
//
// `float` holds the dtype-parameterized float coverage. `supported_dtypes("cpu")`
// there never yields an integer dtype, so `integer_cuda` holds the hand-built
// CUDA-vs-CPU I32/I64 coverage that fills that hole.

pub mod float;
pub mod integer_cuda;
