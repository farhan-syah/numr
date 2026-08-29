// Backend parity tests for RandomOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Random operations produce backend-specific values - we test shape, dtype, and statistical
// properties rather than exact value parity.
//
// Split by seam: `distributions` covers unseeded shape/dtype/statistical invariants,
// `seeded` covers seed reproducibility and the seed-derivation regression coverage.

pub mod distributions;
pub mod seeded;
