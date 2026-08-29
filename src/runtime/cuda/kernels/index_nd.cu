// Coordinate-addressed indexing CUDA kernels: gather_nd, gather_2d,
// slice_assign — three kernels per dtype from one row macro.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8, bool
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/index/ from dtype_suffix() in loader.rs:
// {op}_{suffix}, e.g. gather_nd_u32.
//
// This is its own translation unit — PTX module "index_nd", see
// kernel_names::INDEX_ND_MODULE — because index.cu is at its size limit with
// the per-element indexing family alone. The bodies live in index_nd_ops.cuh.

#include "index_nd_ops.cuh"

extern "C" {

NUMR_INDEX_ND_ROW(float, f32)
NUMR_INDEX_ND_ROW(double, f64)
NUMR_INDEX_ND_ROW(__half, f16)
NUMR_INDEX_ND_ROW(__nv_bfloat16, bf16)
NUMR_INDEX_ND_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_INDEX_ND_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_INDEX_ND_ROW(int64_t, i64)
NUMR_INDEX_ND_ROW(int32_t, i32)
NUMR_INDEX_ND_ROW(int16_t, i16)
NUMR_INDEX_ND_ROW(int8_t, i8)
NUMR_INDEX_ND_ROW(uint64_t, u64)
NUMR_INDEX_ND_ROW(uint32_t, u32)
NUMR_INDEX_ND_ROW(uint16_t, u16)
NUMR_INDEX_ND_ROW(uint8_t, u8)
NUMR_INDEX_ND_ROW(unsigned char, bool)

} // extern "C"
