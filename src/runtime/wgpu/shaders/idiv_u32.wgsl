// Integer division for the U32 binary, broadcast and scalar shaders.
//
// The crate's contract, stated in `runtime/cpu/kernels/binary_int.rs`, is that
// a zero divisor yields 0. WGSL instead defines `e1 / 0` as `e1`, so the bare
// operator returns the dividend where CPU and CUDA
// (`NUMR_BINOP_INT_DIV_UNSIGNED` in `runtime/cuda/kernels/binary_ops.cuh`) both
// return 0. There is no overflow case: no unsigned quotient leaves u32.
//
// Every call site is one division outside any loop. Do NOT call this from
// inside a loop: an integer divide in a WGSL loop fails NVIDIA's shader
// compiler with "NVVM compilation failed", which is why `int_saturate.wgsl`
// bans division outright.
fn numr_idiv_u32(a: u32, b: u32) -> u32 {
    if (b == 0u) {
        return 0u;
    }
    return a / b;
}
