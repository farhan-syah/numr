// Integer division for the I32 binary, broadcast and scalar shaders.
//
// The crate's contract, stated in `runtime/cpu/kernels/binary_int.rs`, is that
// a zero divisor yields 0 and `i32::MIN / -1` yields `i32::MIN`. WGSL instead
// defines `e1 / 0` as `e1`, so the bare operator returns the dividend where CPU
// and CUDA (`NUMR_BINOP_INT_DIV_SIGNED` in `runtime/cuda/kernels/binary_ops.cuh`)
// both return 0.
//
// WGSL also defines `i32::MIN / -1` as `i32::MIN`, which already agrees with
// CPU's `wrapping_div`. The case is still computed here rather than left to the
// operator, because the SPIR-V that naga emits leaves a signed division
// overflow undefined and the divisor is data, not a constant the shader can
// reason about. CUDA guards the same case for the same reason.
//
// Every call site is one division outside any loop. Do NOT call this from
// inside a loop: an integer divide in a WGSL loop fails NVIDIA's shader
// compiler with "NVVM compilation failed", which is why `int_saturate.wgsl`
// bans division outright.
fn numr_idiv_i32(a: i32, b: i32) -> i32 {
    if (b == 0) {
        return 0;
    }
    if (b == -1) {
        // Wrapping negation through u32, so i32::MIN comes out as itself
        // instead of overflowing.
        return bitcast<i32>(0u - bitcast<u32>(a));
    }
    return a / b;
}
