// Integer unary CUDA kernels (I8, I16, I32, I64, U8, U16, U32, U64)
//
// Split out of `unary.cu` for the same reason `reduce_int.cu`,
// `cumulative_int.cu`, `matmul_int.cu` and `gemv_int.cu` were: the integer
// dtypes need different arithmetic from the float ones, and `unary.cu` is far
// past its size limit. `unary_module` in
// `runtime/cuda/kernels/loader/dtype_modules.rs` routes every integer dtype
// here. The kernel names and the launch ABI are identical to `unary.cu`'s, so
// that is a swap of module, never of kernel name.
//
// Which unary ops an integer dtype gets:
//
//   neg      every integer dtype. On an unsigned one it is `0 - a` in modular
//            arithmetic, so `neg(1u32)` is `u32::MAX`, matching CPU, WebGPU,
//            this crate's unsigned `sub`, and NumPy.
//   abs      every integer dtype. The identity on the unsigned ones.
//   sign     every integer dtype. -1/0/1 signed, 0/1 unsigned.
//   square   every integer dtype.
//
// The transcendentals (exp, log, sqrt, the trig and hyperbolic families) and
// the rounding ops are float-only in this crate and have no instantiation for
// any integer dtype in any backend.
//
// Element-wise integer ops WRAP. That is the convention documented in
// `src/runtime/cpu/kernels/wide_acc.rs`, implemented on CPU by
// `src/runtime/cpu/kernels/unary/int.rs`, and shared with the binary ops in
// `binary_ops.cuh`. So `neg`, `abs` and `square` of a value that leaves the
// range answer modulo 2^N rather than saturating, and `neg(i32::MIN)` and
// `abs(i32::MIN)` are both `i32::MIN`.
//
// Signed overflow is undefined behaviour in C++, so every signed operation runs
// in the dtype's unsigned counterpart and converts back, which is exactly the
// idiom `binary_ops.cuh` uses. That conversion is modulo 2^N, which is what
// Rust's `wrapping_neg`, `wrapping_abs` and `wrapping_mul` produce. A bare
// `-a`, `abs(a)` or `a * a` would be UB on the extremes instead.

// ============================================================================
// Templated device functions (outside extern "C")
// ============================================================================

// T is the signed element type, U its unsigned counterpart of the same width.
template<typename T, typename U>
__device__ __forceinline__ T numr_int_neg(T a) {
    return (T)(U)((U)0 - (U)a);
}

template<typename T, typename U>
__device__ __forceinline__ T numr_int_abs(T a) {
    return (a < (T)0) ? (T)(U)((U)0 - (U)a) : a;
}

template<typename T>
__device__ __forceinline__ T numr_int_sign(T a) {
    return (a > (T)0) ? (T)1 : ((a < (T)0) ? (T)(-1) : (T)0);
}

template<typename T, typename U>
__device__ __forceinline__ T numr_int_square(T a) {
    return (T)(U)((U)a * (U)a);
}

// The unsigned counterparts. Written separately rather than reusing the ones
// above: `a < 0` and `(T)(-1)` on an unsigned type are dead branches that nvcc
// warns about, and `abs` is the identity there rather than a computation.
template<typename T>
__device__ __forceinline__ T numr_uint_neg(T a) {
    // Unsigned subtraction is already modulo 2^N in C++, never UB.
    return (T)((T)0 - a);
}

template<typename T>
__device__ __forceinline__ T numr_uint_sign(T a) {
    return (a > (T)0) ? (T)1 : (T)0;
}

template<typename T>
__device__ __forceinline__ T numr_uint_square(T a) {
    // Unsigned multiplication is already modulo 2^N in C++, never UB.
    return (T)(a * a);
}

// ============================================================================
// Instantiation matrix
// ============================================================================

#define NUMR_UNARY_INT_SIGNED(T, U, SUFFIX) \
extern "C" __global__ void neg_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_int_neg<T, U>(a[idx]); } \
} \
extern "C" __global__ void abs_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_int_abs<T, U>(a[idx]); } \
} \
extern "C" __global__ void sign_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_int_sign<T>(a[idx]); } \
} \
extern "C" __global__ void square_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_int_square<T, U>(a[idx]); } \
}

#define NUMR_UNARY_INT_UNSIGNED(T, SUFFIX) \
extern "C" __global__ void neg_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_uint_neg<T>(a[idx]); } \
} \
extern "C" __global__ void abs_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = a[idx]; } \
} \
extern "C" __global__ void sign_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_uint_sign<T>(a[idx]); } \
} \
extern "C" __global__ void square_##SUFFIX(const T* a, T* out, unsigned int n) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { out[idx] = numr_uint_square<T>(a[idx]); } \
}

NUMR_UNARY_INT_SIGNED(signed char, unsigned char, i8)
NUMR_UNARY_INT_SIGNED(short, unsigned short, i16)
NUMR_UNARY_INT_SIGNED(int, unsigned int, i32)
NUMR_UNARY_INT_SIGNED(long long, unsigned long long, i64)

NUMR_UNARY_INT_UNSIGNED(unsigned char, u8)
NUMR_UNARY_INT_UNSIGNED(unsigned short, u16)
NUMR_UNARY_INT_UNSIGNED(unsigned int, u32)
NUMR_UNARY_INT_UNSIGNED(unsigned long long, u64)
