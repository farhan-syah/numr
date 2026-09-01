// Activation helpers shared by the GEMM epilogue kernels.
//
// activation_type: 0=None, 1=ReLU, 2=GELU, 3=SiLU, 4=Sigmoid, 5=Tanh —
// the codes `activation_to_u32` emits in
// src/runtime/cuda/kernels/gemm_epilogue/launcher.rs.
//
// The generic epilogue kernels (gemm_epilogue.cu) and the WMMA tensor-core
// epilogue kernels (matmul_wmma.cu) both call apply_activation_f32, so the
// two paths cannot produce different numbers for the same activation.

#ifndef NUMR_GEMM_ACTIVATION_CUH
#define NUMR_GEMM_ACTIVATION_CUH

__device__ __forceinline__ float apply_activation_f32(float x, unsigned int act_type) {
    switch (act_type) {
        case 0: return x; // None
        case 1: return fmaxf(x, 0.0f); // ReLU
        case 2: { // GELU
            const float sqrt_2_over_pi = 0.7978845608f;
            const float coef = 0.044715f;
            float inner = sqrt_2_over_pi * (x + coef * x * x * x);
            return 0.5f * x * (1.0f + tanhf(inner));
        }
        case 3: { // SiLU
            return x / (1.0f + expf(-x));
        }
        case 4: { // Sigmoid
            return 1.0f / (1.0f + expf(-x));
        }
        case 5: { // Tanh
            return tanhf(x);
        }
        default: return x;
    }
}

__device__ __forceinline__ double apply_activation_f64(double x, unsigned int act_type) {
    switch (act_type) {
        case 0: return x;
        case 1: return fmax(x, 0.0);
        case 2: {
            const double sqrt_2_over_pi = 0.7978845608028654;
            const double coef = 0.044715;
            double inner = sqrt_2_over_pi * (x + coef * x * x * x);
            return 0.5 * x * (1.0 + tanh(inner));
        }
        case 3: return x / (1.0 + exp(-x));
        case 4: return 1.0 / (1.0 + exp(-x));
        case 5: return tanh(x);
        default: return x;
    }
}

#endif  // NUMR_GEMM_ACTIVATION_CUH
