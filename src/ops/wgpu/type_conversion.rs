//! Type conversion operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::tensor::Tensor;

impl WgpuClient {
    /// CPU-side type conversion for non-native WebGPU types.
    /// This handles conversions where source or target type is not natively
    /// supported by WGSL (e.g., I64, Bool, F64, F16, BF16, FP8).
    fn cast_via_cpu(
        &self,
        a: &Tensor<WgpuRuntime>,
        src_dtype: DType,
        dst_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        use crate::runtime::{RuntimeClient, ensure_contiguous};

        let a_contig = ensure_contiguous(a)?;
        let shape = a_contig.shape().to_vec();

        // Read raw bytes as f64 intermediary values, then write as target type.
        // We go through f64 to handle all source types uniformly.
        let f64_values: Vec<f64> = match src_dtype {
            DType::F32 => a_contig.to_vec::<f32>().iter().map(|&v| v as f64).collect(),
            DType::F64 => a_contig.to_vec::<f64>(),
            DType::I32 => a_contig.to_vec::<i32>().iter().map(|&v| v as f64).collect(),
            DType::I64 => a_contig.to_vec::<i64>().iter().map(|&v| v as f64).collect(),
            DType::U32 => a_contig.to_vec::<u32>().iter().map(|&v| v as f64).collect(),
            DType::Bool => a_contig
                .to_vec::<u8>()
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect(),
            #[cfg(feature = "f16")]
            DType::F16 => a_contig
                .to_vec::<half::f16>()
                .iter()
                .map(|&v| f64::from(f32::from(v)))
                .collect(),
            #[cfg(feature = "f16")]
            DType::BF16 => a_contig
                .to_vec::<half::bf16>()
                .iter()
                .map(|&v| f64::from(f32::from(v)))
                .collect(),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => {
                use crate::dtype::FP8E4M3;
                a_contig
                    .to_vec::<FP8E4M3>()
                    .iter()
                    .map(|&v| f64::from(v.to_f32()))
                    .collect()
            }
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => {
                use crate::dtype::FP8E5M2;
                a_contig
                    .to_vec::<FP8E5M2>()
                    .iter()
                    .map(|&v| f64::from(v.to_f32()))
                    .collect()
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: src_dtype,
                    op: "cast (WebGPU source type)",
                });
            }
        };

        // Convert f64 values to target type and create tensor
        let device = self.device();
        match dst_dtype {
            DType::F32 => {
                let data: Vec<f32> = f64_values.iter().map(|&v| v as f32).collect();
                Tensor::from_slice(&data, &shape, device)
            }
            DType::I32 => {
                let data: Vec<i32> = f64_values.iter().map(|&v| v as i32).collect();
                Tensor::from_slice(&data, &shape, device)
            }
            DType::U32 => {
                let data: Vec<u32> = f64_values.iter().map(|&v| v as u32).collect();
                Tensor::from_slice(&data, &shape, device)
            }
            DType::I64 => {
                let data: Vec<i64> = f64_values.iter().map(|&v| v as i64).collect();
                Tensor::from_slice(&data, &shape, device)
            }
            DType::F64 => Tensor::from_slice(&f64_values, &shape, device),
            DType::Bool => {
                let data: Vec<u8> = f64_values
                    .iter()
                    .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
                    .collect();
                Tensor::from_slice(&data, &shape, device)
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                let data: Vec<half::f16> =
                    f64_values.iter().map(|&v| half::f16::from_f64(v)).collect();
                Tensor::from_slice(&data, &shape, device)
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                let data: Vec<half::bf16> = f64_values
                    .iter()
                    .map(|&v| half::bf16::from_f64(v))
                    .collect();
                Tensor::from_slice(&data, &shape, device)
            }
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => {
                use crate::dtype::FP8E4M3;
                let data: Vec<FP8E4M3> = f64_values
                    .iter()
                    .map(|&v| FP8E4M3::from_f32(v as f32))
                    .collect();
                Tensor::from_slice(&data, &shape, device)
            }
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => {
                use crate::dtype::FP8E5M2;
                let data: Vec<FP8E5M2> = f64_values
                    .iter()
                    .map(|&v| FP8E5M2::from_f32(v as f32))
                    .collect();
                Tensor::from_slice(&data, &shape, device)
            }
            _ => Err(Error::UnsupportedDType {
                dtype: dst_dtype,
                op: "cast (WebGPU target type)",
            }),
        }
    }
}

impl TypeConversionOps<WgpuRuntime> for WgpuClient {
    fn cast(&self, a: &Tensor<WgpuRuntime>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let src_dtype = a.dtype();

        if src_dtype == dtype {
            return Ok(a.clone());
        }

        // WebGPU natively supports 32-bit types only (F32, I32, U32).
        // Casts between native types use WGSL shaders on-device.
        let wgpu_native = [DType::F32, DType::I32, DType::U32];
        let native_cast = wgpu_native.contains(&src_dtype) && wgpu_native.contains(&dtype);

        if native_cast {
            use crate::runtime::wgpu::ops::native::native_cast_op;
            return native_cast_op(self, a, dtype);
        }

        // Non-native type conversion: CPU-side boundary conversion.
        // Types like I64, Bool, F64, F16, BF16, FP8 can't be processed by WGSL shaders,
        // but data may arrive in these formats (e.g., I64 indices) or be requested as output.
        // We read the raw bytes back, convert on CPU, and create a new tensor.
        // This is NOT a forbidden GPU↔CPU transfer - the data was never on GPU in usable form.
        self.cast_via_cpu(a, src_dtype, dtype)
    }
}
