//! Backend-independent device capability description
//!
//! Kernel and tile selection needs more than dtype+shape: it needs to know what
//! the device can actually do (tensor cores, dp4a, shared memory budget). This
//! module exists so that decision can be made once, from real hardware data,
//! instead of hardcoded per-op constants that silently mis-select on GPUs the
//! author never tested against.

/// GPU/CPU microarchitecture generation, coarse enough to gate kernel choice
/// without hardcoding a `(major, minor)` compute-capability pair at every
/// call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceArch {
    /// Host CPU.
    Cpu,
    /// WebGPU, whose adapter reports no microarchitecture.
    Wgpu,
    /// sm_61..sm_70. First generation with `dp4a`.
    CudaPascal,
    /// sm_70..sm_75. Adds f16 tensor cores.
    CudaVolta,
    /// sm_75..sm_80. Adds int8 tensor cores (`mma.*.s8.s8.s32`).
    CudaTuring,
    /// sm_80..sm_89. Adds native bf16.
    CudaAmpere,
    /// sm_89..sm_90.
    CudaAda,
    /// sm_90..sm_100.
    CudaHopper,
    /// sm_100 and newer.
    CudaBlackwell,
    /// Recognized CUDA device but below sm_61, or a compute capability newer
    /// than anything mapped here.
    CudaUnknown,
    /// Backend/arch could not be determined.
    #[default]
    Unknown,
}

/// Hardware instructions relevant to kernel selection.
///
/// Named for the instruction, not the SM generation that introduced it —
/// callers should ask "does this device have dp4a" not "is this sm_75+".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceCaps {
    /// Packed int8 dot product (`__dp4a`, CUDA sm_61+).
    pub dp4a: bool,
    /// int8 tensor-core `mma` (CUDA sm_75+).
    pub int8_mma: bool,
    /// int8 tensor-core `mma` at the `m16n8k32` shape (CUDA sm_80+).
    /// `int8_mma` alone is sm_75, where only `m8n8k16` exists.
    pub int8_mma_m16n8k32: bool,
    /// f16 tensor-core `mma` (CUDA sm_70+).
    pub f16_mma: bool,
    /// Native bf16 arithmetic (CUDA sm_80+).
    pub bf16: bool,
}

/// Device capability snapshot used for kernel and tile selection.
///
/// Every field is a physical property of the device, not a policy choice —
/// policy (which kernel, which tile size) is decided by the caller from
/// these numbers. `unknown()` gives a conservative value so a backend that
/// has not implemented `Device::profile()` degrades to "assume nothing"
/// rather than panicking or guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Backend identity: "cpu" | "cuda" | "wgpu".
    pub backend: &'static str,
    /// Microarchitecture generation, for gating instruction-set choices.
    pub arch: DeviceArch,
    /// CUDA SMs, AMD CUs, CPU cores.
    pub compute_units: u32,
    /// Bytes; 0 where the concept does not apply (e.g. CPU).
    pub shared_mem_per_block: u32,
    /// Bytes per compute unit.
    pub shared_mem_per_unit: u32,
    /// Upper bound on a single launch's block size.
    pub max_threads_per_block: u32,
    /// Warp 32 / wavefront 64 / 1 for CPU.
    pub lane_width: u32,
    /// Instruction-set features a kernel may rely on.
    pub caps: DeviceCaps,
}

impl DeviceProfile {
    /// Conservative fallback for a device whose real capabilities were not
    /// queried (backend without a `profile()` override, or a failed query).
    /// Every cap is false and every count is zero so a caller that blindly
    /// trusts this value picks the safest, most portable kernel path.
    pub fn unknown(backend: &'static str) -> Self {
        Self {
            backend,
            arch: DeviceArch::Unknown,
            compute_units: 0,
            shared_mem_per_block: 0,
            shared_mem_per_unit: 0,
            max_threads_per_block: 0,
            lane_width: 0,
            caps: DeviceCaps::default(),
        }
    }

    /// Map a CUDA `(major, minor)` compute capability to `(DeviceArch, DeviceCaps)`.
    ///
    /// Pure function of the two version numbers so it is testable without a
    /// GPU present. Boundaries follow NVIDIA's SM generation ranges:
    /// Pascal 6.1–6.x, Volta 7.0–7.4, Turing 7.5–7.x, Ampere 8.0–8.8,
    /// Ada 8.9, Hopper 9.0–9.x, Blackwell 10.0+. Below 6.1 has none of the
    /// tracked instructions.
    pub fn arch_and_caps_for_compute_capability(
        major: u32,
        minor: u32,
    ) -> (DeviceArch, DeviceCaps) {
        let cc = (major, minor);
        if cc < (6, 1) {
            (DeviceArch::CudaUnknown, DeviceCaps::default())
        } else if cc < (7, 0) {
            (
                DeviceArch::CudaPascal,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: false,
                    int8_mma_m16n8k32: false,
                    f16_mma: false,
                    bf16: false,
                },
            )
        } else if cc < (7, 5) {
            (
                DeviceArch::CudaVolta,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: false,
                    int8_mma_m16n8k32: false,
                    f16_mma: true,
                    bf16: false,
                },
            )
        } else if cc < (8, 0) {
            (
                DeviceArch::CudaTuring,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: true,
                    int8_mma_m16n8k32: false,
                    f16_mma: true,
                    bf16: false,
                },
            )
        } else if cc < (8, 9) {
            (
                DeviceArch::CudaAmpere,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: true,
                    int8_mma_m16n8k32: true,
                    f16_mma: true,
                    bf16: true,
                },
            )
        } else if cc < (9, 0) {
            (
                DeviceArch::CudaAda,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: true,
                    int8_mma_m16n8k32: true,
                    f16_mma: true,
                    bf16: true,
                },
            )
        } else if cc < (10, 0) {
            (
                DeviceArch::CudaHopper,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: true,
                    int8_mma_m16n8k32: true,
                    f16_mma: true,
                    bf16: true,
                },
            )
        } else {
            (
                DeviceArch::CudaBlackwell,
                DeviceCaps {
                    dp4a: true,
                    int8_mma: true,
                    int8_mma_m16n8k32: true,
                    f16_mma: true,
                    bf16: true,
                },
            )
        }
    }
}

/// The CUDA query path, which the pure mapping tests above cannot reach.
///
/// `CudaDevice::profile()` swallows a driver error and returns `unknown()`, so
/// a mistyped attribute would degrade silently and forever with nothing
/// failing. These assertions are machine-independent — they check that the
/// query RAN, not what this particular card is.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::runtime::Device;
    use crate::runtime::cuda::CudaDevice;
    use crate::runtime::traits::profile::DeviceArch;

    #[test]
    fn cuda_profile_is_queried_not_defaulted() {
        let profile = CudaDevice::new(0).profile();

        assert_eq!(profile.backend, "cuda", "backend must identify as cuda");
        assert_ne!(
            profile.arch,
            DeviceArch::Unknown,
            "arch is Unknown, so the compute-capability query fell back"
        );
        assert!(
            profile.compute_units > 0,
            "compute_units is 0, so MULTIPROCESSOR_COUNT was not read"
        );
        assert_eq!(profile.lane_width, 32, "every CUDA device has warp 32");
        assert!(
            profile.shared_mem_per_block > 0 && profile.shared_mem_per_unit > 0,
            "shared memory attributes were not read"
        );
        assert!(
            profile.max_threads_per_block >= 1024,
            "every CUDA device since Fermi allows 1024 threads per block"
        );

        // int8 tensor cores imply dp4a; a profile claiming otherwise has its
        // capability mapping inverted.
        if profile.caps.int8_mma {
            assert!(profile.caps.dp4a, "int8_mma without dp4a is not a real GPU");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps_none() -> DeviceCaps {
        DeviceCaps::default()
    }

    #[test]
    fn below_pascal_has_no_caps() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(6, 0);
        assert_eq!(arch, DeviceArch::CudaUnknown);
        assert_eq!(caps, caps_none());
    }

    #[test]
    fn pascal_boundary_6_1() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(6, 1);
        assert_eq!(arch, DeviceArch::CudaPascal);
        assert!(caps.dp4a);
        assert!(!caps.f16_mma);
        assert!(!caps.int8_mma);
        assert!(!caps.int8_mma_m16n8k32);
        assert!(!caps.bf16);
    }

    #[test]
    fn volta_boundary_7_0() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(7, 0);
        assert_eq!(arch, DeviceArch::CudaVolta);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(!caps.int8_mma);
        assert!(!caps.int8_mma_m16n8k32);
        assert!(!caps.bf16);
    }

    #[test]
    fn turing_boundary_7_5() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(7, 5);
        assert_eq!(arch, DeviceArch::CudaTuring);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(caps.int8_mma);
        assert!(!caps.int8_mma_m16n8k32);
        assert!(!caps.bf16);
    }

    #[test]
    fn ampere_boundary_8_0() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(8, 0);
        assert_eq!(arch, DeviceArch::CudaAmpere);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(caps.int8_mma);
        assert!(caps.int8_mma_m16n8k32);
        assert!(caps.bf16);
    }

    #[test]
    fn ampere_8_6_stays_ampere() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(8, 6);
        assert_eq!(arch, DeviceArch::CudaAmpere);
        assert!(caps.int8_mma_m16n8k32);
        assert!(caps.bf16);
    }

    #[test]
    fn ada_boundary_8_9() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(8, 9);
        assert_eq!(arch, DeviceArch::CudaAda);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(caps.int8_mma);
        assert!(caps.int8_mma_m16n8k32);
        assert!(caps.bf16);
    }

    #[test]
    fn hopper_boundary_9_0() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(9, 0);
        assert_eq!(arch, DeviceArch::CudaHopper);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(caps.int8_mma);
        assert!(caps.int8_mma_m16n8k32);
        assert!(caps.bf16);
    }

    #[test]
    fn blackwell_boundary_10_0() {
        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(10, 0);
        assert_eq!(arch, DeviceArch::CudaBlackwell);
        assert!(caps.dp4a);
        assert!(caps.f16_mma);
        assert!(caps.int8_mma);
        assert!(caps.int8_mma_m16n8k32);
        assert!(caps.bf16);
    }

    #[test]
    fn unknown_profile_is_conservative() {
        let p = DeviceProfile::unknown("cuda");
        assert_eq!(p.backend, "cuda");
        assert_eq!(p.arch, DeviceArch::Unknown);
        assert_eq!(p.compute_units, 0);
        assert_eq!(p.caps, DeviceCaps::default());
    }
}
