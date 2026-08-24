//! CPU runtime implementation

use super::client::{CpuAllocator, CpuClient};
use super::device::CpuDevice;
use crate::runtime::{NoOpGraph, Runtime};
use std::alloc::{Layout as AllocLayout, alloc, dealloc};

/// CPU compute runtime
///
/// This is the default runtime that works on any platform.
/// Memory is allocated on the heap using the system allocator.
#[derive(Clone, Debug, Default)]
pub struct CpuRuntime;

impl Runtime for CpuRuntime {
    type Device = CpuDevice;
    type Client = CpuClient;
    type Allocator = CpuAllocator;
    type Graph = NoOpGraph;
    type RawHandle = ();
    type DType = crate::dtype::DType;

    fn name() -> &'static str {
        "cpu"
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // Use aligned allocation for SIMD compatibility
        let align = 64; // AVX-512 alignment
        let layout = AllocLayout::from_size_align(size_bytes, align)
            .map_err(|_| crate::error::Error::OutOfMemory { size: size_bytes })?;

        // Use alloc (not alloc_zeroed) — Tensor::empty is explicitly uninitialized.
        // Operations that need zeroed memory (e.g. Tensor::zeros) handle zeroing themselves.
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(crate::error::Error::OutOfMemory { size: size_bytes });
        }

        Ok(ptr as u64)
    }

    fn deallocate(ptr: u64, size_bytes: usize, _device: &Self::Device) {
        if ptr == 0 || size_bytes == 0 {
            return;
        }

        let align = 64;
        // A layout this size could never have been allocated (allocate() rejects
        // it), so there is nothing to free. Returning is the only sound option —
        // panicking in a Drop path would abort the process.
        let Ok(layout) = AllocLayout::from_size_align(size_bytes, align) else {
            return;
        };

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn copy_to_device(src: &[u8], dst: u64, _device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
        Ok(())
    }

    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
        }

        unsafe {
            // Use copy (not copy_nonoverlapping) in case src and dst overlap
            std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
        }
        Ok(())
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        _device: &Self::Device,
    ) -> crate::error::Result<()> {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return Ok(());
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(());
        }

        // For CPU, we can use pointer arithmetic directly
        let src_base = (src_handle as usize + src_byte_offset) as *const u8;
        let dst_base = dst_handle as *mut u8;

        // Iterate over all elements using indices
        let mut indices = vec![0usize; shape.len()];

        for dst_offset in 0..numel {
            // Calculate source byte offset for current indices
            let mut src_elem_offset: isize = 0;
            for (i, &idx) in indices.iter().enumerate() {
                src_elem_offset += (idx as isize) * strides[i];
            }

            // Copy element
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_base.offset(src_elem_offset * elem_size as isize),
                    dst_base.add(dst_offset * elem_size),
                    elem_size,
                );
            }

            // Increment indices (row-major order)
            for dim in (0..shape.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < shape[dim] {
                    break;
                }
                indices[dim] = 0;
            }
        }
        Ok(())
    }

    fn default_device() -> Self::Device {
        CpuDevice::new()
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        CpuClient::new(device.clone())
    }

    fn raw_handle(_client: &Self::Client) -> &Self::RawHandle {
        &()
    }

    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&crate::tensor::Tensor<Self>],
        outputs: &[&crate::tensor::Tensor<Self>],
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> crate::error::Result<()>,
    {
        f(client)?;
        let owned_inputs: Vec<_> = inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<_> = outputs.iter().map(|t| (*t).clone()).collect();
        Ok(crate::runtime::CapturedGraph::new(
            crate::runtime::NoOpGraph,
            owned_inputs,
            owned_outputs,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    #[test]
    fn test_cpu_supports_graph_capture() {
        assert!(!CpuRuntime::supports_graph_capture());
    }

    #[test]
    fn test_cpu_capture_graph_into_executes_eagerly() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Allocate a 4-element f32 output tensor.
        let out =
            crate::tensor::Tensor::<CpuRuntime>::try_zeros(&[4], crate::dtype::DType::F32, &device)
                .unwrap();
        let input = crate::tensor::Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[4],
            &device,
        )
        .unwrap();

        let mut closure_ran = false;
        let graph = CpuRuntime::capture_graph_into(&client, &[&input], &[&out], |_c| {
            closure_ran = true;
            // Copy input bytes into `out` in-place — visible after capture.
            let size = input.numel() * input.dtype().size_in_bytes();
            CpuRuntime::copy_within_device(input.ptr(), out.ptr(), size, input.device())
        })
        .expect("cpu capture_graph_into should succeed");

        assert!(closure_ran, "closure must run eagerly");

        // The copy performed inside the closure must be visible.
        let result = out.to_vec::<f32>();
        assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);

        // launch() is a no-op on CPU — should not error.
        graph.launch().expect("launch should be a no-op on CPU");
    }

    #[test]
    fn test_cpu_capture_graph_into_propagates_error() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let result = CpuRuntime::capture_graph_into(&client, &[], &[], |_c| {
            Err(crate::error::Error::Backend(
                "sentinel error from closure".into(),
            ))
        });

        assert!(result.is_err(), "error from the closure must be propagated");
    }
}
