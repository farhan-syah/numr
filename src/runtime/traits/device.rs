//! Trait for device identification

use super::profile::DeviceProfile;

/// Trait for device identification
pub trait Device: Clone + Send + Sync + 'static {
    /// Unique identifier for this device
    fn id(&self) -> usize;

    /// Check if two devices are the same
    fn is_same(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Human-readable name
    fn name(&self) -> String {
        format!("Device({})", self.id())
    }

    /// Real hardware capability snapshot for kernel/tile selection.
    ///
    /// Default is `DeviceProfile::unknown()` so existing backends keep
    /// compiling without an override; a backend that wants callers to make
    /// informed kernel choices must query and cache its own real values.
    fn profile(&self) -> DeviceProfile {
        DeviceProfile::unknown("unknown")
    }
}
