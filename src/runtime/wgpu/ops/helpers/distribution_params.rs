//! Params structs for the distribution sampling kernels, from multinomial
//! through the continuous families.

/// Params for multinomial sampling operation (with replacement)
/// Samples indices from categorical distributions defined by probability rows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MultinomialWithReplacementParams {
    pub(crate) num_distributions: u32,
    pub(crate) num_categories: u32,
    pub(crate) num_samples: u32,
    pub(crate) seed: u32,
}

/// Params for multinomial sampling operation (without replacement)
/// Samples indices from categorical distributions without replacement.
/// Uses workgroup shared memory for modified probabilities.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MultinomialWithoutReplacementParams {
    pub(crate) num_distributions: u32,
    pub(crate) num_categories: u32,
    pub(crate) num_samples: u32,
    pub(crate) seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BernoulliParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) p: f32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BetaDistParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GammaDistParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) shape: f32,
    pub(crate) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ExponentialParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) rate: f32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PoissonParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) lambda: f32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BinomialParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) n_trials: u32,
    pub(crate) p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LaplaceParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) loc: f32,
    pub(crate) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ChiSquaredParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) df: f32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StudentTParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) df: f32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FDistributionParams {
    pub(crate) numel: u32,
    pub(crate) seed: u32,
    pub(crate) df1: f32,
    pub(crate) df2: f32,
}
