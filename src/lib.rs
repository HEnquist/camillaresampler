//! An audio sample rate conversion library for Rust.
//!
//! This library provides resamplers to process audio in chunks.
//!
//! The ratio between input and output sample rates is completely free.
//! Implementations are available that accept a fixed length input
//! while returning a variable length output, and vice versa.
//!
//! ## Asynchronous resampling
//! The resampling is based on band-limited interpolation using sinc
//! interpolation filters. The sinc interpolation upsamples by an adjustable factor,
//! and then the new sample points are calculated by interpolating between these points.
//! The resampling ratio can be updated at any time.
//!
//! ## Synchronous resampling
//! Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
//! and then inverse FFT:ed to get the resampled data.
//! This type of resampler is considerably faster but doesn't support changing the resampling ratio.
//!
//! ## SIMD acceleration
//! The asynchronous resampler is designed to benefit from auto-vectorization, meaning that the Rust compiler
//! can recognize calculations that can be done in parallel. It will then use SIMD instructions for those.
//! This works quite well, but there is still room for improvement.
//! On x86_64 it will always use SSE3 if available. The speed benefit compared to auto-vectorization
//! depends on the CPU, but tends to be in the range 20-30% for 64-bit data, and 50-100% for 32-bit data.
//!
//! ## Cargo features
//! #### `avx`: AVX on x86_64
//! The `avx` feature is enabled by default, and enables the use of AVX when it's available.
//! The speed increase compared to SSE depends on the CPU, and tends to range from zero to 50%.
//! On other architectures than x86_64 the `avx` feature does nothing.
//!
//! #### `neon`: Experimental Neon support on aarch64
//! Experimental support for Neon is available for aarch64 (64-bit Arm) by enabling the `neon` feature.
//! This requires the use of a nightly compiler, as the Neon support in Rust is still experimental.
//! On a Raspberry Pi 4, this gives a boost of about 10% for 64-bit floats and 50% for 32-bit floats when
//! compared to the auto-vectorized implementation.
//! Note that this only works on a full 64-bit operating system.
//!
//! ## Documentation
//!
//! The full documentation can be generated by rustdoc. To generate and view it run:
//! ```text
//! cargo doc --open
//! ```
//!
//! ## Example
//! Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
//! See also the "fixedin64" example that can be used to process a file from disk.
//! ```
//! use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};
//! let params = InterpolationParameters {
//!     sinc_len: 256,
//!     f_cutoff: 0.95,
//!     interpolation: InterpolationType::Linear,
//!     oversampling_factor: 256,
//!     window: WindowFunction::BlackmanHarris2,
//! };
//! let mut resampler = SincFixedIn::<f64>::new(
//!     48000 as f64 / 44100 as f64,
//!     params,
//!     1024,
//!     2,
//! );
//!
//! let waves_in = vec![vec![0.0f64; 1024];2];
//! let waves_out = resampler.process(&waves_in).unwrap();
//! ```
//!
//! ## Compatibility
//!
//! The `rubato` crate requires rustc version 1.40 or newer.

#![cfg_attr(feature = "neon", feature(aarch64_target_feature))]
#![cfg_attr(feature = "neon", feature(stdsimd))]

mod asynchro;
mod error;
mod interpolation;
mod sample;
mod sinc;
mod synchro;
mod windows;

pub use crate::asynchro::{ScalarInterpolator, SincFixedIn, SincFixedOut};
pub use crate::error::{CpuFeature, MissingCpuFeature, ResampleError, ResampleResult};
pub use crate::sample::Sample;
pub use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut};
pub use crate::windows::WindowFunction;

/// Helper macro to define a dummy implementation of the sample trait if a
/// feature is not supported.
macro_rules! interpolator {
    (
    #[cfg($($cond:tt)*)]
    mod $mod:ident;
    trait $trait:ident;
    ) => {
        #[cfg($($cond)*)]
        pub mod $mod;

        #[cfg($($cond)*)]
        use self::$mod::$trait;

        /// Dummy trait when not supported.
        #[cfg(not($($cond)*))]
        pub trait $trait {
        }

        /// Dummy impl of trait when not supported.
        #[cfg(not($($cond)*))]
        impl<T> $trait for T where T: Sample {
        }
    }
}

interpolator! {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    mod interpolator_avx;
    trait AvxSample;
}

interpolator! {
    #[cfg(target_arch = "x86_64")]
    mod interpolator_sse;
    trait SseSample;
}

interpolator! {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    mod interpolator_neon;
    trait NeonSample;
}

#[macro_use]
extern crate log;

/// A struct holding the parameters for interpolation.
#[derive(Debug)]
pub struct InterpolationParameters {
    /// Length of the windowed sinc interpolation filter.
    /// Higher values can allow a higher cut-off frequency leading to less high frequency roll-off
    /// at the expense of higher cpu usage. 256 is a good starting point.
    /// The value will be rounded up to the nearest multiple of 8.
    pub sinc_len: usize,
    /// Relative cutoff frequency of the sinc interpolation filter
    /// (relative to the lowest one of fs_in/2 or fs_out/2). Start at 0.95, and increase if needed.
    pub f_cutoff: f32,
    /// The number of intermediate points to use for interpolation.
    /// Higher values use more memory for storing the sinc filters.
    /// Only the points actually needed are calculated dusing processing
    /// so a larger number does not directly lead to higher cpu usage.
    /// But keeping it down helps in keeping the sincs in the cpu cache. Start at 128.
    pub oversampling_factor: usize,
    /// Interpolation type, see `InterpolationType`
    pub interpolation: InterpolationType,
    /// Window function to use.
    pub window: WindowFunction,
}

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between inut and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of cpu time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then the new value is calculated by interpolation between those points.
#[derive(Debug)]
pub enum InterpolationType {
    /// For cubic interpolation, the four nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then a cubic polynomial is fitted to these points, and is then used to calculate the new sample value.
    /// The computation time as about twice the one for linear interpolation,
    /// but it requires much fewer intermediate points for a good result.
    Cubic,
    /// With linear interpolation the new sample value is calculated by linear interpolation
    /// between the two nearest points.
    /// This requires two intermediate points to be calcuated using sinc interpolation,
    /// and te output is a weighted average of these two.
    /// This is relatively fast, but needs a large number of intermediate points to
    /// push the resampling artefacts below the noise floor.
    Linear,
    /// The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point.
    /// This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2,
    /// like 48kHz->96kHz.
    /// Then setting the oversampling_factor to 2, and using Nearest mode,
    /// no unneccesary computations are performed and the result is the same as for synchronous resampling.
    /// This also works for other ratios that can be expressed by a fraction. For 44.1kHz -> 48 kHz,
    /// setting oversampling_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz).
    Nearest,
}

/// A resampler that us used to resample a chunk of audio to a new sample rate.
/// The rate can be adjusted as required.
pub trait Resampler<T> {
    /// Resample a chunk of audio. Input and output data is stored in a vector,
    /// where each element contains a vector with all samples for a single channel.
    fn process<V: AsRef<[T]>>(&mut self, wave_in: &[V]) -> ResampleResult<Vec<Vec<T>>>;

    /// Query for the number of frames needed for the next call to "process".
    fn nbr_frames_needed(&self) -> usize;

    /// Update the resample ratio.
    fn set_resample_ratio(&mut self, new_ratio: f64) -> ResampleResult<()>;

    /// Update the resample ratio relative to the original one.
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> ResampleResult<()>;
}
