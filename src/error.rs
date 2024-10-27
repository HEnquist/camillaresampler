use std::error;
use std::fmt;

/// An identifier for a cpu feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    /// x86 sse3 cpu feature.
    #[cfg(target_arch = "x86_64")]
    Sse3,
    /// x86_64 avx cpu feature.
    #[cfg(target_arch = "x86_64")]
    Avx,
    /// the fma cpu feature.
    #[cfg(target_arch = "x86_64")]
    Fma,
    /// aarc64 neon cpu feature.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl CpuFeature {
    /// Test if the given CPU feature is detected.
    pub fn is_detected(&self) -> bool {
        match *self {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse3 => {
                is_x86_feature_detected!("sse3")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx => {
                is_x86_feature_detected!("avx")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Fma => {
                is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => {
                std::arch::is_aarch64_feature_detected!("neon")
            }
        }
    }
}

#[allow(unused_variables)]
impl fmt::Display for CpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse3 => {
                write!(f, "sse3")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx => {
                write!(f, "avx")
            }
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Fma => {
                write!(f, "fma")
            }
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => {
                write!(f, "neon")
            }
        }
    }
}

/// Error raised when trying to use a CPU feature which is not supported.
#[derive(Debug, Clone, Copy)]
pub struct MissingCpuFeature(pub(crate) CpuFeature);

impl fmt::Display for MissingCpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Missing CPU feature `{}`", self.0)
    }
}

impl error::Error for MissingCpuFeature {}

/// The error type returned when constructing [Resampler](crate::Resampler).
pub enum ResamplerConstructionError {
    InvalidSampleRate { input: usize, output: usize },
    InvalidRelativeRatio(f64),
    InvalidRatio(f64),
    InvalidChunkSize(usize),
}

impl fmt::Display for ResamplerConstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidSampleRate{input, output} => write!(formatter,
                "Input and output sample rates must both be > 0. Provided input: {}, provided output: {}", input, output
            ),
            Self::InvalidRatio(provided) => write!(formatter,
                "Invalid resample_ratio provided: {}. resample_ratio must be > 0", provided
            ),
            Self::InvalidRelativeRatio(provided) => write!(formatter,
                "Invalid max_resample_ratio_relative provided: {}. max_resample_ratio_relative must be >= 1", provided
            ),
            Self::InvalidChunkSize(provided) => write!(formatter,
                "Invalid chunk_size provided: {}. chunk_size must be >= 1", provided
            ),
        }
    }
}

impl fmt::Debug for ResamplerConstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self)
    }
}

impl error::Error for ResamplerConstructionError {}

/// The error type used by `rubato`.
pub enum ResampleError {
    /// Error raised when [Resampler::set_resample_ratio](crate::Resampler::set_resample_ratio)
    /// is called with a ratio outside the maximum range specified when
    /// the resampler was constructed.
    RatioOutOfBounds {
        provided: f64,
        original: f64,
        max_relative_ratio: f64,
    },
    /// Error raised when calling [Resampler::set_resample_ratio](crate::Resampler::set_resample_ratio)
    /// on a synchronous resampler.
    SyncNotAdjustable,
    /// Error raised when the number of channels in the input buffer doesn't match the value expected.
    WrongNumberOfInputChannels {
        expected: usize,
        actual: usize,
    },
    /// Error raised when the number of channels in the output buffer doesn't match the value expected.
    WrongNumberOfOutputChannels {
        expected: usize,
        actual: usize,
    },
    /// Error raised when the number of channels in the mask doesn't match the value expected.
    WrongNumberOfMaskChannels {
        expected: usize,
        actual: usize,
    },
    /// Error raised when the number of frames in an input channel is less
    /// than the minimum expected.
    InsufficientInputBufferSize {
        expected: usize,
        actual: usize,
    },
    /// Error raised when the number of frames in an output channel is less
    /// than the minimum expected.
    InsufficientOutputBufferSize {
        expected: usize,
        actual: usize,
    },
    InvalidChunkSize {
        max: usize,
        requested: usize,
    },
    ChunkSizeNotAdjustable,
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RatioOutOfBounds {
                provided,
                original,
                max_relative_ratio,
            } => {
                write!(f, "New resample ratio out of bounds. Provided ratio {}, original resample ratio {}, maximum relative ratio {}, allowed absolute range {} to {}",
                provided, original, max_relative_ratio, original / max_relative_ratio, original * max_relative_ratio)
            }
            Self::SyncNotAdjustable { .. } => {
                write!(f, "Not possible to adjust a synchronous resampler")
            }
            Self::WrongNumberOfInputChannels { expected, actual } => {
                write!(
                    f,
                    "Wrong number of channels {} in input, expected {}",
                    actual, expected
                )
            }
            Self::WrongNumberOfOutputChannels { expected, actual } => {
                write!(
                    f,
                    "Wrong number of channels {} in output, expected {}",
                    actual, expected
                )
            }
            Self::WrongNumberOfMaskChannels { expected, actual } => {
                write!(
                    f,
                    "Wrong number of channels {} in mask, expected {}",
                    actual, expected
                )
            }
            Self::InsufficientInputBufferSize { expected, actual } => {
                write!(
                    f,
                    "Insufficient buffer size {}, expected {} frames",
                    actual, expected
                )
            }
            Self::InsufficientOutputBufferSize { expected, actual } => {
                write!(
                    f,
                    "Insufficient buffer size {}, expected {} frames",
                    actual, expected
                )
            }
            Self::InvalidChunkSize { max, requested } => {
                write!(
                    f,
                    "Invalid chunk size {}, value must be non-zero and cannot exceed {}",
                    requested, max
                )
            }
            Self::ChunkSizeNotAdjustable { .. } => {
                write!(f, "This resampler does not support changing the chunk size")
            }
        }
    }
}

impl fmt::Debug for ResampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self)
    }
}

impl error::Error for ResampleError {}

/// A result alias for the error type used by `rubato`.
pub type ResampleResult<T> = ::std::result::Result<T, ResampleError>;
