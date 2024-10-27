#![doc = include_str!("../README.md")]

#[cfg(feature = "log")]
extern crate log;
use audioadapter::owned::InterleavedOwned;
use audioadapter::{Adapter, AdapterMut};

// Logging wrapper macros to avoid cluttering the code with conditionals.
#[allow(unused)]
macro_rules! trace { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::trace!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! debug { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::debug!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! info { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::info!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! warn { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::warn!($($x)*)
    }
) }
#[allow(unused)]
macro_rules! error { ($($x:tt)*) => (
    #[cfg(feature = "log")] {
        log::error!($($x)*)
    }
) }

mod asynchro;
mod asynchro_fast;
mod asynchro_sinc;
mod error;
mod interpolation;
mod sample;
mod sinc;
#[cfg(feature = "fft_resampler")]
mod synchro;
mod windows;

pub mod sinc_interpolator;

pub use crate::asynchro::{Async, FixedAsync};
pub use crate::asynchro_fast::PolynomialDegree;
pub use crate::asynchro_sinc::{SincInterpolationParameters, SincInterpolationType};
pub use crate::error::{
    CpuFeature, MissingCpuFeature, ResampleError, ResampleResult, ResamplerConstructionError,
};
pub use crate::sample::Sample;
#[cfg(feature = "fft_resampler")]
pub use crate::synchro::{Fft, FixedSync};
pub use crate::windows::{calculate_cutoff, WindowFunction};

/// A struct for providing optional parameters when calling
/// [process_into_buffer](Resampler::process_into_buffer).
#[derive(Debug)]
pub struct Indexing {
    pub input_offset: usize,
    pub output_offset: usize,
    pub partial_len: Option<usize>,
    pub active_channels_mask: Option<Vec<bool>>,
}

pub(crate) fn get_offsets(indexing: &Option<&Indexing>) -> (usize, usize) {
    indexing
        .as_ref()
        .map(|idx| (idx.input_offset, idx.output_offset))
        .unwrap_or((0, 0))
}

pub(crate) fn get_partial_len(indexing: &Option<&Indexing>) -> Option<usize> {
    indexing.as_ref().and_then(|idx| idx.partial_len)
}

/// Helper to update the mask from an optional Indexing struct
pub(crate) fn update_mask(indexing: &Option<&Indexing>, mask: &mut [bool]) {
    if let Some(idx) = indexing {
        if let Some(new_mask) = &idx.active_channels_mask {
            mask.copy_from_slice(new_mask);
            return;
        }
    }
    mask.iter_mut().for_each(|v| *v = true);
}

/// A resampler that is used to resample a chunk of audio to a new sample rate.
/// For asynchronous resamplers, the rate can be adjusted as required.
pub trait Resampler<T>: Send
where
    T: Sample,
{
    /// This is a convenience wrapper for [process_into_buffer](Resampler::process_into_buffer)
    /// that allocates the output buffer with each call. For realtime applications, use
    /// [process_into_buffer](Resampler::process_into_buffer) with a pre-allocated buffer
    /// instead of this function.
    ///
    /// The output is returned as an [InterleavedOwned] struct that wraps a `Vec<T>`
    /// of interleaved samples.
    ///
    /// The `input_offset` and `active_channels_mask` parameters have the same meaning as in
    /// [process_into_buffer](Resampler::process_into_buffer).
    fn process(
        &mut self,
        buffer_in: &dyn Adapter<'_, T>,
        input_offset: usize,
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<InterleavedOwned<T>> {
        let frames = self.output_frames_next();
        let channels = self.nbr_channels();
        let mut buffer_out = InterleavedOwned::<T>::new(T::coerce_from(0.0), channels, frames);

        let indexing = Indexing {
            input_offset,
            output_offset: 0,
            partial_len: None,
            active_channels_mask: active_channels_mask.map(|m| m.to_vec()),
        };
        self.process_into_buffer(buffer_in, &mut buffer_out, Some(&indexing))?;
        Ok(buffer_out)
    }

    /// Resample a buffer of audio to a pre-allocated output buffer.
    /// Use this in real-time applications where the unpredictable time required to allocate
    /// memory from the heap can cause glitches. If this is not a problem, you may use
    /// the [process](Resampler::process) method instead.
    ///
    /// The input and output buffers are buffers from the [audioadapter] crate.
    /// The input buffer must implement the [Adapter] trait,
    /// and the output the corresponding [AdapterMut] trait.
    /// This ensures that this method is able to both read and write
    /// audio data from and to buffers with different layout, as well as different sample formats.
    ///
    /// The `indexing` parameter is optional. When left out, the default values are used.
    ///  - `input_offset` and `output_offset`: these determine how many frames at the beginning
    ///    of the input and output buffers will be skipped before reading or writing starts.
    ///    See the `process_f64` example for how these may be used to process a longer sound clip.
    ///  - `partial_len`: If the input buffer has fewer frames than the required input length,
    ///    set `partial_len` to the available number.
    ///    The resampler will then insert silence in place of the missing frames.
    ///    This is useful for processing a longer buffer with repeated process calls,
    ///    where at the last iteration there may be fewer frames left than what the resampler needs.
    ///  - `active_channels_mask`: A vector of booleans determining what channels are to be processed.
    ///    Any channel marked as inactive by a false value will be skipped during processing
    ///    and the corresponding output will be left unchanged.
    ///    If `None` is given, all channels will be considered active.
    ///
    /// Before processing, the input and output buffer sizes are checked.
    /// If either has the wrong number of channels, or if the buffer can hold too few frames,
    /// a [ResampleError] is returned.
    /// Both input and output are allowed to be longer than required.
    /// The number of input samples consumed and the number output samples written
    /// per channel is returned in a tuple, `(input_frames, output_frames)`.
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)>;

    /// Convenience method for processing audio clips of arbitrary length
    /// from and to buffers in memory.
    /// This method repeatedly calls [process_into_buffer](Resampler::process_into_buffer)
    /// until all frames of the input buffer have been processed.
    /// The processed frames are written to the output buffer,
    /// with the initial silence (caused by the resampler delay) trimmed off.
    ///
    /// Use [process_all_needed_output_len](Resampler::process_all_needed_output_len)
    /// to get the minimal length of the output buffer required
    /// to resample a clip of a given length.
    ///
    /// The `active_channels_mask` parameter has the same meaning as in
    /// [process_into_buffer](Resampler::process_into_buffer).
    fn process_all_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, T>,
        buffer_out: &mut dyn AdapterMut<'a, T>,
        input_len: usize,
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        let expected_output_len = (self.resample_ratio() * input_len as f64).ceil() as usize;

        let mut indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            active_channels_mask: active_channels_mask.map(|m| m.to_vec()),
            partial_len: None,
        };

        let mut frames_left = input_len;
        let mut output_len = 0;
        let mut frames_to_trim = self.output_delay();
        debug!(
            "resamping {} input frames to {} output frames, delay to trim off {} frames",
            input_len, expected_output_len, frames_to_trim
        );

        let next_nbr_input_frames = self.input_frames_next();
        while frames_left > next_nbr_input_frames {
            debug!("process, {} input frames left", frames_left);
            let (nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            frames_left -= nbr_in;
            output_len += nbr_out;
            indexing.input_offset += nbr_in;
            indexing.output_offset += nbr_out;
            if frames_to_trim > 0 && output_len > frames_to_trim {
                debug!(
                    "output, {} is longer  than delay to trim, {}, trimming..",
                    output_len, frames_to_trim
                );
                // move useful output data to start of output buffer
                buffer_out.copy_frames_within(frames_to_trim, 0, frames_to_trim);
                // update counters
                output_len -= frames_to_trim;
                indexing.output_offset -= frames_to_trim;
                frames_to_trim = 0;
            }
        }
        if frames_left > 0 {
            debug!("process the last partial chunk, len {}", frames_left);
            indexing.partial_len = Some(frames_left);
            let (_nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            output_len += nbr_out;
            indexing.output_offset += nbr_out;
        }
        indexing.partial_len = Some(0);
        while output_len < expected_output_len {
            debug!(
                "output is still too short, {} < {}, pump zeros..",
                output_len, expected_output_len
            );
            let (_nbr_in, nbr_out) =
                self.process_into_buffer(buffer_in, buffer_out, Some(&indexing))?;
            output_len += nbr_out;
            indexing.output_offset += nbr_out;
        }
        Ok((input_len, expected_output_len))
    }

    /// Calculate the minimal length of the output buffer
    /// needed to process a clip of length `input_len` using the
    /// [process_all_into_buffer](Resampler::process_all_into_buffer) method.
    fn process_all_needed_output_len(&mut self, input_len: usize) -> usize {
        let delay_frames = self.output_delay();
        let output_frames_next = self.output_frames_next();
        let expected_output_len = (self.resample_ratio() * input_len as f64).ceil() as usize;
        delay_frames + output_frames_next + expected_output_len
    }

    /// Get the maximum possible number of input frames per channel the resampler could require.
    fn input_frames_max(&self) -> usize;

    /// Get the number of frames per channel needed for the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn input_frames_next(&self) -> usize;

    /// Get the number of channels this Resampler is configured for.
    fn nbr_channels(&self) -> usize;

    /// Get the maximum possible number of output frames per channel.
    fn output_frames_max(&self) -> usize;

    /// Get the number of frames per channel that will be output from the next call to
    /// [process_into_buffer](Resampler::process_into_buffer) or [process](Resampler::process).
    fn output_frames_next(&self) -> usize;

    /// Get the delay for the resampler, reported as a number of output frames.
    /// This gives how many frames any event in the input is delayed before it appears in the output.
    fn output_delay(&self) -> usize;

    /// Update the resample ratio.
    ///
    /// For asynchronous resamplers, the ratio must be within
    /// `original / maximum` to `original * maximum`, where the original and maximum are the
    /// resampling ratios that were provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    ///
    /// If the argument `ramp` is set to true, the ratio will be ramped from the old to the new value
    /// during processing of the next chunk. This allows smooth transitions from one ratio to another.
    /// If `ramp` is false, the new ratio will be applied from the start of the next chunk.
    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()>;

    /// Get the current resample ratio, defined as output sample rate divided by input sample rate.
    fn resample_ratio(&self) -> f64;

    /// Update the resample ratio as a factor relative to the original one.
    ///
    /// For asynchronous resamplers, the relative ratio must be within
    /// `1 / maximum` to `maximum`, where `maximum` is the maximum
    /// resampling ratio that was provided to the constructor. Trying to set the ratio
    /// outside these bounds will return [ResampleError::RatioOutOfBounds].
    ///
    /// Ratios above 1.0 slow down the output and lower the pitch, while ratios
    /// below 1.0 speed up the output and raise the pitch.
    ///
    /// For synchronous resamplers, this will always return [ResampleError::SyncNotAdjustable].
    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()>;

    /// Reset the resampler state and clear all internal buffers.
    fn reset(&mut self);

    /// Change the chunk size for the resampler.
    /// This is not supported by all resampler types.
    /// The value must be equal to or smaller than the chunk size the value
    /// that the resampler was created with.
    /// [ResampleError::InvalidChunkSize] is returned if the value is zero or too large.
    ///
    /// The meaning of chunk size depends on the resampler,
    /// it refers to the input size for resamplers with fixed input size,
    /// and output size for resamplers with fixed output size.
    ///
    /// Resamplers that do not support changing the chunk size
    /// return [ResampleError::ChunkSizeNotAdjustable].
    fn set_chunk_size(&mut self, _chunksize: usize) -> ResampleResult<()> {
        Err(ResampleError::ChunkSizeNotAdjustable)
    }
}

pub(crate) fn validate_buffers<'a, T: 'a>(
    wave_in: &dyn Adapter<'a, T>,
    wave_out: &dyn AdapterMut<'a, T>,
    mask: &[bool],
    channels: usize,
    min_input_len: usize,
    min_output_len: usize,
) -> ResampleResult<()> {
    if wave_in.channels() != channels {
        return Err(ResampleError::WrongNumberOfInputChannels {
            expected: channels,
            actual: wave_in.channels(),
        });
    }
    if mask.len() != channels {
        return Err(ResampleError::WrongNumberOfMaskChannels {
            expected: channels,
            actual: mask.len(),
        });
    }
    if wave_in.frames() < min_input_len {
        return Err(ResampleError::InsufficientInputBufferSize {
            expected: min_input_len,
            actual: wave_in.frames(),
        });
    }
    if wave_out.channels() != channels {
        return Err(ResampleError::WrongNumberOfOutputChannels {
            expected: channels,
            actual: wave_out.channels(),
        });
    }
    if wave_out.frames() < min_output_len {
        return Err(ResampleError::InsufficientOutputBufferSize {
            expected: min_output_len,
            actual: wave_out.frames(),
        });
    }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    #[cfg(feature = "fft_resampler")]
    use crate::Fft;
    use crate::Resampler;
    use crate::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    use audioadapter::direct::SequentialSliceOfVecs;
    use audioadapter::Adapter;

    #[test_log::test]
    fn process_all() {
        let mut resampler = Async::<f64>::new_sinc(
            88200 as f64 / 44100 as f64,
            1.1,
            SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 16,
                window: WindowFunction::BlackmanHarris2,
            },
            1024,
            2,
            FixedAsync::Input,
        )
        .unwrap();
        let input_len = 12345;
        let samples: Vec<f64> = (0..input_len).map(|v| v as f64 / 10.0).collect();
        let input_data = vec![samples; 2];
        // add a ramp to the input
        let input = SequentialSliceOfVecs::new(&input_data, 2, input_len).unwrap();
        let output_len = resampler.process_all_needed_output_len(input_len);
        let mut output_data = vec![vec![0.0f64; output_len]; 2];
        let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 2, output_len).unwrap();
        let (nbr_in, nbr_out) = resampler
            .process_all_into_buffer(&input, &mut output, input_len, None)
            .unwrap();
        assert_eq!(nbr_in, input_len);
        // This is a simple ratio, output should be twice as long as input
        assert_eq!(2 * nbr_in, nbr_out);

        // check that the output follows the input ramp, within suitable margins
        let increment = 0.1 / resampler.resample_ratio();
        let delay = resampler.output_delay();
        let margin = (delay as f64 * resampler.resample_ratio()) as usize;
        let mut expected = margin as f64 * increment;
        for frame in margin..(nbr_out - margin) {
            for chan in 0..2 {
                let val = output.read_sample(chan, frame).unwrap();
                assert!(
                    val - expected < 100.0 * increment,
                    "frame: {}, value: {}, expected: {}",
                    frame,
                    val,
                    expected
                );
                assert!(
                    expected - val < 100.0 * increment,
                    "frame: {}, value: {}, expected: {}",
                    frame,
                    val,
                    expected
                );
            }
            expected += increment;
        }
    }

    // This tests that a Resampler can be boxed.
    #[test_log::test]
    fn boxed_resampler() {
        let mut boxed: Box<dyn Resampler<f64>> = Box::new(
            Async::<f64>::new_sinc(
                88200 as f64 / 44100 as f64,
                1.1,
                SincInterpolationParameters {
                    sinc_len: 64,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Cubic,
                    oversampling_factor: 16,
                    window: WindowFunction::BlackmanHarris2,
                },
                1024,
                2,
                FixedAsync::Input,
            )
            .unwrap(),
        );
        let max_frames_out = boxed.output_frames_max();
        let nbr_frames_in_next = boxed.input_frames_next();
        let waves = vec![vec![0.0f64; nbr_frames_in_next]; 2];
        let mut waves_out = vec![vec![0.0f64; max_frames_out]; 2];
        let input = SequentialSliceOfVecs::new(&waves, 2, nbr_frames_in_next).unwrap();
        let mut output = SequentialSliceOfVecs::new_mut(&mut waves_out, 2, max_frames_out).unwrap();
        let _ = process_with_boxed(&mut boxed, &input, &mut output);
    }

    fn process_with_boxed<'a>(
        resampler: &mut Box<dyn Resampler<f64>>,
        input: &SequentialSliceOfVecs<&'a [Vec<f64>]>,
        output: &mut SequentialSliceOfVecs<&'a mut [Vec<f64>]>,
    ) {
        resampler.process_into_buffer(input, output, None).unwrap();
    }

    fn impl_send<T: Send>() {
        fn is_send<T: Send>() {}
        is_send::<Async<T>>();
        #[cfg(feature = "fft_resampler")]
        {
            is_send::<Fft<T>>();
        }
    }

    // This tests that all resamplers are Send.
    #[test]
    fn test_impl_send() {
        impl_send::<f32>();
        impl_send::<f64>();
    }

    pub fn expected_output_value(idx: usize, delay: usize, ratio: f64) -> f64 {
        if idx <= delay {
            return 0.0;
        }
        (idx - delay) as f64 * 0.1 / ratio
    }

    #[macro_export]
    macro_rules! check_output {
        ($resampler:ident, $fty:ty) => {
            let mut ramp_value: $fty = 0.0;
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            let ratio = $resampler.resample_ratio() as $fty;
            let delay = $resampler.output_delay();
            let mut output_index = 0;

            let out_incr = 0.1 / ratio;

            let nbr_iterations =
                100000 / ($resampler.output_frames_next() + $resampler.input_frames_next());
            for _n in 0..nbr_iterations {
                let expected_frames_in = $resampler.input_frames_next();
                let expected_frames_out = $resampler.output_frames_next();
                // Check that lengths are within the reported max values
                assert!(expected_frames_in <= max_input_len);
                assert!(expected_frames_out <= max_output_len);
                let mut input_data = vec![vec![0.0 as $fty; expected_frames_in]; 2];
                for m in 0..expected_frames_in {
                    for ch in 0..2 {
                        input_data[ch][m] = ramp_value;
                    }
                    ramp_value = ramp_value + 0.1;
                }
                let input = SequentialSliceOfVecs::new(&input_data, 2, expected_frames_in).unwrap();
                let mut output_data = vec![vec![0.0 as $fty; expected_frames_out]; 2];
                let mut output =
                    SequentialSliceOfVecs::new_mut(&mut output_data, 2, expected_frames_out)
                        .unwrap();

                trace!("resample...");
                let (input_frames, output_frames) = $resampler
                    .process_into_buffer(&input, &mut output, None)
                    .unwrap();
                trace!("assert lengths");
                assert_eq!(input_frames, expected_frames_in);
                assert_eq!(output_frames, expected_frames_out);
                trace!("check output");
                for idx in 0..output_frames {
                    let expected = expected_output_value(output_index + idx, delay, ratio) as $fty;
                    for ch in 0..2 {
                        let value = output_data[ch][idx];
                        let margin = 3.0 * out_incr;
                        assert!(
                            value > expected - margin,
                            "Value at frame {} is too small, {} < {} - {}",
                            output_index + idx,
                            value,
                            expected,
                            margin
                        );
                        assert!(
                            value < expected + margin,
                            "Value at frame {} is too large, {} > {} + {}",
                            output_index + idx,
                            value,
                            expected,
                            margin
                        );
                    }
                }
                output_index += output_frames;
            }
            assert!(output_index > 1000, "Too few frames checked!");
        };
    }

    #[macro_export]
    macro_rules! check_ratio {
        ($resampler:ident, $repetitions:expr, $margin:expr, $fty:ty) => {
            let ratio = $resampler.resample_ratio();
            let max_input_len = $resampler.input_frames_max();
            let max_output_len = $resampler.output_frames_max();
            let waves_in = vec![vec![0.0 as $fty; max_input_len]; 2];
            let input = SequentialSliceOfVecs::new(&waves_in, 2, max_input_len).unwrap();
            let mut waves_out = vec![vec![0.0 as $fty; max_output_len]; 2];
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut waves_out, 2, max_output_len).unwrap();
            let mut total_in = 0;
            let mut total_out = 0;
            for _ in 0..$repetitions {
                let out = $resampler
                    .process_into_buffer(&input, &mut output, None)
                    .unwrap();
                total_in += out.0;
                total_out += out.1
            }
            let measured_ratio = total_out as f64 / total_in as f64;
            assert!(
                measured_ratio / ratio > (1.0 - $margin),
                "Measured ratio is too small, measured / expected = {}",
                measured_ratio / ratio
            );
            assert!(
                measured_ratio / ratio < (1.0 + $margin),
                "Measured ratio is too large, measured / expected = {}",
                measured_ratio / ratio
            );
        };
    }

    #[macro_export]
    macro_rules! assert_fi_len {
        ($resampler:ident, $chunksize:expr) => {
            let nbr_frames_in_next = $resampler.input_frames_next();
            let nbr_frames_in_max = $resampler.input_frames_max();
            assert_eq!(
                nbr_frames_in_next, $chunksize,
                "expected {} for next input samples, got {}",
                $chunksize, nbr_frames_in_next
            );
            assert_eq!(
                nbr_frames_in_next, $chunksize,
                "expected {} for max input samples, got {}",
                $chunksize, nbr_frames_in_max
            );
        };
    }

    #[macro_export]
    macro_rules! assert_fo_len {
        ($resampler:ident, $chunksize:expr) => {
            let nbr_frames_out_next = $resampler.output_frames_next();
            let nbr_frames_out_max = $resampler.output_frames_max();
            assert_eq!(
                nbr_frames_out_next, $chunksize,
                "expected {} for next output samples, got {}",
                $chunksize, nbr_frames_out_next
            );
            assert_eq!(
                nbr_frames_out_next, $chunksize,
                "expected {} for max output samples, got {}",
                $chunksize, nbr_frames_out_max
            );
        };
    }

    #[macro_export]
    macro_rules! assert_fb_len {
        ($resampler:ident) => {
            let nbr_frames_out_next = $resampler.output_frames_next();
            let nbr_frames_out_max = $resampler.output_frames_max();
            let nbr_frames_in_next = $resampler.input_frames_next();
            let nbr_frames_in_max = $resampler.input_frames_max();
            let ratio = $resampler.resample_ratio();
            assert_eq!(
                nbr_frames_out_next, nbr_frames_out_max,
                "next output frames, {}, is different than max, {}",
                nbr_frames_out_next, nbr_frames_out_next
            );
            assert_eq!(
                nbr_frames_in_next, nbr_frames_in_max,
                "next input frames, {}, is different than max, {}",
                nbr_frames_in_next, nbr_frames_in_max
            );
            let frames_ratio = nbr_frames_out_next as f64 / nbr_frames_in_next as f64;
            assert_abs_diff_eq!(frames_ratio, ratio, epsilon = 0.000001);
        };
    }

    #[macro_export]
    macro_rules! check_reset {
        ($resampler:ident) => {
            let frames_in = $resampler.input_frames_next();

            let mut rng = rand::thread_rng();
            let mut input_data = vec![vec![0.0f64; frames_in]; 2];
            input_data
                .iter_mut()
                .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));

            let input = SequentialSliceOfVecs::new(&input_data, 2, frames_in).unwrap();

            let frames_out = $resampler.output_frames_next();
            let mut output_data_1 = vec![vec![0.0; frames_out]; 2];
            let mut output_1 =
                SequentialSliceOfVecs::new_mut(&mut output_data_1, 2, frames_out).unwrap();
            $resampler
                .process_into_buffer(&input, &mut output_1, None)
                .unwrap();
            $resampler.reset();
            assert_eq!(
                frames_in,
                $resampler.input_frames_next(),
                "Resampler requires different number of frames when new and after a reset."
            );
            let mut output_data_2 = vec![vec![0.0; frames_out]; 2];
            let mut output_2 =
                SequentialSliceOfVecs::new_mut(&mut output_data_2, 2, frames_out).unwrap();
            $resampler
                .process_into_buffer(&input, &mut output_2, None)
                .unwrap();
            assert_eq!(
                output_data_1, output_data_2,
                "Resampler gives different output when new and after a reset."
            );
        };
    }

    #[macro_export]
    macro_rules! check_input_offset {
        ($resampler:ident) => {
            let frames_in = $resampler.input_frames_next();

            let mut rng = rand::thread_rng();
            let mut input_data_1 = vec![vec![0.0f64; frames_in]; 2];
            input_data_1
                .iter_mut()
                .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));

            let offset = 123;
            let mut input_data_2 = vec![vec![0.0f64; frames_in + offset]; 2];
            for (ch, data) in input_data_2.iter_mut().enumerate() {
                data[offset..offset + frames_in].clone_from_slice(&input_data_1[ch][..])
            }

            let input_1 = SequentialSliceOfVecs::new(&input_data_1, 2, frames_in).unwrap();
            let input_2 = SequentialSliceOfVecs::new(&input_data_2, 2, frames_in + offset).unwrap();

            let frames_out = $resampler.output_frames_next();
            let mut output_data_1 = vec![vec![0.0; frames_out]; 2];
            let mut output_1 =
                SequentialSliceOfVecs::new_mut(&mut output_data_1, 2, frames_out).unwrap();
            $resampler
                .process_into_buffer(&input_1, &mut output_1, None)
                .unwrap();
            $resampler.reset();
            assert_eq!(
                frames_in,
                $resampler.input_frames_next(),
                "Resampler requires different number of frames when new and after a reset."
            );
            let mut output_data_2 = vec![vec![0.0; frames_out]; 2];
            let mut output_2 =
                SequentialSliceOfVecs::new_mut(&mut output_data_2, 2, frames_out).unwrap();

            let indexing = Indexing {
                input_offset: offset,
                output_offset: 0,
                active_channels_mask: None,
                partial_len: None,
            };
            $resampler
                .process_into_buffer(&input_2, &mut output_2, Some(&indexing))
                .unwrap();
            assert_eq!(
                output_data_1, output_data_2,
                "Resampler gives different output when new and after a reset."
            );
        };
    }

    #[macro_export]
    macro_rules! check_output_offset {
        ($resampler:ident) => {
            let frames_in = $resampler.input_frames_next();

            let mut rng = rand::thread_rng();
            let mut input_data = vec![vec![0.0f64; frames_in]; 2];
            input_data
                .iter_mut()
                .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));

            let input = SequentialSliceOfVecs::new(&input_data, 2, frames_in).unwrap();

            let frames_out = $resampler.output_frames_next();
            let mut output_data_1 = vec![vec![0.0; frames_out]; 2];
            let mut output_1 =
                SequentialSliceOfVecs::new_mut(&mut output_data_1, 2, frames_out).unwrap();
            $resampler
                .process_into_buffer(&input, &mut output_1, None)
                .unwrap();
            $resampler.reset();
            assert_eq!(
                frames_in,
                $resampler.input_frames_next(),
                "Resampler requires different number of frames when new and after a reset."
            );
            let offset = 123;
            let mut output_data_2 = vec![vec![0.0; frames_out + offset]; 2];
            let mut output_2 =
                SequentialSliceOfVecs::new_mut(&mut output_data_2, 2, frames_out + offset).unwrap();
            let indexing = Indexing {
                input_offset: 0,
                output_offset: offset,
                active_channels_mask: None,
                partial_len: None,
            };
            $resampler
                .process_into_buffer(&input, &mut output_2, Some(&indexing))
                .unwrap();
            assert_eq!(
                output_data_1[0][..],
                output_data_2[0][offset..],
                "Resampler gives different output when new and after a reset."
            );
            assert_eq!(
                output_data_1[1][..],
                output_data_2[1][offset..],
                "Resampler gives different output when new and after a reset."
            );
        };
    }

    #[macro_export]
    macro_rules! check_masked {
        ($resampler:ident) => {
            let frames_in = $resampler.input_frames_next();

            let mut rng = rand::thread_rng();
            let mut input_data = vec![vec![0.0f64; frames_in]; 2];
            input_data
                .iter_mut()
                .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));

            let input = SequentialSliceOfVecs::new(&input_data, 2, frames_in).unwrap();

            let frames_out = $resampler.output_frames_next();
            let mut output_data = vec![vec![0.0; frames_out]; 2];
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut output_data, 2, frames_out).unwrap();

            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                active_channels_mask: Some(vec![false, true]),
                partial_len: None,
            };
            $resampler
                .process_into_buffer(&input, &mut output, Some(&indexing))
                .unwrap();

            let non_zero_chan_0 = output_data[0].iter().filter(|&v| *v != 0.0).count();
            let non_zero_chan_1 = output_data[1].iter().filter(|&v| *v != 0.0).count();
            // assert channel 0 is all zero
            assert_eq!(
                non_zero_chan_0, 0,
                "Some sample in the non-active channel has a non-zero value"
            );
            // assert channel 1 has some values
            assert!(
                non_zero_chan_1 > 0,
                "No sample in the active channel has a non-zero value"
            );
        };
    }

    #[macro_export]
    macro_rules! check_resize {
        ($resampler:ident) => {};
    }
}
