use crate::error::{ResampleError, ResampleResult, ResamplerConstructionError};
use crate::interpolation::*;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "aarch64")]
use crate::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;
use crate::sinc_interpolator::{ScalarInterpolator, SincInterpolator};
use crate::windows::WindowFunction;
use crate::{update_mask_from_buffers, validate_buffers, Fixed, Resampler, Sample};
use std::fmt;

/// A struct holding the parameters for sinc interpolation.
#[derive(Debug)]
pub struct SincInterpolationParameters {
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
    /// Only the points actually needed are calculated during processing
    /// so a larger number does not directly lead to higher cpu usage.
    /// A lower value helps in keeping the sincs in the cpu cache. Start at 128.
    pub oversampling_factor: usize,
    /// Interpolation type, see `SincInterpolationType`
    pub interpolation: SincInterpolationType,
    /// Window function to use.
    pub window: WindowFunction,
}

/// Interpolation methods that can be selected. For asynchronous interpolation where the
/// ratio between input and output sample rates can be any number, it's not possible to
/// pre-calculate all the needed interpolation filters.
/// Instead they have to be computed as needed, which becomes impractical since the
/// sincs are very expensive to generate in terms of cpu time.
/// It's more efficient to combine the sinc filters with some other interpolation technique.
/// Then, sinc filters are used to provide a fixed number of interpolated points between input samples,
/// and then, the new value is calculated by interpolation between those points.
#[derive(Debug)]
pub enum SincInterpolationType {
    /// For cubic interpolation, the four nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then, a cubic polynomial is fitted to these points, and is used to calculate the new sample value.
    /// The computation time is approximately twice as long as that of linear interpolation,
    /// but it requires much fewer intermediate points for a good result.
    Cubic,
    /// For quadratic interpolation, the three nearest intermediate points are calculated
    /// using sinc interpolation.
    /// Then, a quadratic polynomial is fitted to these points, and is used to calculate the new sample value.
    /// The computation time lies approximately halfway between that of linear and quadratic interpolation.
    Quadratic,
    /// For linear interpolation, the new sample value is calculated by linear interpolation
    /// between the two nearest points.
    /// This requires two intermediate points to be calculated using sinc interpolation,
    /// and the output is obtained by taking a weighted average of these two points.
    /// This is relatively fast, but needs a large number of intermediate points to
    /// push the resampling artefacts below the noise floor.
    Linear,
    /// The Nearest mode doesn't do any interpolation, but simply picks the nearest intermediate point.
    /// This is useful when the nearest point is actually the correct one, for example when upsampling by a factor 2,
    /// like 48kHz->96kHz.
    /// Then, when setting the oversampling_factor to 2 and using Nearest mode,
    /// no unnecessary computations are performed and the result is equivalent to that of synchronous resampling.
    /// This also works for other ratios that can be expressed by a fraction. For 44.1kHz -> 48 kHz,
    /// setting oversampling_factor to 160 gives the desired result (since 48kHz = 160/147 * 44.1kHz).
    Nearest,
}

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
/// The number of input frames is determined by the chunk size argument to the constructor.
/// This value can be changed by the `set_chunk_size()` method,
/// to let the resampler process smaller chunks of audio data.
/// Note that the chunk size cannot exceed the value given at creation time.
/// The maximum value can be retrieved using the `input_size_max()` method,
/// and `input_frames_next()` gives the current value.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
///
/// The resampling ratio can be freely adjusted within the range specified to the constructor.
/// Adjusting the ratio does not recalculate the sinc functions used by the anti-aliasing filter.
/// This causes no issue when increasing the ratio (which slows down the output).
/// However, when decreasing more than a few percent (or speeding up the output),
/// the filters can no longer suppress all aliasing and this may lead to some artefacts.
/// Higher maximum ratios require more memory to be allocated by [Resampler::output_buffer_allocate].
pub struct Sinc<T> {
    nbr_channels: usize,
    chunk_size: usize,
    max_chunk_size: usize,
    needed_input_size: usize,
    needed_output_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    target_ratio: f64,
    max_relative_ratio: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: SincInterpolationType,
    channel_mask: Vec<bool>,
    fixed: Fixed,
}

impl<T> fmt::Debug for Sinc<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Sinc")
            .field("nbr_channels", &self.nbr_channels)
            .field("chunk_size,", &self.chunk_size)
            .field("max_chunk_size,", &self.max_chunk_size)
            .field("needed_input_size,", &self.needed_input_size)
            .field("needed_output_size,", &self.needed_output_size)
            .field("last_index", &self.last_index)
            .field("current_buffer_fill", &self.current_buffer_fill)
            .field("resample_ratio", &self.resample_ratio)
            .field("resample_ratio_original", &self.resample_ratio_original)
            .field("target_ratio", &self.target_ratio)
            .field("max_relative_ratio", &self.max_relative_ratio)
            .field("interpolator.len()", &self.interpolator.len())
            .field("buffer[0].len()", &self.buffer[0].len())
            .field("interpolation", &self.interpolation)
            .field("channel_mask", &self.channel_mask)
            .field("fixed", &self.fixed)
            .finish()
    }
}

pub fn make_interpolator<T>(
    sinc_len: usize,
    resample_ratio: f64,
    f_cutoff: f32,
    oversampling_factor: usize,
    window: WindowFunction,
) -> Box<dyn SincInterpolator<T>>
where
    T: Sample,
{
    let sinc_len = 8 * (((sinc_len as f32) / 8.0).ceil() as usize);
    let f_cutoff = if resample_ratio >= 1.0 {
        f_cutoff
    } else {
        f_cutoff * resample_ratio as f32
    };

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        AvxInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(target_arch = "x86_64")]
    if let Ok(interpolator) =
        SseInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    #[cfg(target_arch = "aarch64")]
    if let Ok(interpolator) =
        NeonInterpolator::<T>::new(sinc_len, oversampling_factor, f_cutoff, window)
    {
        return Box::new(interpolator);
    }

    Box::new(ScalarInterpolator::<T>::new(
        sinc_len,
        oversampling_factor,
        f_cutoff,
        window,
    ))
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2.
fn interp_cubic<T>(x: T, yvals: &[T; 4]) -> T
where
    T: Sample,
{
    let a0 = yvals[1];
    let a1 = -(T::one() / T::coerce(3.0)) * yvals[0] - T::coerce(0.5) * yvals[1] + yvals[2]
        - (T::one() / T::coerce(6.0)) * yvals[3];
    let a2 = T::coerce(0.5) * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = T::coerce(0.5) * (yvals[1] - yvals[2])
        + (T::one() / T::coerce(6.0)) * (yvals[3] - yvals[0]);
    let x2 = x * x;
    let x3 = x2 * x;
    a0 + a1 * x + a2 * x2 + a3 * x3
}

/// Perform quadratic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = 0, 1, 2.
fn interp_quad<T>(x: T, yvals: &[T; 3]) -> T
where
    T: Sample,
{
    let a2 = yvals[0] - T::coerce(2.0) * yvals[1] + yvals[2];
    let a1 = -T::coerce(3.0) * yvals[0] + T::coerce(4.0) * yvals[1] - yvals[2];
    let a0 = T::coerce(2.0) * yvals[0];
    let x2 = x * x;
    T::coerce(0.5) * (a0 + a1 * x + a2 * x2)
}

/// Perform linear interpolation between two points at x=0 and x=1.
fn interp_lin<T>(x: T, yvals: &[T; 2]) -> T
where
    T: Sample,
{
    yvals[0] + x * (yvals[1] - yvals[0])
}

fn validate_ratios(
    resample_ratio: f64,
    max_resample_ratio_relative: f64,
) -> Result<(), ResamplerConstructionError> {
    if resample_ratio <= 0.0 {
        return Err(ResamplerConstructionError::InvalidRatio(resample_ratio));
    }
    if max_resample_ratio_relative < 1.0 {
        return Err(ResamplerConstructionError::InvalidRelativeRatio(
            max_resample_ratio_relative,
        ));
    }
    Ok(())
}

impl<T> Sinc<T>
where
    T: Sample,
{
    /// Create a new Sinc.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio * 10.0` and `resample_ratio / 10.0`.
    /// - `parameters`: Parameters for interpolation, see `SincInterpolationParameters`.
    /// - `chunk_size`: Size of input data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        parameters: SincInterpolationParameters,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: Fixed,
    ) -> Result<Self, ResamplerConstructionError> {
        debug!(
            "Create new Sinc fixed {:?}, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
            fixed, resample_ratio, chunk_size, nbr_channels, parameters
        );

        let interpolator = make_interpolator(
            parameters.sinc_len,
            resample_ratio,
            parameters.f_cutoff,
            parameters.oversampling_factor,
            parameters.window,
        );

        Self::new_with_interpolator(
            resample_ratio,
            max_resample_ratio_relative,
            parameters.interpolation,
            interpolator,
            chunk_size,
            nbr_channels,
            fixed,
        )
    }

    /// Create a new Sinc using an existing Interpolator.
    ///
    /// Parameters are:
    /// - `resample_ratio`: Starting ratio between output and input sample rates, must be > 0.
    /// - `max_resample_ratio_relative`: Maximum ratio that can be set with [Resampler::set_resample_ratio] relative to `resample_ratio`, must be >= 1.0. The minimum relative ratio is the reciprocal of the maximum. For example, with `max_resample_ratio_relative` of 10.0, the ratio can be set between `resample_ratio` * 10.0 and `resample_ratio` / 10.0.
    /// - `interpolation_type`: Parameters for interpolation, see `SincInterpolationParameters`.
    /// - `interpolator`: The interpolator to use.
    /// - `chunk_size`: Size of output data in frames.
    /// - `nbr_channels`: Number of channels in input/output.
    pub fn new_with_interpolator(
        resample_ratio: f64,
        max_resample_ratio_relative: f64,
        interpolation_type: SincInterpolationType,
        interpolator: Box<dyn SincInterpolator<T>>,
        chunk_size: usize,
        nbr_channels: usize,
        fixed: Fixed,
    ) -> Result<Self, ResamplerConstructionError> {
        validate_ratios(resample_ratio, max_resample_ratio_relative)?;

        let interpolator_len = interpolator.len();

        let last_index = -(interpolator_len as f64) / 2.0;
        let needed_input_size = Self::calculate_input_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );
        let needed_output_size = Self::calculate_output_size(
            chunk_size,
            resample_ratio,
            resample_ratio,
            last_index,
            interpolator_len,
            &fixed,
        );

        let buffer_len = Self::calculate_max_input_size(
            chunk_size,
            resample_ratio,
            max_resample_ratio_relative,
            interpolator_len,
            &fixed,
        ) + 2 * interpolator_len;

        let buffer = vec![vec![T::zero(); buffer_len]; nbr_channels];

        let channel_mask = vec![true; nbr_channels];

        Ok(Sinc {
            nbr_channels,
            chunk_size,
            max_chunk_size: chunk_size,
            needed_input_size,
            needed_output_size,
            last_index: -((interpolator.len() / 2) as f64),
            current_buffer_fill: needed_input_size,
            resample_ratio,
            resample_ratio_original: resample_ratio,
            target_ratio: resample_ratio,
            max_relative_ratio: max_resample_ratio_relative,
            interpolator,
            buffer,
            interpolation: interpolation_type,
            channel_mask,
            fixed,
        })
    }

    fn calculate_input_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Input => chunk_size,
            Fixed::Output => (last_index
                + chunk_size as f64 / (0.5 * resample_ratio + 0.5 * target_ratio)
                + interpolator_len as f64)
                .ceil() as usize,
        }
    }

    fn calculate_output_size(
        chunk_size: usize,
        resample_ratio: f64,
        target_ratio: f64,
        last_index: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Output => chunk_size,
            Fixed::Input => ((chunk_size as f64 - (interpolator_len + 1) as f64 - last_index)
                * (0.5 * resample_ratio + 0.5 * target_ratio))
                .floor() as usize,
        }
    }

    fn calculate_max_input_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        interpolator_len: usize,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Input => chunk_size,
            Fixed::Output => {
                (chunk_size as f64 / resample_ratio_original * max_relative_ratio).ceil() as usize
                    + 2
                    + interpolator_len / 2
            }
        }
    }

    fn calculate_max_output_size(
        chunk_size: usize,
        resample_ratio_original: f64,
        max_relative_ratio: f64,
        fixed: &Fixed,
    ) -> usize {
        match fixed {
            Fixed::Output => chunk_size,
            Fixed::Input => {
                (chunk_size as f64 * resample_ratio_original * max_relative_ratio + 10.0) as usize
            }
        }
    }

    fn update_lengths(&mut self) {
        self.needed_input_size = Self::calculate_input_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.interpolator.len(),
            &self.fixed,
        );
        self.needed_output_size = Self::calculate_output_size(
            self.chunk_size,
            self.resample_ratio,
            self.target_ratio,
            self.last_index,
            self.interpolator.len(),
            &self.fixed,
        );
        trace!(
            "Updated lengths, input: {}, output: {}",
            self.needed_input_size,
            self.needed_output_size
        );
    }
}

impl<T> Resampler<T> for Sinc<T>
where
    T: Sample,
{
    fn process_into_buffer<Vin: AsRef<[T]>, Vout: AsMut<[T]>>(
        &mut self,
        wave_in: &[Vin],
        wave_out: &mut [Vout],
        active_channels_mask: Option<&[bool]>,
    ) -> ResampleResult<(usize, usize)> {
        if let Some(mask) = active_channels_mask {
            self.channel_mask.copy_from_slice(mask);
        } else {
            update_mask_from_buffers(&mut self.channel_mask);
        };
        trace!("Start processing, {:?}", self);

        validate_buffers(
            wave_in,
            wave_out,
            &self.channel_mask,
            self.nbr_channels,
            self.needed_input_size,
            self.needed_output_size,
        )?;

        let sinc_len = self.interpolator.len();
        let oversampling_factor = self.interpolator.nbr_sincs();
        let mut t_ratio = 1.0 / self.resample_ratio;
        let t_ratio_end = 1.0 / self.target_ratio;

        let t_ratio_increment = (t_ratio_end - t_ratio) / self.needed_output_size as f64;

        // Update buffer with new data.
        for buf in self.buffer.iter_mut() {
            buf.copy_within(
                self.needed_input_size..self.needed_input_size + 2 * sinc_len,
                0,
            );
        }

        for (chan, active) in self.channel_mask.iter().enumerate() {
            if *active {
                debug_assert!(self.needed_output_size <= wave_out[chan].as_mut().len());
                self.buffer[chan][2 * sinc_len..2 * sinc_len + self.needed_input_size]
                    .copy_from_slice(&wave_in[chan].as_ref()[..self.needed_input_size]);
            }
        }

        let mut idx = self.last_index;

        match self.interpolation {
            SincInterpolationType::Cubic => {
                let mut points = [T::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];
                for n in 0..self.needed_output_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan].as_mut()[n] = interp_cubic(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Quadratic => {
                let mut points = [T::zero(); 3];
                let mut nearest = [(0isize, 0isize); 3];
                for n in 0..self.needed_output_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_3(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan].as_mut()[n] = interp_quad(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Linear => {
                let mut points = [T::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];
                for n in 0..self.needed_output_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = T::coerce(frac);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * sinc_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out[chan].as_mut()[n] = interp_lin(frac_offset, &points);
                        }
                    }
                }
            }
            SincInterpolationType::Nearest => {
                let mut point;
                let mut nearest;
                for n in 0..self.needed_output_size {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    nearest = get_nearest_time(idx, oversampling_factor as isize);
                    for (chan, active) in self.channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &self.buffer[chan];
                            point = self.interpolator.get_sinc_interpolated(
                                buf,
                                (nearest.0 + 2 * sinc_len as isize) as usize,
                                nearest.1 as usize,
                            );
                            wave_out[chan].as_mut()[n] = point;
                        }
                    }
                }
            }
        }

        // Store last index for next iteration.
        self.last_index = idx - self.needed_input_size as f64;
        self.resample_ratio = self.target_ratio;
        trace!(
            "Resampling channels {:?}, {} frames in, {} frames out",
            active_channels_mask,
            self.needed_input_size,
            self.needed_output_size,
        );
        let input_size = self.needed_input_size;
        let output_size = self.needed_output_size;
        self.update_lengths();
        Ok((input_size, output_size))
    }

    fn output_frames_max(&self) -> usize {
        Sinc::<T>::calculate_max_output_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            &self.fixed,
        )
    }

    fn output_frames_next(&self) -> usize {
        self.needed_output_size
    }

    fn output_delay(&self) -> usize {
        (self.interpolator.len() as f64 * self.resample_ratio / 2.0) as usize
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn input_frames_max(&self) -> usize {
        Sinc::<T>::calculate_max_input_size(
            self.max_chunk_size,
            self.resample_ratio_original,
            self.max_relative_ratio,
            self.interpolator.len(),
            &self.fixed,
        )
    }

    fn input_frames_next(&self) -> usize {
        self.needed_input_size
    }

    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()> {
        trace!("Change resample ratio to {}", new_ratio);
        if (new_ratio / self.resample_ratio_original >= 1.0 / self.max_relative_ratio)
            && (new_ratio / self.resample_ratio_original <= self.max_relative_ratio)
        {
            if !ramp {
                self.resample_ratio = new_ratio;
            }
            self.target_ratio = new_ratio;
            self.update_lengths();
            Ok(())
        } else {
            Err(ResampleError::RatioOutOfBounds {
                provided: new_ratio,
                original: self.resample_ratio_original,
                max_relative_ratio: self.max_relative_ratio,
            })
        }
    }

    fn set_resample_ratio_relative(&mut self, rel_ratio: f64, ramp: bool) -> ResampleResult<()> {
        let new_ratio = self.resample_ratio_original * rel_ratio;
        self.set_resample_ratio(new_ratio, ramp)
    }

    fn reset(&mut self) {
        self.buffer
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = T::zero()));
        self.channel_mask.iter_mut().for_each(|val| *val = true);
        self.last_index = -((self.interpolator.len() / 2) as f64);
        self.resample_ratio = self.resample_ratio_original;
        self.target_ratio = self.resample_ratio_original;
        self.chunk_size = self.max_chunk_size;
        self.update_lengths();
    }

    fn set_chunk_size(&mut self, chunksize: usize) -> ResampleResult<()> {
        if chunksize > self.max_chunk_size || chunksize == 0 {
            return Err(ResampleError::InvalidChunkSize {
                max: self.max_chunk_size,
                requested: chunksize,
            });
        }
        self.chunk_size = chunksize;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{interp_cubic, interp_lin};
    use crate::Resampler;
    use crate::SincInterpolationParameters;
    use crate::SincInterpolationType;
    use crate::WindowFunction;
    use crate::{check_output, check_ratio};
    use crate::{Fixed, Sinc};
    use rand::Rng;
    use test_log::test;

    fn basic_params() -> SincInterpolationParameters {
        SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        }
    }

    #[test]
    fn int_cubic() {
        let params = basic_params();
        let _resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let params = basic_params();
        let _resampler = Sinc::<f32>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let params = basic_params();
        let _resampler = Sinc::<f32>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let params = basic_params();
        let _resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let yvals = [1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn make_resampler_fi() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn reset_resampler_fi() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();

        let mut rng = rand::thread_rng();
        let mut waves = vec![vec![0.0f64; 1024]; 2];
        waves
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));
        let out1 = resampler.process(&waves, None).unwrap();
        resampler.reset();
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(
            out1, out2,
            "Resampler gives different output when new and after a reset."
        );
    }

    #[test]
    fn make_resampler_fi_32() {
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 1150 && out[0].len() < 1229,
            "expected {} - {} samples, got {}",
            1150,
            1229,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 1226 && out2[0].len() < 1232,
            "expected {} - {} samples, got {}",
            1226,
            1232,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let mask = vec![true, false];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let mask = vec![false, true];
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_resampler_fi_downsample() {
        // Replicate settings from reported issue
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Sinc::<f64>::new(
            16000 as f64 / 96000 as f64,
            1.0,
            params,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 140 && out[0].len() < 200,
            "expected {} - {} samples, got {}",
            140,
            200,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 167 && out2[0].len() < 173,
            "expected {} - {} samples, got {}",
            167,
            173,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fi_upsample() {
        // Replicate settings from reported issue
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Sinc::<f64>::new(
            192000 as f64 / 44100 as f64,
            1.0,
            params,
            1024,
            2,
            Fixed::Input,
        )
        .unwrap();
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert!(
            out[0].len() > 3800 && out[0].len() < 4458,
            "expected {} - {} samples, got {}",
            3800,
            4458,
            out[0].len()
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(out2.len(), 2, "Expected {} channels, got {}", 2, out2.len());
        assert!(
            out2[0].len() > 4455 && out2[0].len() < 4461,
            "expected {} - {} samples, got {}",
            4455,
            4461,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fo() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn reset_resampler_fo() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();

        let mut rng = rand::thread_rng();
        let mut waves = vec![vec![0.0f64; frames]; 2];
        waves
            .iter_mut()
            .for_each(|ch| ch.iter_mut().for_each(|s| *s = rng.gen()));
        let out1 = resampler.process(&waves, None).unwrap();
        resampler.reset();
        assert_eq!(
            frames,
            resampler.input_frames_next(),
            "Resampler requires different number of frames when new and after a reset."
        );
        let out2 = resampler.process(&waves, None).unwrap();
        assert_eq!(
            out1, out2,
            "Resampler gives different output when new and after a reset."
        );
    }

    #[test]
    fn make_resampler_fo_32() {
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        let mask = vec![true, false];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.input_frames_next();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        let mask = vec![false, true];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves, Some(&mask)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }

    #[test]
    fn make_resampler_fo_downsample() {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Sinc::<f64>::new(0.125, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 8192 && frames < 9000,
            "expected {}..{} samples, got {}",
            8192,
            9000,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 8189 && frames2 < 8195,
            "expected {}..{} samples, got {}",
            8189,
            8195,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn make_resampler_fo_upsample() {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 160,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = Sinc::<f64>::new(8.0, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        let frames = resampler.input_frames_next();
        println!("{}", frames);
        assert!(
            frames > 128 && frames < 300,
            "expected {}..{} samples, got {}",
            140,
            200,
            frames
        );
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves, None).unwrap();
        assert_eq!(out.len(), 2, "Expected {} channels, got {}", 2, out.len());
        assert_eq!(
            out[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out[0].len()
        );
        let frames2 = resampler.input_frames_next();
        assert!(
            frames2 > 125 && frames2 < 131,
            "expected {}..{} samples, got {}",
            125,
            131,
            frames2
        );
        let waves2 = vec![vec![0.0f64; frames2]; 2];
        let out2 = resampler.process(&waves2, None).unwrap();
        assert_eq!(
            out2[0].len(),
            1024,
            "Expected {} frames, got {}",
            1024,
            out2[0].len()
        );
    }

    #[test]
    fn check_fo_output_up() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fo_output_down() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(0.8, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fi_output_up() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn check_fi_output_down() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(0.8, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_output!(resampler);
    }

    #[test]
    fn resample_small_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_big_fo_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_small_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_big_fo_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_small_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_big_fi_up() {
        let ratio = 96000.0 / 44100.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn resample_small_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100000);
    }

    #[test]
    fn resample_big_fi_down() {
        let ratio = 44100.0 / 96000.0;
        let params = basic_params();
        let mut resampler = Sinc::<f32>::new(ratio, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        check_ratio!(resampler, ratio, 100);
    }

    #[test]
    fn check_fo_output_resize() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Output).unwrap();
        assert_eq!(resampler.output_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.output_frames_next(), 256);
        check_output!(resampler);
    }

    #[test]
    fn check_fi_output_resize() {
        let params = basic_params();
        let mut resampler = Sinc::<f64>::new(1.2, 1.0, params, 1024, 2, Fixed::Input).unwrap();
        assert_eq!(resampler.input_frames_next(), 1024);
        resampler.set_chunk_size(256).unwrap();
        assert_eq!(resampler.input_frames_next(), 256);
        check_output!(resampler);
    }
}
