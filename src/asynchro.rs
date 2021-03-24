use crate::windows::WindowFunction;

use crate::interpolation::*;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::interpolator_avx::AvxInterpolator;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::interpolator_sse::SseInterpolator;
use crate::sinc::make_sincs;
use crate::{InterpolationParameters, InterpolationType};

use num_traits::Float;
use std::error;

type Res<T> = Result<T, Box<dyn error::Error>>;

use crate::Resampler;
use crate::ResamplerError;

/// Functions for making the scalar product with a sinc
pub trait SincInterpolator<T> {
    /// Make the scalar product between the waveform starting at `index` and the sinc of `subindex`.
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T;

    /// Get sinc length
    fn len(&self) -> usize;

    /// Check if sincs are empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of sincs used for oversampling
    fn nbr_sincs(&self) -> usize;
}

/// A plain scalar interpolator
pub struct ScalarInterpolator<T> {
    sincs: Vec<Vec<T>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T: Float> SincInterpolator<T> for ScalarInterpolator<T> {
    /// Calculate the scalar produt of an input wave and the selected sinc filter
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        let wave_cut = &wave[index..(index + self.sincs[subindex].len())];
        let sinc = &self.sincs[subindex];
        unsafe {
            let mut acc0 = T::zero();
            let mut acc1 = T::zero();
            let mut acc2 = T::zero();
            let mut acc3 = T::zero();
            let mut acc4 = T::zero();
            let mut acc5 = T::zero();
            let mut acc6 = T::zero();
            let mut acc7 = T::zero();
            let mut idx = 0;
            for _ in 0..wave_cut.len() / 8 {
                acc0 = acc0 + *wave_cut.get_unchecked(idx) * *sinc.get_unchecked(idx);
                acc1 = acc1 + *wave_cut.get_unchecked(idx + 1) * *sinc.get_unchecked(idx + 1);
                acc2 = acc2 + *wave_cut.get_unchecked(idx + 2) * *sinc.get_unchecked(idx + 2);
                acc3 = acc3 + *wave_cut.get_unchecked(idx + 3) * *sinc.get_unchecked(idx + 3);
                acc4 = acc4 + *wave_cut.get_unchecked(idx + 4) * *sinc.get_unchecked(idx + 4);
                acc5 = acc5 + *wave_cut.get_unchecked(idx + 5) * *sinc.get_unchecked(idx + 5);
                acc6 = acc6 + *wave_cut.get_unchecked(idx + 6) * *sinc.get_unchecked(idx + 6);
                acc7 = acc7 + *wave_cut.get_unchecked(idx + 7) * *sinc.get_unchecked(idx + 7);
                idx += 8;
            }
            acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7
        }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T: Float> ScalarInterpolator<T> {
    /// Create a new ScalarInterpolator
    ///
    /// Parameters are:
    /// - `sinc_len`: Length of sinc functions.
    /// - `oversampling_factor`: Number of intermediate sincs (oversampling factor).
    /// - `f_cutoff`: Relative cutoff frequency.
    /// - `window`: Window function to use.
    pub fn new(
        sinc_len: usize,
        oversampling_factor: usize,
        f_cutoff: f32,
        window: WindowFunction,
    ) -> Self {
        assert!(sinc_len % 8 == 0);
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        }
    }
}

/// An asynchronous resampler that accepts a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub struct SincFixedIn<T> {
    nbr_channels: usize,
    chunk_size: usize,
    last_index: f64,
    resample_ratio: f64,
    resample_ratio_original: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
}

/// An asynchronous resampler that return a fixed number of audio frames.
/// The number of input frames required is given by the frames_needed function.
///
/// The resampling is done by creating a number of intermediate points (defined by oversampling_factor)
/// by sinc interpolation. The new samples are then calculated by interpolating between these points.
pub struct SincFixedOut<T> {
    nbr_channels: usize,
    chunk_size: usize,
    needed_input_size: usize,
    last_index: f64,
    current_buffer_fill: usize,
    resample_ratio: f64,
    resample_ratio_original: f64,
    interpolator: Box<dyn SincInterpolator<T>>,
    buffer: Vec<Vec<T>>,
    interpolation: InterpolationType,
}

macro_rules! impl_resampler {
    ($ft:ty, $rt:ty) => {
        impl $rt {
            pub fn make_interpolator(
                sinc_len: usize,
                resample_ratio: f64,
                f_cutoff: f32,
                oversampling_factor: usize,
                window: WindowFunction,
            ) -> Box<dyn SincInterpolator<$ft>> {
                let sinc_len = 8 * (((sinc_len as f32) / 8.0).ceil() as usize);
                let f_cutoff = if resample_ratio >= 1.0 {
                    f_cutoff
                } else {
                    f_cutoff * resample_ratio as f32
                };
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                    return Box::new(AvxInterpolator::<$ft>::new(
                        sinc_len,
                        oversampling_factor,
                        f_cutoff,
                        window,
                    ));
                }
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("sse3") {
                    return Box::new(SseInterpolator::<$ft>::new(
                        sinc_len,
                        oversampling_factor,
                        f_cutoff,
                        window,
                    ));
                }
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                if is_aarch64_feature_detected!("neon") {
                    return Box::new(NeonInterpolator::<$ft>::new(
                        sinc_len,
                        oversampling_factor,
                        f_cutoff,
                        window,
                    ));
                }
                Box::new(ScalarInterpolator::<$ft>::new(
                    sinc_len,
                    oversampling_factor,
                    f_cutoff,
                    window,
                ))
            }

            /// Perform cubic polynomial interpolation to get value at x.
            /// Input points are assumed to be at x = -1, 0, 1, 2
            fn interp_cubic(&self, x: $ft, yvals: &[$ft; 4]) -> $ft {
                let a0 = yvals[1];
                let a1 =
                    -(1.0 / 3.0) * yvals[0] - 0.5 * yvals[1] + yvals[2] - (1.0 / 6.0) * yvals[3];
                let a2 = 0.5 * (yvals[0] + yvals[2]) - yvals[1];
                let a3 = 0.5 * (yvals[1] - yvals[2]) + (1.0 / 6.0) * (yvals[3] - yvals[0]);
                let x2 = x * x;
                let x3 = x2 * x;
                a0 + a1 * x + a2 * x2 + a3 * x3
            }

            /// Linear interpolation between two points at x=0 and x=1
            fn interp_lin(&self, x: $ft, yvals: &[$ft; 2]) -> $ft {
                (1.0 - x) * yvals[0] + x * yvals[1]
            }
        }
    };
}
impl_resampler!(f32, SincFixedIn<f32>);
impl_resampler!(f64, SincFixedIn<f64>);
impl_resampler!(f32, SincFixedOut<f32>);
impl_resampler!(f64, SincFixedOut<f64>);

macro_rules! impl_new_sincfixedin {
    ($t:ty) => {
        impl SincFixedIn<$t> {
            /// Create a new SincFixedIn
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
            /// - `chunk_size`: size of input data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new(
                resample_ratio: f64,
                parameters: InterpolationParameters,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new SincFixedIn, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
                    resample_ratio, chunk_size, nbr_channels, parameters
                );
                let interpolator = Self::make_interpolator(parameters.sinc_len,
                    resample_ratio,
                    parameters.f_cutoff,
                    parameters.oversampling_factor,
                    parameters.window);

                Self::new_with_interpolator(resample_ratio, parameters.interpolation, interpolator, chunk_size, nbr_channels)
            }

            /// Create a new SincFixedIn using an existing Interpolator
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `interpolation_type`: Parameters for interpolation, see `InterpolationParameters`
            /// - `interpolator`:  The interpolator to use
            /// - `chunk_size`: size of output data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new_with_interpolator(
                resample_ratio: f64,
                interpolation_type: InterpolationType,
                interpolator: Box<dyn SincInterpolator<$t>>,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {

                let buffer = vec![vec![0.0; chunk_size + 2 * interpolator.len()]; nbr_channels];

                SincFixedIn {
                    nbr_channels,
                    chunk_size,
                    last_index: -((interpolator.len() / 2) as f64),
                    resample_ratio,
                    resample_ratio_original: resample_ratio,
                    interpolator,
                    buffer,
                    interpolation: interpolation_type,
                }
            }


        }
    }
}
impl_new_sincfixedin!(f32);
impl_new_sincfixedin!(f64);

macro_rules! resampler_sincfixedin {
    ($t:ty) => {
        impl Resampler<$t> for SincFixedIn<$t> {
            /// Resample a chunk of audio. The input length is fixed, and the output varies in length.
            /// If the waveform for a channel is empty, this channel will be ignored and produce a
            /// corresponding empty output waveform.
            /// # Errors
            ///
            /// The function returns an error if the length of the input data is not equal
            /// to the number of channels and chunk size defined when creating the instance.
            fn process_into_buffer(
                &mut self,
                wave_in: &[Vec<$t>],
                wave_out: &mut Vec<Vec<$t>>,
            ) -> Res<()> {
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                let mut used_channels = Vec::new();
                for (chan, wave) in wave_in.iter().enumerate() {
                    if !wave.is_empty() {
                        used_channels.push(chan);
                        if wave.len() != self.chunk_size {
                            return Err(Box::new(ResamplerError::new(
                                "Wrong number of frames in input",
                            )));
                        }
                    }
                }
                let sinc_len = self.interpolator.len();
                let oversampling_factor = self.interpolator.nbr_sincs();
                let end_idx = self.chunk_size as isize - (sinc_len as isize + 1);
                //update buffer with new data
                for wav in self.buffer.iter_mut() {
                    for idx in 0..(2 * sinc_len) {
                        wav[idx] = wav[idx + self.chunk_size];
                    }
                }

                wave_out.resize(self.nbr_channels, Vec::new());

                for chan in used_channels.iter() {
                    for (idx, sample) in wave_in[*chan].iter().enumerate() {
                        self.buffer[*chan][idx + 2 * sinc_len] = *sample;
                    }
                    let new_len = (self.chunk_size as f64 * self.resample_ratio + 10.0) as usize;
                    wave_out[*chan].resize(new_len, 0.0);
                }

                let mut idx = self.last_index;
                let t_ratio = 1.0 / self.resample_ratio as f64;

                let mut n = 0;

                match self.interpolation {
                    InterpolationType::Cubic => {
                        let mut points = [0.0 as $t; 4];
                        let mut nearest = [(0isize, 0isize); 4];
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                            let frac = idx * oversampling_factor as f64
                                - (idx * oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                wave_out[*chan][n] = self.interp_cubic(frac_offset, &points);
                            }
                            n += 1;
                        }
                    }
                    InterpolationType::Linear => {
                        let mut points = [0.0 as $t; 2];
                        let mut nearest = [(0isize, 0isize); 2];
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                            let frac = idx * oversampling_factor as f64
                                - (idx * oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                wave_out[*chan][n] = self.interp_lin(frac_offset, &points);
                            }
                            n += 1;
                        }
                    }
                    InterpolationType::Nearest => {
                        let mut point;
                        let mut nearest;
                        while idx < end_idx as f64 {
                            idx += t_ratio;
                            nearest = get_nearest_time(idx, oversampling_factor as isize);
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                point = self.interpolator.get_sinc_interpolated(
                                    &buf,
                                    (nearest.0 + 2 * sinc_len as isize) as usize,
                                    nearest.1 as usize,
                                );
                                wave_out[*chan][n] = point;
                            }
                            n += 1;
                        }
                    }
                }

                // store last index for next iteration
                self.last_index = idx - self.chunk_size as f64;
                for chan in used_channels.iter() {
                    //for w in wave_out.iter_mut() {
                    wave_out[*chan].truncate(n);
                }
                trace!(
                    "Resampling channels {:?}, {} frames in, {} frames out",
                    used_channels,
                    self.chunk_size,
                    n,
                );
                Ok(())
            }

            /// Update the resample ratio. New value must be within +-10% of the original one
            fn set_resample_ratio(&mut self, new_ratio: f64) -> Res<()> {
                trace!("Change resample ratio to {}", new_ratio);
                if (new_ratio / self.resample_ratio_original > 0.9)
                    && (new_ratio / self.resample_ratio_original < 1.1)
                {
                    self.resample_ratio = new_ratio;
                    Ok(())
                } else {
                    Err(Box::new(ResamplerError::new(
                        "New resample ratio is too far off from original",
                    )))
                }
            }
            /// Update the resample ratio relative to the original one
            fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> Res<()> {
                let new_ratio = self.resample_ratio_original * rel_ratio;
                self.set_resample_ratio(new_ratio)
            }

            /// Query for the number of frames needed for the next call to "process".
            /// Will always return the chunk_size defined when creating the instance.
            fn nbr_frames_needed(&self) -> usize {
                self.chunk_size
            }
        }
    };
}
resampler_sincfixedin!(f32);
resampler_sincfixedin!(f64);

macro_rules! impl_new_sincfixedout {
    ($t:ty) => {
        impl SincFixedOut<$t> {
            /// Create a new SincFixedOut
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `parameters`: Parameters for interpolation, see `InterpolationParameters`
            /// - `chunk_size`: size of output data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new(
                resample_ratio: f64,
                parameters: InterpolationParameters,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new SincFixedIn, ratio: {}, chunk_size: {}, channels: {}, parameters: {:?}",
                    resample_ratio, chunk_size, nbr_channels, parameters
                );
                let interpolator = Self::make_interpolator(parameters.sinc_len,
                    resample_ratio,
                    parameters.f_cutoff,
                    parameters.oversampling_factor,
                    parameters.window);

                Self::new_with_interpolator(resample_ratio, parameters.interpolation, interpolator, chunk_size, nbr_channels)
            }

            /// Create a new SincFixedOut using an existing Interpolator
            ///
            /// Parameters are:
            /// - `resample_ratio`: Ratio between output and input sample rates.
            /// - `interpolation_type`: Parameters for interpolation, see `InterpolationParameters`
            /// - `interpolator`:  The interpolator to use
            /// - `chunk_size`: size of output data in frames
            /// - `nbr_channels`: number of channels in input/output
            pub fn new_with_interpolator(
                resample_ratio: f64,
                interpolation_type: InterpolationType,
                interpolator: Box<dyn SincInterpolator<$t>>,
                chunk_size: usize,
                nbr_channels: usize,
            ) -> Self {

                let needed_input_size =
                    (chunk_size as f64 / resample_ratio).ceil() as usize + 2 + interpolator.len() / 2;
                let buffer = vec![vec![0.0; 3 * needed_input_size / 2 + 2 * interpolator.len()]; nbr_channels];

                SincFixedOut {
                    nbr_channels,
                    chunk_size,
                    needed_input_size,
                    last_index: -((interpolator.len() / 2) as f64),
                    current_buffer_fill: needed_input_size,
                    resample_ratio,
                    resample_ratio_original: resample_ratio,
                    interpolator,
                    buffer,
                    interpolation: interpolation_type,
                }
            }
        }
    }
}
impl_new_sincfixedout!(f32);
impl_new_sincfixedout!(f64);

macro_rules! resampler_sincfixedout {
    ($t:ty) => {
        impl Resampler<$t> for SincFixedOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.needed_input_size
            }

            /// Update the resample ratio. New value must be within +-10% of the original one
            fn set_resample_ratio(&mut self, new_ratio: f64) -> Res<()> {
                trace!("Change resample ratio to {}", new_ratio);
                if (new_ratio / self.resample_ratio_original > 0.9)
                    && (new_ratio / self.resample_ratio_original < 1.1)
                {
                    self.resample_ratio = new_ratio;
                    self.needed_input_size = (self.last_index as f32
                        + self.chunk_size as f32 / self.resample_ratio as f32
                        + self.interpolator.len() as f32)
                        .ceil() as usize
                        + 2;
                    Ok(())
                } else {
                    Err(Box::new(ResamplerError::new(
                        "New resample ratio is too far off from original",
                    )))
                }
            }

            /// Update the resample ratio relative to the original one
            fn set_resample_ratio_relative(&mut self, rel_ratio: f64) -> Res<()> {
                let new_ratio = self.resample_ratio_original * rel_ratio;
                self.set_resample_ratio(new_ratio)
            }

            /// Resample a chunk of audio. The required input length is provided by
            /// the "nbr_frames_needed" function, and the output length is fixed.
            /// If the waveform for a channel is empty, this channel will be ignored and produce a
            /// corresponding empty output waveform.
            /// # Errors
            ///
            /// The function returns an error if the length of the input data is not
            /// equal to the number of channels defined when creating the instance,
            /// and the number of audio frames given by "nbr_frames_needed".
            fn process_into_buffer(&mut self, wave_in: &[Vec<$t>], wave_out: &mut Vec<Vec<$t>>) -> Res<()> {
                //update buffer with new data
                if wave_in.len() != self.nbr_channels {
                    return Err(Box::new(ResamplerError::new(
                        "Wrong number of channels in input",
                    )));
                }
                let sinc_len = self.interpolator.len();
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut used_channels = Vec::new();
                for (chan, wave) in wave_in.iter().enumerate() {
                    if !wave.is_empty() {
                        used_channels.push(chan);
                        if wave.len() != self.needed_input_size {
                            return Err(Box::new(ResamplerError::new(
                                "Wrong number of frames in input",
                            )));
                        }
                    }
                }
                for wav in self.buffer.iter_mut() {
                    for idx in 0..(2 * sinc_len) {
                        wav[idx] = wav[idx + self.current_buffer_fill];
                    }
                }
                self.current_buffer_fill = self.needed_input_size;

                wave_out.resize(self.nbr_channels, Vec::new());

                for chan in used_channels.iter() {
                    for (idx, sample) in wave_in[*chan].iter().enumerate() {
                        self.buffer[*chan][idx + 2 * sinc_len] = *sample;
                    }
                    wave_out[*chan].resize(self.chunk_size, 0.0);
                }

                let mut idx = self.last_index;
                let t_ratio = 1.0 / self.resample_ratio as f64;

                match self.interpolation {
                    InterpolationType::Cubic => {
                        let mut points = [0.0 as $t; 4];
                        let mut nearest = [(0isize, 0isize); 4];
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                            let frac = idx * oversampling_factor as f64
                                - (idx * oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                wave_out[*chan][n] = self.interp_cubic(frac_offset, &points);
                            }
                        }
                    }
                    InterpolationType::Linear => {
                        let mut points = [0.0 as $t; 2];
                        let mut nearest = [(0isize, 0isize); 2];
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                            let frac = idx * oversampling_factor as f64
                                - (idx * oversampling_factor as f64).floor();
                            let frac_offset = frac as $t;
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                    *p = self.interpolator.get_sinc_interpolated(
                                        &buf,
                                        (n.0 + 2 * sinc_len as isize) as usize,
                                        n.1 as usize,
                                    );
                                }
                                wave_out[*chan][n] = self.interp_lin(frac_offset, &points);
                            }
                        }
                    }
                    InterpolationType::Nearest => {
                        let mut point;
                        let mut nearest;
                        for n in 0..self.chunk_size {
                            idx += t_ratio;
                            nearest = get_nearest_time(idx, oversampling_factor as isize);
                            for chan in used_channels.iter() {
                                let buf = &self.buffer[*chan];
                                point = self.interpolator.get_sinc_interpolated(
                                    &buf,
                                    (nearest.0 + 2 * sinc_len as isize) as usize,
                                    nearest.1 as usize,
                                );
                                wave_out[*chan][n] = point;
                            }
                        }
                    }
                }

                let prev_input_len = self.needed_input_size;
                // store last index for next iteration
                self.last_index = idx - self.current_buffer_fill as f64;
                self.needed_input_size = (self.last_index as f32
                    + self.chunk_size as f32 / self.resample_ratio as f32
                    + sinc_len as f32)
                    .ceil() as usize
                    + 2;
                trace!(
                    "Resampling channels {:?}, {} frames in, {} frames out. Next needed length: {} frames, last index {}",
                    used_channels,
                    prev_input_len,
                    self.chunk_size,
                    self.needed_input_size,
                    self.last_index
                );
                Ok(())
            }
        }
    }
}
resampler_sincfixedout!(f32);
resampler_sincfixedout!(f64);

#[cfg(test)]
mod tests {
    use crate::asynchro::ScalarInterpolator;
    use crate::asynchro::SincInterpolator;
    use crate::InterpolationParameters;
    use crate::InterpolationType;
    use crate::Resampler;
    use crate::WindowFunction;
    use crate::{SincFixedIn, SincFixedOut};
    use num_traits::Float;
    use rand::Rng;

    fn get_sinc_interpolated<T: Float>(wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        wave_cut
            .iter()
            .zip(sinc.iter())
            .fold(T::zero(), |acc, (x, y)| acc + *x * *y)
    }

    #[test]
    fn test_scalar_interpolator_64() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_scalar_interpolator_32() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;

        let interpolator =
            ScalarInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window);
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &interpolator.sincs[123]);
        assert!((value - check).abs() < 1.0e-6);
    }

    #[test]
    fn int_cubic() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = resampler.interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = [1.0f32, 5.0f32];
        let interp = resampler.interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = resampler.interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let yvals = [1.0f64, 5.0f64];
        let interp = resampler.interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }

    #[test]
    fn make_resampler_fi() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f64; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
    }

    #[test]
    fn make_resampler_fi_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f32; 1024]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f64>::new(1.2, params, 1024, 2);
        let waves = vec![vec![0.0f64; 1024], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].len() > 1150 && out[0].len() < 1250);
        assert!(out[1].is_empty());
        let waves = vec![Vec::new(), vec![0.0f64; 1024]];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[1].len() > 1150 && out[0].len() < 1250);
        assert!(out[0].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_32() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f32>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let waves = vec![vec![0.0f32; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let params = InterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedOut::<f64>::new(1.2, params, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        println!("{}", frames);
        assert!(frames > 800 && frames < 900);
        let mut waves = vec![vec![0.0f64; frames], Vec::new()];
        waves[0][100] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
        println!("{:?}", out[0]);
        let summed = out[0].iter().sum::<f64>();
        println!("sum: {}", summed);
        assert!(summed < 4.0);
        assert!(summed > 2.0);

        let frames = resampler.nbr_frames_needed();
        let mut waves = vec![Vec::new(), vec![0.0f64; frames]];
        waves[1][10] = 3.0;
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].len(), 1024);
        assert!(out[0].is_empty());
        let summed = out[1].iter().sum::<f64>();
        assert!(summed < 4.0);
        assert!(summed > 2.0);
    }
}
