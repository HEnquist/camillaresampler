use crate::sinc::make_sincs;
use crate::windows::WindowFunction;
use num_complex::Complex;
use num_integer as integer;
use num_traits::Zero;
use std::error;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn error::Error>>;

use crate::Resampler;
use crate::ResamplerError;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// A helper for resampling a single chunk of data.
struct FftResampler<T> {
    fft_size_in: usize,
    fft_size_out: usize,
    filter_f: Vec<Complex<T>>,
    fft: Arc<dyn RealToComplex<T>>,
    ifft: Arc<dyn ComplexToReal<T>>,
    scratch_fw: Vec<Complex<T>>,
    scratch_inv: Vec<Complex<T>>,
    input_buf: Vec<T>,
    input_f: Vec<Complex<T>>,
    output_f: Vec<Complex<T>>,
    output_buf: Vec<T>,
}

/// A synchronous resampler that needs a fixed number of audio frames for input
/// and returns a variable number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedIn<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    input_buffers: Vec<Vec<T>>,
    saved_frames: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that needs a varying number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedOut<T> {
    nbr_channels: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    fft_size_out: usize,
    overlaps: Vec<Vec<T>>,
    output_buffers: Vec<Vec<T>>,
    saved_frames: usize,
    frames_needed: usize,
    resampler: FftResampler<T>,
}

/// A synchronous resampler that accepts a fixed number of audio frames for input
/// and returns a fixed number of frames.
///
/// The resampling is done by FFT:ing the input data. The spectrum is then extended or
/// truncated as well as multiplied with an antialiasing filter
/// before it's inverse transformed to get the resampled waveforms.
pub struct FftFixedInOut<T> {
    nbr_channels: usize,
    chunk_size_in: usize,
    chunk_size_out: usize,
    fft_size_in: usize,
    overlaps: Vec<Vec<T>>,
    resampler: FftResampler<T>,
}

macro_rules! impl_resampler {
    ($ft:ty, $rt:ty) => {
        impl $rt {
            //
            pub fn new(fft_size_in: usize, fft_size_out: usize) -> Self {
                // calculate antialiasing cutoff
                let cutoff = if fft_size_in > fft_size_out {
                    0.4f32.powf(16.0 / fft_size_in as f32) * fft_size_out as f32
                        / fft_size_in as f32
                } else {
                    0.4f32.powf(16.0 / fft_size_in as f32)
                };
                debug!(
                    "Create new FftResampler, fft_size_in: {}, fft_size_out: {}, cutoff: {}",
                    fft_size_in, fft_size_out, cutoff
                );
                let sinc =
                    make_sincs::<$ft>(fft_size_in, 1, cutoff, WindowFunction::BlackmanHarris2);
                let mut filter_t: Vec<$ft> = vec![0.0; 2 * fft_size_in];
                let mut filter_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_in + 1];
                for n in 0..fft_size_in {
                    filter_t[n] = sinc[0][n] / (2 * fft_size_in) as $ft;
                }

                let input_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_in + 1];
                let input_buf: Vec<$ft> = vec![0.0; 2 * fft_size_in];
                let output_f: Vec<Complex<$ft>> = vec![Complex::zero(); fft_size_out + 1];
                let output_buf: Vec<$ft> = vec![0.0; 2 * fft_size_out];
                let mut planner = RealFftPlanner::<$ft>::new();
                let fft = planner.plan_fft_forward(2 * fft_size_in);
                let ifft = planner.plan_fft_inverse(2 * fft_size_out);
                fft.process(&mut filter_t, &mut filter_f).unwrap();
                let scratch_fw = fft.make_scratch_vec();
                let scratch_inv = ifft.make_scratch_vec();
                FftResampler {
                    fft_size_in,
                    fft_size_out,
                    filter_f,
                    fft,
                    ifft,
                    input_buf,
                    input_f,
                    output_f,
                    output_buf,
                    scratch_fw,
                    scratch_inv,
                }
            }

            /// Resample a small chunk
            fn resample_unit(
                &mut self,
                wave_in: &[$ft],
                wave_out: &mut [$ft],
                overlap: &mut [$ft],
            ) {
                // Copy to input buffer and clear padding area
                self.input_buf[0..self.fft_size_in].copy_from_slice(wave_in);
                for item in self
                    .input_buf
                    .iter_mut()
                    .skip(self.fft_size_in)
                    .take(self.fft_size_in)
                {
                    *item = 0.0;
                }

                // FFT and store result in history, update index
                self.fft
                    .process_with_scratch(
                        &mut self.input_buf,
                        &mut self.input_f,
                        &mut self.scratch_fw,
                    )
                    .unwrap();

                // multiply with filter FT
                self.input_f
                    .iter_mut()
                    .take(self.fft_size_in + 1)
                    .zip(self.filter_f.iter())
                    .for_each(|(spec, filt)| *spec *= filt);
                let new_len = if self.fft_size_in < self.fft_size_out {
                    self.fft_size_in + 1
                } else {
                    self.fft_size_out
                };

                // copy to modified spectrum
                self.output_f[0..new_len].copy_from_slice(&self.input_f[0..new_len]);
                for val in self.output_f[new_len..].iter_mut() {
                    *val = Complex::zero();
                }
                //self.output_f[self.fft_size_out] = self.input_f[self.fft_size_in];

                // IFFT result, store result and overlap
                self.ifft
                    .process_with_scratch(
                        &mut self.output_f,
                        &mut self.output_buf,
                        &mut self.scratch_inv,
                    )
                    .unwrap();
                for (n, item) in wave_out.iter_mut().enumerate().take(self.fft_size_out) {
                    *item = self.output_buf[n] + overlap[n];
                }
                overlap.copy_from_slice(&self.output_buf[self.fft_size_out..]);
            }
        }
    };
}
impl_resampler!(f32, FftResampler<f32>);
impl_resampler!(f64, FftResampler<f64>);

macro_rules! impl_fixedinout {
    ($ft:ty) => {
        impl FftFixedInOut<$ft> {
            /// Create a new FftFixedInOut
            ///
            /// Parameters are:
            /// - `fs_in`: Input sample rate.
            /// - `fs_out`: Output sample rate.
            /// - `chunk_size_in`: desired length of input data in frames, actual value may be different.
            /// - `nbr_channels`: number of channels in input/output.
            pub fn new(
                fs_in: usize,
                fs_out: usize,
                chunk_size_in: usize,
                nbr_channels: usize,
            ) -> Self {
                debug!(
                    "Create new FftFixedInOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}",
                    fs_in, fs_out, chunk_size_in, nbr_channels
                );

                let gcd = integer::gcd(fs_in, fs_out);
                let min_chunk_out = fs_out/gcd;
                let wanted = chunk_size_in;
                let fft_chunks = (wanted as f32 / min_chunk_out as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];

                FftFixedInOut {
                    nbr_channels,
                    chunk_size_in: fft_size_in,
                    chunk_size_out: fft_size_out,
                    fft_size_in,
                    overlaps,
                    resampler,
                }
            }
        }
    }
}
impl_fixedinout!(f64);
impl_fixedinout!(f32);

macro_rules! resampler_FftFixedinout {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedInOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.fft_size_in
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Resample a chunk of audio. The input and output lengths are fixed.
            /// If the waveform for a channel is empty, this channel will be ignored and produce a
            /// corresponding empty output waveform.
            /// # Errors
            ///
            /// The function returns an error if the size of the input data is not equal
            /// to the number of channels and input size defined when creating the instance.
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
                        if wave.len() != self.chunk_size_in {
                            return Err(Box::new(ResamplerError::new(
                                format!(
                                    "Wrong number of frames in input, expected {}, got {}",
                                    self.chunk_size_in,
                                    wave_in[0].len()
                                )
                                .as_str(),
                            )));
                        }
                    }
                }
                wave_out.resize(self.nbr_channels, Vec::new());
                for chan in used_channels.iter() {
                    wave_out[*chan].resize(self.chunk_size_out, 0.0);
                }

                for n in used_channels.iter() {
                    self.resampler.resample_unit(
                        &wave_in[*n],
                        &mut wave_out[*n],
                        &mut self.overlaps[*n],
                    )
                }
                Ok(())
            }
        }
    };
}
resampler_FftFixedinout!(f32);
resampler_FftFixedinout!(f64);

macro_rules! impl_fixedout {
    ($ft:ty) => {
        impl FftFixedOut<$ft> {
            /// Create a new FftFixedOut
            ///
            /// Parameters are:
            /// - `fs_in`: Input sample rate.
            /// - `fs_out`: Output sample rate.
            /// - `chunk_size_out`: length of output data in frames.
            /// - `sub_chunks`: desired number of subchunks for processing, actual number may be different.
            /// - `nbr_channels`: number of channels in input/output.
            pub fn new(
                fs_in: usize,
                fs_out: usize,
                chunk_size_out: usize,
                sub_chunks: usize,
                nbr_channels: usize,
            ) -> Self {


                let gcd = integer::gcd(fs_in, fs_out);
                let min_chunk_out = fs_out/gcd;
                let wanted_subsize = chunk_size_out/sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_out as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);

                debug!(
                    "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
                    fs_in, fs_out, chunk_size_out, nbr_channels, fft_size_in, fft_size_out
                );

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];
                let output_buffers: Vec<Vec<$ft>> = vec![vec![0.0; chunk_size_out+fft_size_out]; nbr_channels];

                let saved_frames = 0;
                let chunks_needed = (chunk_size_out as f32 / fft_size_out as f32).ceil() as usize;
                let frames_needed = chunks_needed*fft_size_in;

                FftFixedOut {
                    nbr_channels,
                    chunk_size_out,
                    fft_size_in,
                    fft_size_out,
                    overlaps,
                    output_buffers,
                    saved_frames,
                    frames_needed,
                    resampler,
                }
            }
        }
    }
}
impl_fixedout!(f64);
impl_fixedout!(f32);

macro_rules! resampler_FftFixedout {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedOut<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.frames_needed
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
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
                        if wave.len() != self.frames_needed {
                            return Err(Box::new(ResamplerError::new(
                                format!(
                                    "Wrong number of frames in input, expected {}, got {}",
                                    self.frames_needed,
                                    wave.len()
                                )
                                .as_str(),
                            )));
                        }
                    }
                }

                wave_out.resize(self.nbr_channels, Vec::new());
                for chan in used_channels.iter() {
                    let new_len = self.output_buffers[*chan].len();
                    wave_out[*chan].resize(new_len, 0.0);
                    wave_out[*chan].copy_from_slice(&self.output_buffers[*chan]);
                }

                for n in used_channels.iter() {
                    for (in_chunk, out_chunk) in wave_in[*n]
                        .chunks(self.fft_size_in)
                        .zip(wave_out[*n][self.saved_frames..].chunks_mut(self.fft_size_out))
                    {
                        self.resampler
                            .resample_unit(in_chunk, out_chunk, &mut self.overlaps[*n]);
                    }
                }
                let processed_frames =
                    self.saved_frames + self.fft_size_out * (self.frames_needed / self.fft_size_in);

                // save extra frames for next round
                self.saved_frames = processed_frames - self.chunk_size_out;
                if processed_frames > self.chunk_size_out {
                    for n in used_channels.iter() {
                        self.output_buffers[*n][0..self.saved_frames].copy_from_slice(
                            &wave_out[*n]
                                [self.chunk_size_out..(self.chunk_size_out + self.saved_frames)],
                        );
                    }
                }
                for n in used_channels.iter() {
                    wave_out[*n].truncate(self.chunk_size_out);
                }
                //calculate number of needed frames from next round
                let frames_needed_out = if self.chunk_size_out > self.saved_frames {
                    self.chunk_size_out - self.saved_frames
                } else {
                    0
                };
                let chunks_needed =
                    (frames_needed_out as f32 / self.fft_size_out as f32).ceil() as usize;
                self.frames_needed = chunks_needed * self.fft_size_in;
                Ok(())
            }
        }
    };
}
resampler_FftFixedout!(f32);
resampler_FftFixedout!(f64);

macro_rules! impl_fixedin {
    ($ft:ty) => {
        impl FftFixedIn<$ft> {
            /// Create a new FftFixedOut
            ///
            /// Parameters are:
            /// - `fs_in`: Input sample rate.
            /// - `fs_out`: Output sample rate.
            /// - `chunk_size_out`: length of output data in frames.
            /// - `sub_chunks`: desired number of subchunks for processing, actual number used may be different.
            /// - `nbr_channels`: number of channels in input/output.
            pub fn new(
                fs_in: usize,
                fs_out: usize,
                chunk_size_in: usize,
                sub_chunks: usize,
                nbr_channels: usize,
            ) -> Self {


                let gcd = integer::gcd(fs_in, fs_out);
                let min_chunk_in = fs_in/gcd;
                let wanted_subsize = chunk_size_in/sub_chunks;
                let fft_chunks = (wanted_subsize as f32 / min_chunk_in as f32).ceil() as usize;
                let fft_size_out = fft_chunks * fs_out / gcd;
                let fft_size_in = fft_chunks * fs_in / gcd;

                let resampler = FftResampler::<$ft>::new(fft_size_in, fft_size_out);
                debug!(
                    "Create new FftFixedOut, fs_in: {}, fs_out: {} chunk_size_in: {}, channels: {}, fft_size_in: {}, fft_size_out: {}",
                    fs_in, fs_out, chunk_size_in, nbr_channels, fft_size_in, fft_size_out
                );

                let overlaps: Vec<Vec<$ft>> = vec![vec![0.0; fft_size_out]; nbr_channels];
                let input_buffers: Vec<Vec<$ft>> = vec![vec![0.0; chunk_size_in+fft_size_out]; nbr_channels];

                let saved_frames = 0;

                FftFixedIn {
                    nbr_channels,
                    chunk_size_in,
                    fft_size_in,
                    fft_size_out,
                    overlaps,
                    input_buffers,
                    saved_frames,
                    resampler,
                }
            }
        }
    }
}
impl_fixedin!(f64);
impl_fixedin!(f32);

macro_rules! resampler_FftFixedin {
    ($t:ty) => {
        impl Resampler<$t> for FftFixedIn<$t> {
            /// Query for the number of frames needed for the next call to "process".
            fn nbr_frames_needed(&self) -> usize {
                self.chunk_size_in
            }

            /// Update the resample ratio. This is not supported by this resampler and always returns an error.
            fn set_resample_ratio(&mut self, _new_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
            }

            /// Update the resample ratio relative to the original one.
            /// This is not supported by this resampler and always returns an error.
            fn set_resample_ratio_relative(&mut self, _rel_ratio: f64) -> Res<()> {
                Err(Box::new(ResamplerError::new(
                    "Not possible to adjust a synchronous resampler)",
                )))
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
                        if wave.len() != self.chunk_size_in {
                            return Err(Box::new(ResamplerError::new(
                                format!(
                                    "Wrong number of frames in input, expected {}, got {}",
                                    self.chunk_size_in,
                                    wave.len()
                                )
                                .as_str(),
                            )));
                        }
                    }
                }

                let mut input_temp = vec![Vec::new(); self.nbr_channels];
                for chan in used_channels.iter() {
                    input_temp[*chan] = vec![0.0; self.saved_frames + self.chunk_size_in];
                }

                // copy new samples to input buffer
                for n in used_channels.iter() {
                    for (input, buffer) in self.input_buffers[*n]
                        .iter()
                        .take(self.saved_frames)
                        .zip(input_temp[*n].iter_mut())
                    {
                        *buffer = *input;
                    }
                }
                for n in used_channels.iter() {
                    for (input, buffer) in wave_in[*n].iter().zip(
                        input_temp[*n]
                            .iter_mut()
                            .skip(self.saved_frames)
                            .take(self.chunk_size_in),
                    ) {
                        *buffer = *input;
                    }
                }
                self.saved_frames += self.chunk_size_in;

                let nbr_chunks_ready =
                    (self.saved_frames as f32 / self.fft_size_in as f32).floor() as usize;
                wave_out.resize(self.nbr_channels, Vec::new());

                for chan in used_channels.iter() {
                    wave_out[*chan].resize(nbr_chunks_ready * self.fft_size_out, 0.0);
                }
                for n in used_channels.iter() {
                    for (in_chunk, out_chunk) in input_temp[*n]
                        .chunks(self.fft_size_in)
                        .take(nbr_chunks_ready)
                        .zip(wave_out[*n].chunks_mut(self.fft_size_out))
                    {
                        self.resampler
                            .resample_unit(in_chunk, out_chunk, &mut self.overlaps[*n]);
                    }
                }

                // save extra frames for next round
                let frames_in_used = nbr_chunks_ready * self.fft_size_in;
                let extra = self.saved_frames - frames_in_used;

                if self.saved_frames > frames_in_used {
                    for n in used_channels.iter() {
                        for (input, buffer) in input_temp[*n]
                            .iter()
                            .skip(frames_in_used)
                            .take(extra)
                            .zip(self.input_buffers[*n].iter_mut())
                        {
                            *buffer = *input;
                        }
                    }
                }
                self.saved_frames = extra;
                Ok(())
            }
        }
    };
}
resampler_FftFixedin!(f32);
resampler_FftFixedin!(f64);

#[cfg(test)]
mod tests {
    use crate::synchro::{FftFixedIn, FftFixedInOut, FftFixedOut, FftResampler};
    use crate::Resampler;

    #[test]
    fn resample_unit() {
        let mut resampler = FftResampler::<f64>::new(147, 1000);
        let mut wave_in = vec![0.0; 147];

        wave_in[0] = 0.3;
        wave_in[1] = 0.7;
        wave_in[2] = 1.0;
        wave_in[3] = 1.0;
        wave_in[4] = 0.7;
        wave_in[5] = 0.3;

        let mut wave_out = vec![0.0; 1000];
        let mut overlap = vec![0.0; 1000];
        resampler.resample_unit(&wave_in, &mut wave_out, &mut overlap);
        let vecsum = wave_out.iter().sum::<f64>();
        let maxval = wave_out.iter().cloned().fold(0. / 0., f64::max);
        assert!((vecsum - 4.0 * 1000.0 / 147.0).abs() < 1.0e-6);
        assert!((maxval - 1.0).abs() < 0.1);
    }

    #[test]
    fn make_resampler_fio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
    }

    #[test]
    fn make_resampler_fio_skipped() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 48000, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1120);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }

    #[test]
    fn make_resampler_fo_skipped() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fo_empty() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 192000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 294);
        let waves = vec![Vec::new(); 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
    }

    #[test]
    fn make_resampler_fi_skipped() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![vec![0.0f64; frames], Vec::new()];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 640);
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fi_empty() {
        let mut resampler = FftFixedIn::<f64>::new(44100, 48000, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 1024);
        let waves = vec![Vec::new(); 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
        assert!(out[1].is_empty());
    }

    #[test]
    fn make_resampler_fio_unusualratio() {
        // asking for 1024 give the nearest which is 1029 -> 1120
        let mut resampler = FftFixedInOut::<f64>::new(44100, 44110, 1024, 2);
        let frames = resampler.nbr_frames_needed();
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4411);
    }

    #[test]
    fn make_resampler_fo_unusualratio() {
        let mut resampler = FftFixedOut::<f64>::new(44100, 44110, 1024, 2, 2);
        let frames = resampler.nbr_frames_needed();
        assert_eq!(frames, 4410);
        let waves = vec![vec![0.0f64; frames]; 2];
        let out = resampler.process(&waves).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1024);
    }
}
