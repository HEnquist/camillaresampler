use crate::asynchro::InnerResampler;
use crate::interpolation::*;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_avx::AvxInterpolator;
#[cfg(target_arch = "aarch64")]
use crate::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
#[cfg(target_arch = "x86_64")]
use crate::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;
use crate::sinc_interpolator::{ScalarInterpolator, SincInterpolator};
use crate::windows::WindowFunction;
use crate::Sample;
use audioadapter::AdapterMut;

macro_rules! t {
    // Shorter form of T::coerce(value)
    ($expression:expr) => {
        T::coerce($expression)
    };
}

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
pub fn interp_cubic<T>(x: T, yvals: &[T; 4]) -> T
where
    T: Sample,
{
    let a0 = yvals[1];
    let a1 = -t!(1.0 / 3.0) * yvals[0] - t!(0.5) * yvals[1] + yvals[2] - t!(1.0 / 6.0) * yvals[3];
    let a2 = t!(0.5) * (yvals[0] + yvals[2]) - yvals[1];
    let a3 = t!(0.5) * (yvals[1] - yvals[2]) + t!(1.0 / 6.0) * (yvals[3] - yvals[0]);
    let x2 = x * x;
    let x3 = x2 * x;
    a0 + a1 * x + a2 * x2 + a3 * x3
}

/// Perform quadratic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = 0, 1, 2.
pub fn interp_quad<T>(x: T, yvals: &[T; 3]) -> T
where
    T: Sample,
{
    let a2 = yvals[0] - t!(2.0) * yvals[1] + yvals[2];
    let a1 = -t!(3.0) * yvals[0] + t!(4.0) * yvals[1] - yvals[2];
    let a0 = t!(2.0) * yvals[0];
    let x2 = x * x;
    t!(0.5) * (a0 + a1 * x + a2 * x2)
}

/// Perform linear interpolation between two points at x=0 and x=1.
pub fn interp_lin<T>(x: T, yvals: &[T; 2]) -> T
where
    T: Sample,
{
    yvals[0] + x * (yvals[1] - yvals[0])
}

pub struct InnerSinc<T> {
    pub interpolator: Box<dyn SincInterpolator<T>>,
    pub interpolation: SincInterpolationType,
}

impl<T> InnerResampler<T> for InnerSinc<T>
where
    T: Sample,
{
    fn process(
        &self,
        idx: f64,
        nbr_frames: usize,
        channel_mask: &[bool],
        t_ratio: f64,
        t_ratio_increment: f64,
        wave_in: &[Vec<T>],
        wave_out: &mut dyn AdapterMut<'_, T>,
        output_offset: usize,
    ) -> f64 {
        let mut t_ratio = t_ratio;
        let mut idx = idx;
        let interpolator_len = self.interpolator.nbr_points();
        match self.interpolation {
            SincInterpolationType::Cubic => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 4];
                let mut nearest = [(0isize, 0isize); 4];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_4(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out.write_sample(
                                chan,
                                frame + output_offset,
                                &interp_cubic(frac_offset, &points),
                            );
                        }
                    }
                }
            }
            SincInterpolationType::Quadratic => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 3];
                let mut nearest = [(0isize, 0isize); 3];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_3(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out.write_sample(
                                chan,
                                frame + output_offset,
                                &interp_quad(frac_offset, &points),
                            );
                        }
                    }
                }
            }
            SincInterpolationType::Linear => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut points = [T::zero(); 2];
                let mut nearest = [(0isize, 0isize); 2];
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    get_nearest_times_2(idx, oversampling_factor as isize, &mut nearest);
                    let frac = idx * oversampling_factor as f64
                        - (idx * oversampling_factor as f64).floor();
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            for (n, p) in nearest.iter().zip(points.iter_mut()) {
                                *p = self.interpolator.get_sinc_interpolated(
                                    buf,
                                    (n.0 + 2 * interpolator_len as isize) as usize,
                                    n.1 as usize,
                                );
                            }
                            wave_out.write_sample(
                                chan,
                                frame + output_offset,
                                &interp_lin(frac_offset, &points),
                            );
                        }
                    }
                }
            }
            SincInterpolationType::Nearest => {
                let oversampling_factor = self.interpolator.nbr_sincs();
                let mut point;
                let mut nearest;
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    nearest = get_nearest_time(idx, oversampling_factor as isize);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            let buf = &wave_in[chan];
                            point = self.interpolator.get_sinc_interpolated(
                                buf,
                                (nearest.0 + 2 * interpolator_len as isize) as usize,
                                nearest.1 as usize,
                            );
                            wave_out.write_sample(chan, frame + output_offset, &point);
                        }
                    }
                }
            }
        }
        idx
    }

    fn nbr_points(&self) -> usize {
        self.interpolator.nbr_points()
    }

    fn init_last_index(&self) -> f64 {
        -(self.interpolator.nbr_points() as f64 - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{interp_cubic, interp_lin};

    #[test]
    fn int_cubic() {
        let yvals = [0.0f64, 2.0f64, 4.0f64, 6.0f64];
        let interp = interp_cubic(0.5f64, &yvals);
        assert_eq!(interp, 3.0f64);
    }

    #[test]
    fn int_lin_32() {
        let yvals = [1.0f32, 5.0f32];
        let interp = interp_lin(0.25f32, &yvals);
        assert_eq!(interp, 2.0f32);
    }

    #[test]
    fn int_cubic_32() {
        let yvals = [0.0f32, 2.0f32, 4.0f32, 6.0f32];
        let interp = interp_cubic(0.5f32, &yvals);
        assert_eq!(interp, 3.0f32);
    }

    #[test]
    fn int_lin() {
        let yvals = [1.0f64, 5.0f64];
        let interp = interp_lin(0.25f64, &yvals);
        assert_eq!(interp, 2.0f64);
    }
}
