use crate::asynchro::InnerResampler;
use crate::Sample;
use audioadapter::AdapterMut;
use std::marker::PhantomData;

macro_rules! t {
    // Shorter form of T::coerce(value)
    ($expression:expr) => {
        T::coerce($expression)
    };
}

/// Degree of the polynomial used for interpolation.
/// A higher degree gives a higher quality result, while taking longer to compute.
#[derive(Debug)]
pub enum PolynomialDegree {
    /// Septic polynomial, fitted using 8 sample points.
    Septic,
    /// Quintic polynomial, fitted using 6 sample points.
    Quintic,
    /// Cubic polynomial, fitted using 4 sample points.
    Cubic,
    /// Linear polynomial, fitted using 2 sample points.
    Linear,
    /// Nearest, uses the nearest sample point without any fitting.
    Nearest,
}

impl PolynomialDegree {
    pub fn nbr_points(&self) -> usize {
        match self {
            PolynomialDegree::Nearest => 1,
            PolynomialDegree::Linear => 2,
            PolynomialDegree::Cubic => 4,
            PolynomialDegree::Quintic => 6,
            PolynomialDegree::Septic => 8,
        }
    }
}

/// Perform septic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -3, -2, -1, 0, 1, 2, 3, 4.
pub fn interp_septic<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    let a = yvals[0];
    let b = yvals[1];
    let c = yvals[2];
    let d = yvals[3];
    let e = yvals[4];
    let f = yvals[5];
    let g = yvals[6];
    let h = yvals[7];
    let k7 = -a + t!(7.0) * b - t!(21.0) * c + t!(35.0) * d - t!(35.0) * e + t!(21.0) * f
        - t!(7.0) * g
        + h;
    let k6 = t!(7.0) * a - t!(42.0) * b + t!(105.0) * c - t!(140.0) * d + t!(105.0) * e
        - t!(42.0) * f
        + t!(7.0) * g;
    let k5 = -t!(7.0) * a - t!(14.0) * b + t!(189.0) * c - t!(490.0) * d + t!(595.0) * e
        - t!(378.0) * f
        + t!(119.0) * g
        - t!(14.0) * h;
    let k4 = -t!(35.0) * a + t!(420.0) * b - t!(1365.0) * c + t!(1960.0) * d - t!(1365.0) * e
        + t!(420.0) * f
        - t!(35.0) * g;
    let k3 = t!(56.0) * a - t!(497.0) * b + t!(336.0) * c + t!(1715.0) * d - t!(3080.0) * e
        + t!(1869.0) * f
        - t!(448.0) * g
        + t!(49.0) * h;
    let k2 = t!(28.0) * a - t!(378.0) * b + t!(3780.0) * c - t!(6860.0) * d + t!(3780.0) * e
        - t!(378.0) * f
        + t!(28.0) * g;
    let k1 = -t!(48.0) * a + t!(504.0) * b - t!(3024.0) * c - t!(1260.0) * d + t!(5040.0) * e
        - t!(1512.0) * f
        + t!(336.0) * g
        - t!(36.0) * h;
    let k0 = t!(5040.0) * d;
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x2 * x3;
    let x6 = x3 * x3;
    let x7 = x3 * x4;
    let val = k7 * x7 + k6 * x6 + k5 * x5 + k4 * x4 + k3 * x3 + k2 * x2 + k1 * x + k0;
    t!(1.0 / 5040.0) * val
}

/// Perform quintic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -2, -1, 0, 1, 2, 3.
pub fn interp_quintic<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    let a = yvals[0];
    let b = yvals[1];
    let c = yvals[2];
    let d = yvals[3];
    let e = yvals[4];
    let f = yvals[5];
    let k5 = -a + t!(5.0) * b - t!(10.0) * c + t!(10.0) * d - t!(5.0) * e + f;
    let k4 = t!(5.0) * a - t!(20.0) * b + t!(30.0) * c - t!(20.0) * d + t!(5.0) * e;
    let k3 = -t!(5.0) * a - t!(5.0) * b + t!(50.0) * c - t!(70.0) * d + t!(35.0) * e - t!(5.0) * f;
    let k2 = -t!(5.0) * a + t!(80.0) * b - t!(150.0) * c + t!(80.0) * d - t!(5.0) * e;
    let k1 = t!(6.0) * a - t!(60.0) * b - t!(40.0) * c + t!(120.0) * d - t!(30.0) * e + t!(4.0) * f;
    let k0 = t!(120.0) * c;
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x2 * x3;
    let val = k5 * x5 + k4 * x4 + k3 * x3 + k2 * x2 + k1 * x + k0;
    t!(1.0 / 120.0) * val
}

/// Perform cubic polynomial interpolation to get value at x.
/// Input points are assumed to be at x = -1, 0, 1, 2.
pub fn interp_cubic<T>(x: T, yvals: &[T]) -> T
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

/// Linear interpolation between two points at x=0 and x=1.
pub fn interp_lin<T>(x: T, yvals: &[T]) -> T
where
    T: Sample,
{
    yvals[0] + x * (yvals[1] - yvals[0])
}

pub struct InnerPoly<T> {
    pub _phantom: PhantomData<T>,
    pub interpolation: PolynomialDegree,
}

impl<T> InnerResampler<T> for InnerPoly<T>
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
        let interpolator_len = self.nbr_points();
        let mut t_ratio = t_ratio;
        let mut idx = idx;
        match self.interpolation {
            PolynomialDegree::Septic => {
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 3;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 8) as usize,
                                );
                                wave_out.write_sample_unchecked(
                                    chan,
                                    frame + output_offset,
                                    &interp_septic(frac_offset, buf),
                                );
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Quintic => {
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 2;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 6) as usize,
                                );
                                wave_out.write_sample_unchecked(
                                    chan,
                                    frame + output_offset,
                                    &interp_quintic(frac_offset, buf),
                                );
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Cubic => {
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize - 1;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 4) as usize,
                                );
                                wave_out.write_sample_unchecked(
                                    chan,
                                    frame + output_offset,
                                    &interp_cubic(frac_offset, buf),
                                );
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Linear => {
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let idx_floor = idx.floor();
                    let start_idx = idx_floor as isize;
                    let frac = idx - idx_floor;
                    let frac_offset = t!(frac);
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let buf = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize
                                        ..(start_idx + 2 * interpolator_len as isize + 2) as usize,
                                );
                                wave_out.write_sample_unchecked(
                                    chan,
                                    frame + output_offset,
                                    &interp_lin(frac_offset, buf),
                                );
                            }
                        }
                    }
                }
            }
            PolynomialDegree::Nearest => {
                for frame in 0..nbr_frames {
                    t_ratio += t_ratio_increment;
                    idx += t_ratio;
                    let start_idx = idx.floor() as isize;
                    for (chan, active) in channel_mask.iter().enumerate() {
                        if *active {
                            unsafe {
                                let point = wave_in.get_unchecked(chan).get_unchecked(
                                    (start_idx + 2 * interpolator_len as isize) as usize,
                                );
                                wave_out.write_sample_unchecked(chan, frame + output_offset, point);
                            }
                        }
                    }
                }
            }
        }
        idx
    }

    fn nbr_points(&self) -> usize {
        self.interpolation.nbr_points()
    }

    fn init_last_index(&self) -> f64 {
        -(self.interpolation.nbr_points() as f64 / 2.0)
    }
}
