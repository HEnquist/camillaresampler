use crate::error::{CpuFeature, MissingCpuFeature};
use crate::sinc::make_sincs;
use crate::sinc_interpolator::SincInterpolator;
use crate::windows::WindowFunction;
use crate::Sample;
use core::arch::aarch64::{float32x4_t, float64x2_t};
use core::arch::aarch64::{
    vadd_f32, vaddq_f32, vfmaq_f32, vget_high_f32, vget_low_f32, vld1q_f32, vmovq_n_f32, vst1_f32,
};
use core::arch::aarch64::{vaddq_f64, vfmaq_f64, vld1q_f64, vmovq_n_f64, vst1q_f64};

/// Collection of cpu features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Neon];

/// Trait governing what can be done with an NeonSample.
pub trait NeonSample: Sized + Send {
    type Sinc: Send;

    /// Pack sincs into a vector.
    ///
    /// # Safety
    ///
    /// This is unsafe because it uses target_enable dispatching. There are no
    /// special requirements from the caller.
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>>;

    /// Interpolate a sinc sample.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the various indexes are not out of bounds
    /// in the collection of sincs.
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[Self],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> Self;
}

impl NeonSample for f32 {
    type Sinc = float32x4_t;

    #[target_feature(enable = "neon")]
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                let packed_elems = vld1q_f32(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f32],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f32 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc0 = vmovq_n_f32(0.0);
        let mut acc1 = vmovq_n_f32(0.0);
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = vld1q_f32(wave_cut.get_unchecked(w_idx));
            let w1 = vld1q_f32(wave_cut.get_unchecked(w_idx + 4));
            acc0 = vfmaq_f32(acc0, w0, *sinc.get_unchecked(s_idx));
            acc1 = vfmaq_f32(acc1, w1, *sinc.get_unchecked(s_idx + 1));
            w_idx += 8;
            s_idx += 2;
        }
        let sum4 = vaddq_f32(acc0, acc1);
        let high = vget_high_f32(sum4);
        let low = vget_low_f32(sum4);
        let sum2 = vadd_f32(high, low);
        let mut array = [0.0, 0.0];
        vst1_f32(array.as_mut_ptr(), sum2);
        array[0] + array[1]
    }
}

impl NeonSample for f64 {
    type Sinc = float64x2_t;

    #[target_feature(enable = "neon")]
    unsafe fn pack_sincs(sincs: Vec<Vec<f64>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(2) {
                let packed_elems = vld1q_f64(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "neon")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f64],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f64 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc0 = vmovq_n_f64(0.0);
        let mut acc1 = vmovq_n_f64(0.0);
        let mut acc2 = vmovq_n_f64(0.0);
        let mut acc3 = vmovq_n_f64(0.0);
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = vld1q_f64(wave_cut.get_unchecked(w_idx));
            let w1 = vld1q_f64(wave_cut.get_unchecked(w_idx + 2));
            let w2 = vld1q_f64(wave_cut.get_unchecked(w_idx + 4));
            let w3 = vld1q_f64(wave_cut.get_unchecked(w_idx + 6));
            acc0 = vfmaq_f64(acc0, w0, *sinc.get_unchecked(s_idx));
            acc1 = vfmaq_f64(acc1, w1, *sinc.get_unchecked(s_idx + 1));
            acc2 = vfmaq_f64(acc2, w2, *sinc.get_unchecked(s_idx + 2));
            acc3 = vfmaq_f64(acc3, w3, *sinc.get_unchecked(s_idx + 3));
            w_idx += 8;
            s_idx += 4;
        }
        let packedsum0 = vaddq_f64(acc0, acc1);
        let packedsum1 = vaddq_f64(acc2, acc3);
        let packedsum2 = vaddq_f64(packedsum0, packedsum1);
        let mut values = [0.0, 0.0];
        vst1q_f64(values.as_mut_ptr(), packedsum2);
        values[0] + values[1]
    }
}

/// A SSE accelerated interpolator.
pub struct NeonInterpolator<T>
where
    T: NeonSample,
{
    sincs: Vec<Vec<T::Sinc>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for NeonInterpolator<T>
where
    T: Sample,
{
    /// Calculate the scalar produt of an input wave and the selected sinc filter.
    fn get_sinc_interpolated(&self, wave: &[T], index: usize, subindex: usize) -> T {
        assert!(
            (index + self.length) < wave.len(),
            "Tried to interpolate for index {}, max for the given input is {}",
            index,
            wave.len() - self.length - 1
        );
        assert!(
            subindex < self.nbr_sincs,
            "Tried to use sinc subindex {}, max is {}",
            subindex,
            self.nbr_sincs - 1
        );
        unsafe { T::get_sinc_interpolated_unsafe(wave, index, subindex, &self.sincs, self.length) }
    }

    fn nbr_points(&self) -> usize {
        self.length
    }

    fn nbr_sincs(&self) -> usize {
        self.nbr_sincs
    }
}

impl<T> NeonInterpolator<T>
where
    T: Sample,
{
    /// Create a new NeonInterpolator.
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
    ) -> Result<Self, MissingCpuFeature> {
        if let Some(feature) = FEATURES.iter().find(|f| !f.is_detected()) {
            return Err(MissingCpuFeature(*feature));
        }

        assert!(sinc_len % 8 == 0, "Sinc length must be a multiple of 8.");
        let sincs = make_sincs(sinc_len, oversampling_factor, f_cutoff, window);
        let sincs = unsafe { <T as NeonSample>::pack_sincs(sincs) };

        Ok(Self {
            sincs,
            length: sinc_len,
            nbr_sincs: oversampling_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::sinc::make_sincs;
    use crate::sinc_interpolator::sinc_interpolator_neon::NeonInterpolator;
    use crate::sinc_interpolator::SincInterpolator;
    use crate::WindowFunction;
    use num_traits::Float;
    use rand::Rng;
    use test_log::test;

    fn get_sinc_interpolated<T: Float>(wave: &[T], index: usize, sinc: &[T]) -> T {
        let wave_cut = &wave[index..(index + sinc.len())];
        wave_cut
            .iter()
            .zip(sinc.iter())
            .fold(T::zero(), |acc, (x, y)| acc + *x * *y)
    }

    #[test]
    fn test_neon_interpolator_64() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f64>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f64>(sinc_len, oversampling_factor, f_cutoff, window);
        let interpolator =
            NeonInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_neon_interpolator_32() {
        let mut rng = rand::thread_rng();
        let mut wave = Vec::new();
        for _ in 0..2048 {
            wave.push(rng.gen::<f32>());
        }
        let sinc_len = 256;
        let f_cutoff = 0.9473371669037001;
        let oversampling_factor = 256;
        let window = WindowFunction::BlackmanHarris2;
        let sincs = make_sincs::<f32>(sinc_len, oversampling_factor, f_cutoff, window);
        let interpolator =
            NeonInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-5);
    }
}
