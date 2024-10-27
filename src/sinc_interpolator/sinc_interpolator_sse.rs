use crate::error::{CpuFeature, MissingCpuFeature};
use crate::sinc::make_sincs;
use crate::sinc_interpolator::SincInterpolator;
use crate::windows::WindowFunction;
use crate::Sample;
use core::arch::x86_64::{__m128, __m128d};
use core::arch::x86_64::{
    _mm_add_pd, _mm_hadd_pd, _mm_loadu_pd, _mm_mul_pd, _mm_setzero_pd, _mm_store_sd,
};
use core::arch::x86_64::{
    _mm_add_ps, _mm_hadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_store_ss,
};

/// Collection of cpu features required for this interpolator.
static FEATURES: &[CpuFeature] = &[CpuFeature::Sse3];

/// Trait governing what can be done with an SseSample.
pub trait SseSample: Sized + Send {
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

impl SseSample for f32 {
    type Sinc = __m128;

    #[target_feature(enable = "sse3")]
    unsafe fn pack_sincs(sincs: Vec<Vec<Self>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(4) {
                let packed_elems = _mm_loadu_ps(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f32],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f32 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx));
            let w1 = _mm_loadu_ps(wave_cut.get_unchecked(w_idx + 4));
            let s0 = _mm_mul_ps(w0, *sinc.get_unchecked(s_idx));
            let s1 = _mm_mul_ps(w1, *sinc.get_unchecked(s_idx + 1));
            acc0 = _mm_add_ps(acc0, s0);
            acc1 = _mm_add_ps(acc1, s1);
            w_idx += 8;
            s_idx += 2;
        }
        let temp4 = _mm_add_ps(acc0, acc1);
        let temp2 = _mm_hadd_ps(temp4, temp4);
        let temp1 = _mm_hadd_ps(temp2, temp2);
        let mut result = 0.0;
        _mm_store_ss(&mut result, temp1);
        result
    }
}

impl SseSample for f64 {
    type Sinc = __m128d;

    #[target_feature(enable = "sse3")]
    unsafe fn pack_sincs(sincs: Vec<Vec<f64>>) -> Vec<Vec<Self::Sinc>> {
        let mut packed_sincs = Vec::new();
        for sinc in sincs.iter() {
            let mut packed = Vec::new();
            for elements in sinc.chunks(2) {
                let packed_elems = _mm_loadu_pd(&elements[0]);
                packed.push(packed_elems);
            }
            packed_sincs.push(packed);
        }
        packed_sincs
    }

    #[target_feature(enable = "sse3")]
    unsafe fn get_sinc_interpolated_unsafe(
        wave: &[f64],
        index: usize,
        subindex: usize,
        sincs: &[Vec<Self::Sinc>],
        length: usize,
    ) -> f64 {
        let sinc = sincs.get_unchecked(subindex);
        let wave_cut = &wave[index..(index + length)];
        let mut acc0 = _mm_setzero_pd();
        let mut acc1 = _mm_setzero_pd();
        let mut acc2 = _mm_setzero_pd();
        let mut acc3 = _mm_setzero_pd();
        let mut w_idx = 0;
        let mut s_idx = 0;
        for _ in 0..wave_cut.len() / 8 {
            let w0 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx));
            let w1 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 2));
            let w2 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 4));
            let w3 = _mm_loadu_pd(wave_cut.get_unchecked(w_idx + 6));
            let s0 = _mm_mul_pd(w0, *sinc.get_unchecked(s_idx));
            let s1 = _mm_mul_pd(w1, *sinc.get_unchecked(s_idx + 1));
            let s2 = _mm_mul_pd(w2, *sinc.get_unchecked(s_idx + 2));
            let s3 = _mm_mul_pd(w3, *sinc.get_unchecked(s_idx + 3));
            acc0 = _mm_add_pd(acc0, s0);
            acc1 = _mm_add_pd(acc1, s1);
            acc2 = _mm_add_pd(acc2, s2);
            acc3 = _mm_add_pd(acc3, s3);
            w_idx += 8;
            s_idx += 4;
        }
        let temp2_0 = _mm_add_pd(acc0, acc1);
        let temp2_1 = _mm_add_pd(acc2, acc3);
        let temp2 = _mm_hadd_pd(temp2_0, temp2_1);
        let temp1 = _mm_hadd_pd(temp2, temp2);
        let mut result = 0.0;
        _mm_store_sd(&mut result, temp1);
        result
    }
}

/// A SSE accelerated interpolator.
pub struct SseInterpolator<T>
where
    T: SseSample,
{
    sincs: Vec<Vec<T::Sinc>>,
    length: usize,
    nbr_sincs: usize,
}

impl<T> SincInterpolator<T> for SseInterpolator<T>
where
    T: SseSample,
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

impl<T> SseInterpolator<T>
where
    T: Sample,
{
    /// Create a new SseInterpolator.
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
        let sincs = unsafe { <T as SseSample>::pack_sincs(sincs) };

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
    use crate::sinc_interpolator::sinc_interpolator_sse::SseInterpolator;
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
    fn test_sse_interpolator_64() {
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
            SseInterpolator::<f64>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-9);
    }

    #[test]
    fn test_sse_interpolator_32() {
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
            SseInterpolator::<f32>::new(sinc_len, oversampling_factor, f_cutoff, window).unwrap();
        let value = interpolator.get_sinc_interpolated(&wave, 333, 123);
        let check = get_sinc_interpolated(&wave, 333, &sincs[123]);
        assert!((value - check).abs() < 1.0e-5);
    }
}
