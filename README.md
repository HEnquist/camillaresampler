# rubato

An audio sample rate conversion library for Rust.

This library provides resamplers to process audio in chunks.

The ratio between input and output sample rates is completely free.
Implementations are available that accept a fixed length input
while returning a variable length output, and vice versa.

### Asynchronous resampling
The resampling is based on band-limited interpolation using sinc
interpolation filters. The sinc interpolation upsamples by an adjustable factor,
and then the new sample points are calculated by interpolating between these points.
The resampling ratio can be updated at any time.

### Synchronous resampling
Synchronous resampling is implemented via FFT. The data is FFT:ed, the spectrum modified,
and then inverse FFT:ed to get the resampled data.
This type of resampler is considerably faster but doesn't support changing the resampling ratio.

### Features
#### `avx`: AVX on x86_64
The asynchronous resampler will always use SSE3 if available. This gives a speedup of about
2x for 64-bit float data, and 3-4x for 32-bit. With the `avx` feature (enabled by default)
it will check for AVX and use that if available. Depending on the cpu, this may give up to 1.5x the speed of SSE.
On other architechtures than x86_64 the `avx` feature will do nothing.

#### `neon`: Experimental Neon support on aarch64
Experimental support for Neon is available for aarch64 (64-bit Arm) by enabling the `neon` feature.
This requires the use of a nightly compiler, as the Neon support in Rust is still experimental.
On a Raspberry Pi 4, this gives a boost of 1.7x for 64-bit floats and 2.4x for 32-bit floats.
Note that this only works on a full 64-bit operating system.

### Documentation

The full documentation can be generated by rustdoc. To generate and view it run:
```
cargo doc --open
```

### Example
Resample a single chunk of a dummy audio file from 44100 to 48000 Hz.
See also the "fixedin64" example that can be used to process a file from disk.
```rust
use rubato::{Resampler, SincFixedIn, InterpolationType, InterpolationParameters, WindowFunction};
let params = InterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: InterpolationType::Nearest,
    oversampling_factor: 160,
    window: WindowFunction::BlackmanHarris2,
};
let mut resampler = SincFixedIn::<f64>::new(
    48000 as f64 / 44100 as f64,
    params,
    1024,
    2,
);

let waves_in = vec![vec![0.0f64; 1024];2];
let waves_out = resampler.process(&waves_in).unwrap();
```

### Compatibility

The `rubato` crate requires rustc version 1.40 or newer.

License: MIT
