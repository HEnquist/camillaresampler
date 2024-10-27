extern crate rubato;
use audioadapter::number_to_float::InterleavedNumbers;
use audioadapter::sample::I16LE;

use rubato::{
    calculate_cutoff, Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Write};
use std::time::Instant;

extern crate env_logger;
extern crate log;
use env_logger::Builder;
use log::LevelFilter;
const BYTE_PER_SAMPLE: usize = std::mem::size_of::<i16>();

// A resampler app that reads a raw file of little-endian 16 bit integers, and writes the output in the same format.
// The command line arguments are resampler type, input filename, output filename, input samplerate, output samplerate, number of channels
// To use a sinc resampler with fixed input size to resample the file `sine_i16_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels, the command is:
// ```
// cargo run --release --example process_f64 SincFixedInput sine_i16_2ch.raw test.raw 44100 192000 2
// ```
// There are two helper python scripts for testing.
//  - `makesineraw.py` to generate test files in raw format.
//    Run it with the `-h` flag for instructions.
//  - `analyze_result.py` to analyze the result.
//    This takes three arguments: number of channels, samplerate, and sample format.
//    Example, to analyze the file created above:
//    ```
//    python examples/analyze_result.py test.raw 2 192000 i16
//    ```

/// Helper to read an entire file to memory as f64 values
fn read_file(filename: &str) -> Vec<u8> {
    let mut f = File::open(filename).expect("Can't open file");
    let mut data = vec![];
    f.read_to_end(&mut data).unwrap();
    data
}

/// Helper to write all frames to a file
fn write_file(filename: &str, data: &[u8], bytes_to_skip: usize, bytes_to_write: usize) {
    let mut f = File::create(filename).expect("Can't open file");
    f.write_all(&data[bytes_to_skip..bytes_to_skip + bytes_to_write])
        .expect("Failed to write data to file");
}

fn main() {
    // init logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Debug).init();

    let file_in = env::args().nth(1).expect("Please specify an input file.");
    let file_out = env::args().nth(2).expect("Please specify an output file.");
    println!("Opening files: {}, {}", file_in, file_out);

    let fs_in_str = env::args()
        .nth(3)
        .expect("Please specify an input sample rate");
    let fs_out_str = env::args()
        .nth(4)
        .expect("Please specify an output sample rate");
    let fs_in = fs_in_str.parse::<usize>().unwrap();
    let fs_out = fs_out_str.parse::<usize>().unwrap();
    println!("Resampling from {} to {}", fs_in, fs_out);

    let channels_str = env::args()
        .nth(5)
        .expect("Please specify number of channels");
    let channels = channels_str.parse::<usize>().unwrap();

    println!("Copy input file to buffer");

    let indata = read_file(&file_in);
    let nbr_input_frames = indata.len() / (channels * BYTE_PER_SAMPLE);

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create buffer for storing output
    let mut outdata: Vec<u8> =
        vec![0; 2 * channels * BYTE_PER_SAMPLE * (nbr_input_frames as f64 * f_ratio) as usize];

    println!("Creating resampler");
    let sinc_len = 128;
    let oversampling_factor = 256;
    let interpolation = SincInterpolationType::Quadratic;
    let window = WindowFunction::Blackman2;
    let f_cutoff = calculate_cutoff(sinc_len, window);
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation,
        oversampling_factor,
        window,
    };
    let mut resampler =
        Async::<f32>::new_sinc(f_ratio, 1.1, params, 1024, channels, FixedAsync::Input).unwrap();

    // Prepare
    let mut input_frames_next = resampler.input_frames_next();
    let resampler_delay = resampler.output_delay();

    let input_adapter =
        InterleavedNumbers::<&[I16LE], f32>::new_from_bytes(&indata, channels, nbr_input_frames)
            .unwrap();
    let outdata_capacity = outdata.len() / (channels * BYTE_PER_SAMPLE);
    let mut output_adapter = InterleavedNumbers::<&mut [I16LE], f32>::new_from_bytes_mut(
        &mut outdata,
        channels,
        outdata_capacity,
    )
    .unwrap();

    println!("Process all full chunks");
    let start = Instant::now();
    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };
    let mut input_frames_left = nbr_input_frames;

    while input_frames_left >= input_frames_next {
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .unwrap();

        indexing.input_offset += nbr_in;
        indexing.output_offset += nbr_out;
        input_frames_left -= nbr_in;
        input_frames_next = resampler.input_frames_next();
    }

    println!("Process a partial chunk with the last frames.");
    indexing.partial_len = Some(input_frames_left);
    let (_nbr_in, _nbr_out) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .unwrap();

    let duration = start.elapsed();
    println!("Resampling took: {:?}", duration);

    let nbr_output_frames = (nbr_input_frames as f32 * fs_out as f32 / fs_in as f32) as usize;
    println!(
        "Processed {} input frames into {} output frames",
        nbr_input_frames, nbr_output_frames
    );

    println!("Write output to file, trimming off the silent frames from both ends.");
    write_file(
        &file_out,
        &outdata,
        resampler_delay * channels * BYTE_PER_SAMPLE,
        nbr_output_frames * channels * BYTE_PER_SAMPLE,
    );
}
