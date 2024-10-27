extern crate rubato;
use audioadapter::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Indexing, PolynomialDegree, Resampler};
use std::convert::TryInto;
use std::env;
use std::fs::File;
use std::io::prelude::{Read, Seek, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;

extern crate env_logger;
extern crate log;
use env_logger::Builder;
use log::LevelFilter;

const BYTE_PER_SAMPLE: usize = std::mem::size_of::<f64>();

// A resampler app that reads a raw file of little-endian 64 bit floats, and writes the output in the same format.
// The command line arguments are input filename, output filename, input samplerate, output samplerate,
// number of channels, final relative ratio in percent, and ramp duration in seconds.
// To resample the file `sine_f64_2ch.raw` from 44.1kHz to 192kHz, and assuming the file has two channels,
// and that the resampling ratio should be ramped to 150% during 3 seconds, the command is:
// ```
// cargo run --release --example polyfixedin_ramp64 sine_f64_2ch.raw test.raw 44100 192000 2 150 3
// ```
// There are two helper python scripts for testing.
//  - `makesineraw.py` to generate test files in raw format.
//    Run it with the `-h` flag for instructions.
//  - `analyze_result.py` to analyze the result.
//    This takes three arguments: number of channels, samplerate, and sample format.
//    Example, to analyze the file created above:
//    ```
//    python examples/analyze_result.py test.raw 2 192000 f64
//    ```

/// Helper to read an entire file to memory as f64 values
fn read_file<R: Read + Seek>(inbuffer: &mut R) -> Vec<f64> {
    let mut buffer = vec![0u8; BYTE_PER_SAMPLE];
    let mut data = Vec::new();
    loop {
        let bytes_read = inbuffer.read(&mut buffer).unwrap();
        if bytes_read == 0 {
            break;
        }
        let value = f64::from_le_bytes(buffer.as_slice().try_into().unwrap());
        data.push(value);
    }
    data
}

/// Helper to write all frames to a file
fn write_file<W: Write + Seek>(data: &[f64], output: &mut W) {
    for value in data.iter() {
        let bytes = value.to_le_bytes();
        output.write_all(&bytes).unwrap();
    }
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

    let ratio_str = env::args()
        .nth(6)
        .expect("Please specify final resampling ratio in percent");
    let final_ratio = ratio_str.parse::<f64>().unwrap();

    let duration_str = env::args()
        .nth(7)
        .expect("Please specify ramp time in seconds");
    let duration = duration_str.parse::<f64>().unwrap();

    println!("Copy input file to buffer");
    let file_in_disk = File::open(file_in).expect("Can't open file");
    let mut file_in_reader = BufReader::new(file_in_disk);
    let indata = read_file(&mut file_in_reader);
    let nbr_input_frames = indata.len() / channels;

    let f_ratio = fs_out as f64 / fs_in as f64;

    // Create buffer for storing output, size is preliminary and may grow
    let mut outdata =
        Vec::with_capacity(2 * channels * (nbr_input_frames as f64 * f_ratio) as usize);

    // parameters

    let chunksize = 1024;
    let target_ratio = final_ratio / 100.0;
    let mut resampler = Async::<f64>::new_poly(
        f_ratio,
        target_ratio,
        PolynomialDegree::Cubic,
        chunksize,
        channels,
        FixedAsync::Input,
    )
    .unwrap();

    let num_chunks = nbr_input_frames / chunksize;
    let mut output_time = 0.0;

    let input_adapter = InterleavedSlice::new(&indata, channels, nbr_input_frames).unwrap();
    let mut indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };

    let start = Instant::now();

    for chunk in 0..num_chunks {
        let input_offset = chunksize * chunk;
        indexing.input_offset = input_offset;
        let mut output_scratch = vec![0.0; channels * resampler.output_frames_next()];
        let mut output_adapter = InterleavedSlice::new_mut(
            &mut output_scratch,
            channels,
            resampler.output_frames_next(),
        )
        .unwrap();
        let (_nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .unwrap();

        outdata.append(&mut output_scratch);
        output_time += nbr_out as f64 / fs_out as f64;
        if output_time < duration {
            let rel_time = output_time / duration;
            let rel_ratio = 1.0 + (target_ratio - 1.0) * rel_time;
            println!("time {}, rel ratio {}", output_time, rel_ratio);
            resampler
                .set_resample_ratio_relative(rel_ratio, true)
                .unwrap();
        }
    }

    let duration = start.elapsed();

    println!("Resampling took: {:?}", duration);

    let mut f_out_disk = BufWriter::new(File::create(file_out).unwrap());
    write_file(&outdata, &mut f_out_disk);
}
