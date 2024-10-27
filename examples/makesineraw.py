# Make a sine for testing purposes
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("samplerate", type=int,
                    help="sample rate of file")
parser.add_argument("channels", type=int,
                    help="number of channels")
parser.add_argument("length", type=float,
                    help="length in seconds")
parser.add_argument("frequency", type=float,
                    help="sine frequency")

args = parser.parse_args()

t = np.linspace(0, args.length, num=int(args.length*args.samplerate), endpoint=False)
wave = np.sin(args.frequency*2*np.pi*t)
wave= np.reshape(wave,(-1,1))
wave = np.concatenate((wave,)*args.channels, axis=1)

wave64 = wave.astype('float64')
wave32 = wave.astype('float32')

# integer version, scale to 50% of max amplitude
scaled_i16 = wave * 2**14
wavei16 = scaled_i16.astype('int16')
srkhz = args.samplerate/1000
fkhz = args.frequency/1000

fname_template = f"sine_{srkhz:.1f}kHz_{fkhz:.1f}kHz_{args.channels}ch_{args.length:.1f}s_{{}}.raw"

for w, f in zip([wave64, wave32, wavei16], ["f64", "f32", "i16"]):
    fname = fname_template.format(f)
    print("Saving:", fname)
    w.tofile(fname)



