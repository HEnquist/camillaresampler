

import numpy as np
import numpy.fft as fft
import sys
from matplotlib import pyplot as plt
import math
import argparse

def blackman_harris(npoints):
    x=np.arange(0,npoints)
    y= 0.35875 - 0.48829*np.cos(2*np.pi*x/npoints) + 0.14128*np.cos(4*np.pi*x/npoints) - 0.01168*np.cos(6*np.pi*x/npoints)
    return y

def plot_spect(indata, window=True):
    
    for wf, fs in indata:
        print(sum(wf))
        npoints = len(wf)
        divfact = npoints/2
        if window:
            wind = blackman_harris(npoints)
            wf = wf*wind
            divfact = sum(wind)/2
        print(npoints)
        t = np.linspace(0, npoints/fs, npoints, endpoint=False) 
        f = np.linspace(0, fs/2.0, math.floor(npoints/2))
        valfft = fft.fft(wf)
        cut = valfft[0:math.floor(npoints/2)]
        ampl = 20*np.log10(np.abs(cut)/divfact)
        phase = 180/np.pi*np.angle(cut)
        #plt.subplot(2,1,1)
        plt.figure(1)
        plt.plot(f, ampl)
        #plt.subplot(2,1,2)
        #plt.semilogx(f, phase)

        #plt.gca().set(xlim=(10, srate/2.0))
        #plt.subplot(2,1,2)
        plt.figure(2)
        plt.plot(t, wf)
        plt.figure(3)
        plt.plot(t[0:-5], np.diff(wf,5))
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("file_in", type=str,
                    help="input filename")
parser.add_argument("channels", type=int,
                    help="number of channels")
parser.add_argument("samplerate", type=int,
                    help="sample rate of file")
parser.add_argument("format", type=str, choices=["f32", "f64", "i16"],
                    help="sample format")
parser.add_argument("--no-window", dest= 'window', default=True, action='store_false',
                    help="skip applying window function")
args = parser.parse_args()

if args.format == "f64":
    values = np.fromfile(args.file_in, dtype=float)
elif args.format == "f32":
    values = np.fromfile(args.file_in, dtype=np.float32)
elif args.format == "i16":
    values = np.fromfile(args.file_in, dtype=np.int16).astype(np.float32) / 2**15 


# Let's look at the first channel only..
values = values.reshape((-1,args.channels))
values = values[:,0]
plot_spect([(values, args.samplerate)], window=args.window)



