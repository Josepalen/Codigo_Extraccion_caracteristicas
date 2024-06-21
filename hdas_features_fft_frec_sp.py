import numpy as np
import datetime
from scipy.signal import butter, sosfilt, spectrogram

from math import ceil
from os.path import basename
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from features.base import delta

from HDAS_File_Open import Load_2D_Data_bin


global window_duration
global overlap_duration
global spatial_point
window_duration = 4.0
overlap_duration = 0.5  # shift_duration

class HDAS:

    
    def __init__(self, dir, filenames, facx=-1, dectype='simple', verbose=False):
        # Read data from Aragon Photonics format
        # dir: base directory of the files
        # filenames: list of individual file names
        # facx: spatial decimation factor (-1=no decimation)
        # dectype: type of spatial decimation (if used):
        #          simple: stack between adjacent traces
        #          median: median between adjacent traces
        # verbose: True for debugging purposes

        # Read metadata
        [dd, header] = Load_2D_Data_bin("%s/%s" % (dir, filenames[0]))

        self.nsens = dd.shape[0]
        self.dx = header[1]
        self.srate = header[6]/header[15]/header[98]
        self.dt = 1. / self.srate

        # 012345678901234567890123456789
        # 2021_10_19_18h07m05s_HDAS_2Dmap_Strain.bin
        ff = basename(filenames[0])
        YY = int(ff[0:4])
        MM = int(ff[5:7])
        DD = int(ff[8:10])
        HH = int(ff[11:13])
        MI = int(ff[14:16])
        SS = int(ff[17:19])

        self.stime = datetime.datetime(YY,MM,DD,HH,MI,SS)

        self.fmax = self.srate / 2
        self.fmin = 0

        self.verbose = verbose

        # Merge files
        mats = []
        for i in range(len(filenames)):
            if verbose:
                print("Reading ", filenames[i])

            [dd, header] = Load_2D_Data_bin("%s/%s" % (dir, filenames[i]))
            dd = np.nan_to_num(dd)

            if facx > 0:
                dd = self._decimateX(dd, facx, dectype)

            mats.append(dd)

        self.da = np.hstack(tuple(mats))
        self.da = self.da.astype('float32')

        if facx > 0:
            self.dx = self.dx * facx
            self.nsens = self.da.shape[0]

        self.xpos = np.arange(self.nsens) * self.dx

        self.nsamp = self.da.shape[1]
        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(milliseconds=int(i * self.dt * 1000)) for i in range(self.nsamp)]
        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)

        if verbose:
            print("Original size (x,t) : ", self.da.shape)

    def _decimateX(self, dd, facx, dectype):
        # Function for internal use only
        self.tabs = [self.stime + datetime.timedelta(milliseconds=int(i * self.dt * 1000)) for i in range(self.nsamp)]

        if dectype == 'simple':
            return dd[::facx, :]

        elif dectype == 'median':
            newsens = int(self.nsens / facx)
            newdd = np.zeros((newsens, self.nsamp), dtype='float32')
            for i in range(newsens):
                ss = dd[i * facx:(i + 1) * facx, :]
                newdd[i, :] = np.median(ss, 0)

            return newdd

    def cutX(self, x1, x2):
        # Cut a spatial range
        # x1,x2: lower and upper limits (in m) of the distance along the cable

        i1 = int((x1 / self.dx))
        i2 = int((x2 / self.dx))
        self.da = self.da[i1:i2, :]
        self.nsens = self.da.shape[0]

        self.xpos = x1 + np.arange(self.nsens) * self.dx

        if self.verbose:
            print("Cut along X (x,t) : ", self.da.shape)

    def decimateX(self, facx, dectype):
        # Spatial decimation
        # facx: spatial decimation factor (-1=no decimation)
        # dectype: type of spatial decimation (if used):
        #          simple: stack between adjacent traces
        #          median: median between adjacent traces

        if dectype == 'simple:':
            self.da = self.da[::facx, :]
            self.xpos = self.xpos[::facx]

        elif dectype == 'median':
            newsens = int(self.nsens / facx)
            newda = np.zeros((newsens, self.nsamp), dtype='float32')
            for i in range(newsens):
                ss = self.da[i * facx:(i + 1) * facx, :]
                newda[i, :] = np.median(ss, 0)

            self.da = newda

        self.dx = self.dx * facx
        self.nsens = self.da.shape[0]

        if dectype == 'simple':
            self.xpos = self.xpos[0] + np.arange(self.nsens) * self.dx
        elif dectype == 'median':
            self.xpos = self.xpos[0] + np.arange(self.nsens) * self.dx + (self.dx * facx / 2)

        if self.verbose:
            print("Decimate along X (x,t) : ", self.da.shape)

    def removeTrend(self):
        # Remove linear trend along individual traces through LSQR fit

        if self.verbose:
            print("Remove trend")

        xx = np.arange(self.nsamp)
        for i in range(self.da.shape[0]):
            dr = self.da[i, :]
            try:
                po = np.polyfit(xx, dr, 1)
            except:
                print(dr)
                #plt.plot(xx,dr,'k-')
                #plt.show()
                exit(1)
            mo = np.polyval(po, xx)
            self.da[i, :] = dr - mo

    def removeCoherentNoise(self, method='median'):
        # Remove coherent synchronous noise from all the traces
        # The trace used for filtering is computed by doing the median
        # of all the individual traces
        #
        # method: filtering type:
        #         simple: subtracts the median from individual traces
        #         fit: compute the best fit correction factor befor subtracting the median

        if self.verbose:
            print("Remove coherent noise")

        md = np.median(self.da, 0)

        if method == 'simple':
            for i in range(self.da.shape[1]):
                self.da[:, i] = self.da[:, i] - md[i]

        elif method == 'fit':

            den = np.sum(md * md)
            for i in range(self.da.shape[0]):
                dd = self.da[i, :]
                am = np.sum(dd * md) / den
                self.da[i, :] = dd - am * md

    def cutT(self, t1, t2):
        # Cut a temporal range
        # t1,t2: lower and upper limits (in s) of the time window to cut

        j1 = int(t1 / self.dt)
        j2 = int(t2 / self.dt)
        self.da = self.da[:, j1:j2]

        self.nsamp = self.da.shape[1]

        self.stime = self.stime + datetime.timedelta(seconds=t1)
        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)

        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(seconds=i * self.dt) for i in range(self.nsamp)]

        if self.verbose:
            print("Cut along T (x,t) : ", self.da.shape)

    
    def len(self):
        # Returns the temporal length of the traces

        return self.dt*self.nsamp

    def decimateT(self, fact):
        # Temporal decimation
        # fact: decimation factor
        # !!! low-pass filtering below Nyquist is not perfmed here

        self.da = self.da[:, ::fact]
        self.dt = self.dt * fact
        self.srate = self.srate / fact

        self.nsamp = self.da.shape[1]

        self.etime = self.stime + datetime.timedelta(seconds=self.nsamp * self.dt)

        self.trel = np.arange(self.nsamp) * self.dt
        self.tabs = [self.stime + datetime.timedelta(seconds=i * self.dt) for i in range(self.nsamp)]

        if self.verbose:
            print("Decimate along T (x,t) : ", self.da.shape)
            print("Fmax ", self.fmax, " New Nyquist  ", self.srate / 2)

    def filter(self, f1, f2):
        # Band-pass filtering of individual traces
        # Butterworth filter with 4 poles is used
        # f1,f2: lower and upper frequency of the filter

        if self.verbose:
            print("Filter")

        sos = butter(4, [f1, f2], 'bandpass', fs=1. / self.dt, output='sos')

        for i in range(self.da.shape[0]):
            dd = self.da[i, :]
            dd = sosfilt(sos, dd)
            self.da[i, :] = dd

        self.fmax = f2
        self.fmin = f1

    def normalize(self, type='rms_c'):
        # Trace normalization
        # type, type of normalization
        #       rms: normalization of the whole DAS image by its RMS
        #       rms_c: normalization of individual traces by their RMS
        #       mad: normalization of the whole DAS image by its MAD (Median Absolute Deviation)
        #       mad_c: normalization of individual traces by their MAD

        if self.verbose:
            print("Normalize ", type)

        if type == 'rms':
            self.da = self.da / np.std(self.da)

        elif type == 'rms_c':
            rms = np.std(self.da, 1)
            for i in range(len(rms)):
                self.da[i, :] = self.da[i, :] / rms[i]

        elif type == 'mad':
            mad = np.median(np.abs(self.da))
            self.da = 0.5 * self.da / mad

        elif type == 'mad_c':
            for i in range(self.nsens):
                mad = np.median(np.abs(self.da[i, :]))
                self.da[i, :] = 0.5 * self.da[i, :] / mad

    def mute(self, perc=95):
        # Muting of noisy traces based on their RMS
        # perc: percentile of RMS above which mute (set to zero) a trace
        # 0 means all the traces muted, 100 no trace is muted

        st = np.std(self.da, axis=1)
        thr = np.percentile(st,perc)
        idx = (st>=thr)
        self.da[idx,:] = 0.

    def check(self):
        # Check the object for debugging purposes
        # writing some of the relevant paramaeters

        print(">>> CHECK HFD5DAS")
        print("NSENS ", self.nsens)
        print("NSAMP ", self.nsamp)
        print("shape ", self.da.shape)
        print("dx,len ", self.dx, self.dx * self.nsens)
        print("xpos ", self.xpos[0], self.xpos[-1])
        print("dt,len ", self.dt, self.dt * self.nsamp)
        print("srate ", self.srate)
        print("stime ", self.stime)
        print("etime ", self.etime)
        print("len ", self.etime - self.stime)
        print("trel ", self.trel[0], self.trel[-1])
        print("tabs ", self.tabs[0], self.tabs[-1])
        print("fmax ", self.fmax)

    def checkHTML(self):
        # Same as check but in HTML format

        print(">>> CHECK HFD5DAS</br>")
        print("NSENS ", self.nsens,"</br>")
        print("NSAMP ", self.nsamp,"</br>")
        print("shape ", self.da.shape,"</br>")
        print("dx,len ", self.dx, self.dx * self.nsens,"</br>")
        print("xpos ", self.xpos[0], self.xpos[-1],"</br>")
        print("dt,len ", self.dt, self.dt * self.nsamp,"</br>")
        print("srate ", self.srate,"</br>")
        print("stime ", self.stime,"</br>")
        print("etime ", self.etime,"</br>")
        print("len ", self.etime - self.stime,"</br>")
        print("trel ", self.trel[0], self.trel[-1],"</br>")
        print("tabs ", self.tabs[0], self.tabs[-1],"</br>")
        print("fmax ", self.fmax,"</br>")

    def plot(self, outfile, palette='seismic', vv=2.0, figsize=None, vel=None, dpi=1200):
        # Plot DAS data as an image
        # outfile: name of the output file (with the correct extension)
        # palette: a matplotlib compatible palette name
        # vv: palette range
        # figsize: figure size in format (X,Y)
        # vel: array of apparent velocities to plot as a reference curves

        if self.verbose:
            print("Plot", flush=True)

        gx, gy = np.meshgrid(self.tabs, self.xpos / 1000.)

        if figsize == None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)

        plt.pcolormesh(gx, gy, self.da, vmin=-vv, vmax=vv, cmap=palette, shading='auto')

        plt.ylabel('Distance (km)')
        plt.xlabel('Time (s)')
        myFmt = mdates.DateFormatter('%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(myFmt)
        # plt.xticks(rotation=90)

        # Vel
        if vel != None:
            dmax = self.nsens * self.dx
            for v in vel:
                tv = dmax / v
                t1 = self.stime + datetime.timedelta(seconds=tv)
                t2 = self.etime - datetime.timedelta(seconds=tv)
                plt.plot([self.stime, t1], [self.xpos[0], self.xpos[-1]], 'k:')
                plt.plot([self.etime, t2], [self.xpos[0], self.xpos[-1]], 'k:')

        plt.grid(axis='x', color='k', linestyle=':', linewidth=2)

        if self.verbose:
            print("Writing Image", flush=True)

        plt.savefig(outfile, dpi=dpi)
        plt.close('all')
        
    def plot_seismogram(self, outfile, sp, palette='seismic', vv=2.0, figsize=None):     
        # Plot strain-varation signal of the 1D signal in spatian point sp
        # outfile: geerated file name
        # sp: spatial point under analysis
        if figsize == None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
            plt.ylabel('Strain Variation ($\mu\epsilon$)')
            plt.xlabel('Time (s)')
            labels = [self.tabs[0], self.tabs[ceil(self.nsamp/3)], self.tabs[2*ceil(self.nsamp/3)], self.tabs[-1]]
            plt.xticks(labels)
            myFmt = mdates.DateFormatter('%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(myFmt)
            plt.plot(self.tabs, self.da[sp,:], 'k')
            plt.suptitle('Earthquake in sp = ' + str(sp*10) + ' m')
            plt.title('Bandwith: ' + str(self.fmin) + '-' + str(self.fmax) + 'Hz')
            plt.savefig(outfile + '_sp' + str(sp*10) + '_' + str(self.fmin) + 'to' + str(self.fmax) + 'Hz.png', dpi=1200)
            plt.close('all')


    def plot_spectrogram(self, outfile, sp, nfft, maxfreq, txttitle, palette='seismic', vv=2.0, figsize=None):
        # Plot frquency spectrogram of the 1D signal in spatian point sp
        # outfile: geerated file name
        # sp: spatial point under analysis
        # nfft: number of points used in the spectrogram analysis
        # maxfreq: maximum frequency of interest in the spectrogram figure (only visual purpose)
        # txttitle: event name 
        f, t, Sxx = spectrogram(self.da[sp,:], self.srate, nfft=nfft)
        freq_interest=[i for i in range(len(f)) if f[i] < maxfreq]
        plt.pcolormesh(t, f[freq_interest], Sxx[0:len(freq_interest),:] ,cmap='seismic', shading ='auto')
        plt.ylabel('Frequency (Hz)',fontsize=10)
        plt.xlabel('Time (s)',fontsize=10)
        plt.title('Spectrogram ' + str(self.fmin) + '-' + str(self.fmax) + 'Hz, sp=' + str(sp*10), fontsize=10 )   
        cbar = plt.colorbar()
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        plt.suptitle(txttitle)
        plt.savefig(outfile + '_sp' + str(sp*10) + '_' + str(self.fmin) + 'to' + str(self.fmax) + 'Hz.png', dpi=1200)
        plt.close('all')

    def plotsismo(self, outfile, sismot, sismodat, palette='seismic', vv=2.0, figsize=None, vel=None):
        # Plot DAS data as an image adding a reference seismogram
        # outfile: name of the output file (with the correct extension)
        # sismot: array of times of the seismogram
        # sismodat: array of amplitudes of the seismogram
        # palette: a matplotlib compatible palette name
        # vv: palette range
        # figsize: figure size in format (X,Y)
        # vel: array of apparent velocities to plot as a reference curves

        if self.verbose:
            print("Plot", flush=True)

        print(self.xpos[0], self.xpos[-1])
        gx, gy = np.meshgrid(self.tabs, self.xpos / 1000.)

        if figsize == None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)

        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

        a0.pcolormesh(gx, gy, self.da, vmin=-vv, vmax=vv, cmap=palette, shading='auto')

        a0.set_ylabel('Distance (km)')
        # a0.set_xlabel('Time (s)')
        # myFmt = mdates.DateFormatter('%H:%M:%S')
        # a0.xaxis.set_major_formatter(myFmt)
        # a0.xaxis.set_label_position('top')
        # plt.xticks(rotation=90)
        a0.set_xticklabels([])

        # Vel
        if vel != None:
            dmax = self.nsens * self.dx
            for v in vel:
                tv = dmax / v
                t1 = self.stime + datetime.timedelta(seconds=tv)
                t2 = self.etime - datetime.timedelta(seconds=tv)
                a0.plot([self.stime, t1], [self.xpos[0], self.xpos[-1]], 'k:')
                a0.plot([self.etime, t2], [self.xpos[0], self.xpos[-1]], 'k:')

        a0.grid(axis='x', color='k', linestyle=':', linewidth=2)

        a1.plot(sismot, sismodat, 'k')
        a1.set_xlim([sismot[0], sismot[-1]])
        a1.set_ylim([-1.1, 1.1])
        a1.set_xlabel('Time (s)')
        myFmt = mdates.DateFormatter('%H:%M:%S')
        a1.xaxis.set_major_formatter(myFmt)

        if self.verbose:
            print("Writing Image", flush=True)

        plt.savefig(outfile, dpi=1200)
        plt.close('all')

    def dump(self, filename):
        # Dump the DAS data matrix in a Numpy compressed file

        if self.verbose:
            print("Dumping data matrix ", self.da.shape)
        np.savez_compressed(filename, da=self.da)
        
        
        
 

                
                
    def fft_128bin(self, sp, fft_points=256):
        
        # Sampling frequency
        sampling_rate = self.srate
    
        # Number of points in each window
        points_per_window = int(window_duration * sampling_rate)
    
        # Number of overlapping points
        overlap_points = int(overlap_duration * sampling_rate)
    
        # Array of times for x-axis
        times_x = np.arange(0, self.nsamp - points_per_window + 1, overlap_points) * self.dt
    
        # Format for the x-axis of the graphs
        times_x = [self.stime + datetime.timedelta(seconds=t) for t in times_x]
    
        # We initialize the matrix to store FFT results.
        matrix_fft = np.zeros((len(times_x), fft_points // 2 + 1))
    
        # We initialize vectors to accumulate frequency values.
        self.val_for_fbin = [np.zeros(len(times_x)) for _ in range(128)]
    
        # Iterate over the overlapping windows and perform the Fourier transform.
        for i, index in enumerate(range(0, self.nsamp - points_per_window + 1, overlap_points)):
            # We define the start and end indexes for the current window.
            start_index = index
            end_index = start_index + points_per_window
    
            # We extract the data from the current window
            data_current_window = self.da[:, start_index:end_index]
    
            # We calculate the frequencies and the FFT of the current window.
            freq, spectrum = np.fft.fftfreq(fft_points, d=1/sampling_rate), np.fft.fft(data_current_window, n=fft_points, axis=1)
           
            # We select only the positive frequencies
            freq_positive = freq >= 0
            
            # We calculate the variations of magnitude in frequency (absolute value).
            magnitude_variations_fft = np.abs(spectrum[:, freq_positive][sp])
    
            # We make sure that magnitude_variations_fft has the right dimension
            magnitude_variations_fft = np.pad(magnitude_variations_fft, (0, matrix_fft.shape[1] - len(magnitude_variations_fft)))
    
            # We update accumulated vectors for the 10 selected bins.
            for bin_index in range(0,128):
                self.val_for_fbin[bin_index][i] = magnitude_variations_fft[bin_index]
    
        # Calculate the width of each bin in terms of frequency.  
        width_bin = float(50 / 127)
        
        
        # Convert lists to numpy arrays    
        self.val_for_fbin = np.array(self.val_for_fbin)
        
        # Calcular las derivadas
        self.deltas_fft = delta(self.val_for_fbin, 2)
        self.deltas_deltas_fft = delta(self.deltas_fft, 2)        
        
        
        # # Plot de val_for_fbin, deltas_fft y deltas_deltas_fft para el primer punto espacial
        # plt.figure(figsize=(10, 6))
        
        # # Plot de val_for_fbin
        # plt.plot(times_x, self.val_for_fbin[:, 0], label='val_for_fbin', color='blue')
        
        # # Plot de deltas_fft
        # plt.plot(times_x, self.deltas_fft[:, 0], label='deltas_fft', color='red')
        
        # # Plot de deltas_deltas_fft
        # plt.plot(times_x, self.deltas_deltas_fft[:, 0], label='deltas_deltas_fft', color='green')
        
        # plt.xlabel('Tiempo')
        # plt.ylabel('Magnitud')
        # plt.title('Val_for_fbin, Deltas_fft y Deltas_deltas_fft para el primer punto espacial')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
   
        