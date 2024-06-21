import numpy as np
import datetime
from scipy.signal import butter, sosfilt, spectrogram
from math import ceil
from os.path import basename
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import antropy as ant
from scipy.signal import periodogram, welch
import librosa.display
import os
from features.base import logfbank 
from features.base import delta
from scipy import signal



from HDAS_File_Open import Load_2D_Data_bin

global window_duration
global overlap_duration
window_duration = 4.0
overlap_duration = 0.5

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
        existing_files = []  # Lista para almacenar la existencia de los archivos

        for i in range(len(filenames)):
            try:
                if verbose:
                    print("Reading ", filenames[i])

                [dd, header] = Load_2D_Data_bin("%s/%s" % (dir, filenames[i]))
                dd = np.nan_to_num(dd)

                if facx > 0:
                    dd = self._decimateX(dd, facx, dectype)

                mats.append(dd)
                existing_files.append('1')  # Indicar que el archivo existe

            except FileNotFoundError:
                print(f"El archivo '{filenames[i]}' no se encontró. Se omitirá.")
                existing_files.append('0')  # Indicar que el archivo no existe

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

        # Añadir la lista de existencia de archivos al diccionario
        self.existing_files = existing_files

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
        
        
        
 

                
                
    def fft_10bin(self, fft_points=128):
        
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
        val_for_fbin = [np.zeros(len(times_x)) for _ in range(10)]
    
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
            magnitude_variations_fft = np.abs(spectrum[:, freq_positive][0])
    
            # We make sure that magnitude_variations_fft has the right dimension
            magnitude_variations_fft = np.pad(magnitude_variations_fft, (0, matrix_fft.shape[1] - len(magnitude_variations_fft)))
    
            # We update accumulated vectors for the 10 selected bins.
            for bin_index in range(2, 12):
                val_for_fbin[bin_index - 2][i] = magnitude_variations_fft[bin_index]
    
        # Calculate the width of each bin in terms of frequency.  
        width_bin = float(50 / 63)
        
        # We plot the 10 vectors in separate graphs.

        for bin_index, values_over_time in enumerate(val_for_fbin):
            start_freq = (bin_index + 2) * width_bin
            end_freq = (bin_index + 3) * width_bin
            plt.figure(figsize=(4, 3))
            plt.plot(times_x, values_over_time)
            plt.title(f'Frecuencia Bin {bin_index + 2} - {start_freq:.3f} Hz a {end_freq:.3f} Hz\nVariación a lo largo del tiempo')
            plt.xlabel('Tiempo')
            plt.ylabel('Magnitud')
    
            # We customize the x-axis format
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
            plt.grid()
            plt.show()
    
        # We plot the 10 vectors on the same plot
        plt.figure(figsize=(10, 6))
        for bin_index, values_over_time in enumerate(val_for_fbin):
            start_freq = (bin_index + 2) * width_bin
            end_freq = (bin_index + 3) * width_bin
            plt.plot(times_x, values_over_time, label=f'Frecuencia Bin {bin_index + 2} - {start_freq:.3f} Hz a {end_freq:.3f} Hz')
        plt.title('Variación a lo largo del tiempo para Bins Seleccionados')
        plt.xlabel('Tiempo')
        plt.ylabel('Magnitud')
    
        # We customize the x-axis format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
        plt.legend()
        plt.grid()
        plt.show()
        
        
        
        
        
    def calculate_hjorth_parameters(self, sp):
        
        # Calculate the number of points in each window and overlapping
        points_per_window = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_window + 1, overlap_points) * self.dt
        
        # Format for the x-axis of the graphs
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
        

    
        # Initialize vectors to accumulate the Hjorth parameters
        self.mobility_vals, self.complexity_vals, self.variance_vals = [], [], []
    
        # Iterate over the windows and calculate the Hjorth parameters.
        for i, index in enumerate(range(0, self.nsamp - points_per_window + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_window
            data_current_window = self.da[:, start_index:end_index]
    
            mobility, complexity = ant.hjorth_params(data_current_window[sp])

            # Calculate the variance
            variance = np.var(data_current_window[sp])
            
            
            # Store parameters in lists
            self.mobility_vals.append([mobility])
            self.complexity_vals.append([complexity])
            self.variance_vals.append([variance])

        # Convert lists to numpy arrays
        self.mobility_vals = np.array(self.mobility_vals)
        self.complexity_vals = np.array(self.complexity_vals)
        self.variance_vals = np.array(self.variance_vals)
    
        # Apply delta function to mobility, complexity, and variance
        self.deltas_mob = delta(self.mobility_vals, 2)
        self.deltas_deltas_mob = delta(self.deltas_mob, 2)
    
        self.deltas_com = delta(self.complexity_vals, 2)
        self.deltas_deltas_com = delta(self.deltas_com, 2)
    
        self.deltas_var = delta(self.variance_vals, 2)
        self.deltas_deltas_var = delta(self.deltas_var, 2)
        
        # Flatten the arrays
        self.mobility_vals = self.mobility_vals.flatten()
        self.complexity_vals = self.complexity_vals.flatten()
        self.variance_vals = self.variance_vals.flatten()
        self.deltas_mob = self.deltas_mob.flatten()
        self.deltas_deltas_mob = self.deltas_deltas_mob.flatten()
        self.deltas_com = self.deltas_com.flatten()
        self.deltas_deltas_com = self.deltas_deltas_com.flatten()
        self.deltas_var = self.deltas_var.flatten()
        self.deltas_deltas_var = self.deltas_deltas_var.flatten()



  
    
            
    


    def LFB(self, sp, hop_length=512, n_LFB=16):
        # Calcular el número de puntos en cada ventana y el solapamiento
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Crear un array de tiempos para el eje x
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Asegurarse de que los tiempos tienen el formato correcto
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
        # Inicializar vectores para acumular los coeficientes LFB
        self.LFB_matrix_vals = np.zeros((len(times), n_LFB)) 
    
        # Iterar sobre las ventanas y calcular los coeficientes LFB
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            frame_data = self.da[:, start_index:end_index]
    
            # Calcular los coeficientes LFB utilizando tu función personalizada
            LFB = np.float32(logfbank(1+frame_data[sp], samplerate=self.srate, winlen=window_duration, winstep=overlap_duration, nfilt=n_LFB, nfft=hop_length, lowfreq=0, highfreq=None, preemph=0.97))
    
            # Almacenar los coeficientes LFB en la matriz
            self.LFB_matrix_vals[i, :] = LFB
            

        # Convert lists to numpy arrays    
        self.LFB_matrix_vals = np.array(self.LFB_matrix_vals)
        
        # Calcular las derivadas de los coeficientes LFB
        self.deltas_LFB = delta(self.LFB_matrix_vals, 2)
        self.deltas_deltas_LFB = delta(self.deltas_LFB, 2)
        
        
        # Transponer las matrices para cambiar las dimensiones a 16x713
        self.LFB_matrix_vals = self.LFB_matrix_vals.T
        self.deltas_LFB = self.deltas_LFB.T
        self.deltas_deltas_LFB = self.deltas_deltas_LFB.T 
    

    
        
        
    def plot_mel_spectrogram(self, n_mels=128, target_max_frequency=20):
        
        # Define the duration of each window and overlapping
        hop_length = int((window_duration - overlap_duration) * self.srate)
    
        # Calculate the spectrogram of mel
        mel_spectrogram = librosa.feature.melspectrogram(y=self.da[0, :], sr=self.srate, n_mels=n_mels, hop_length=hop_length)
    
        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
        # Adjust the number of pixels on the y-axis to display only the range 0 to 20 Hz.
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=self.srate/2)
        specified_bins = np.where(mel_freqs <= target_max_frequency)[0][-1] + 1
        mel_spectrogram_db = mel_spectrogram_db[:specified_bins, :]
    
        # Create an array of times for the x-axis
        times_x = np.arange(0, len(mel_spectrogram_db[0]), 1) * hop_length / self.srate
    
        # Format for the x-axis of the graphs
        times_x = [self.stime + datetime.timedelta(seconds=t) for t in times_x]
    
        # Display mel spectrogram with adjusted range on y-axis
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=self.srate, hop_length=hop_length, cmap='viridis', x_coords=times_x, y_coords=np.linspace(0, target_max_frequency, mel_spectrogram_db.shape[0]))
        
        
        #To represent the red line equivalent to the earthquake:
        
        # Calculate the number of points in each window and overlapping
        points_per_window = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_window + 1, overlap_points) * self.dt
        
        # Format for the x-axis of the graphs
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
        
 
            
  
    
    def spectral_entropy(self, sp, method='welch', nperseg=None, normalize=True):
        
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
    
        # Initialize vectors for accumulating spectral entropy values
        self.entropy_vals = []
    
        # Iterate over the windows and calculate the spectral entropy
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            frame_data = self.da[:, start_index:end_index]
    
            # Choose between fft or Welch
            if method == 'fft':
                _, psd = periodogram(frame_data[sp], fs=self.srate)
            elif method == 'welch':
                _, psd = welch(frame_data[sp], fs=self.srate, nperseg=nperseg)
                
            # Calculate the probability distribution function of psd
            prob_psd = psd/sum(psd)
    
            # Calculate spectral entropy
            entropy = -np.sum(prob_psd * np.log2(prob_psd + 1e-10))  # Agregar pequeño valor para evitar log(0)
    
            # Normalize if necessary
            if normalize:
                entropy /= np.log2(psd.size)
    
            # Store entropy values in the list
            self.entropy_vals.append([entropy])
            

        # Convert lists to numpy arrays
        self.entropy_vals = np.array(self.entropy_vals)
     
    
        # Apply delta function to mobility, complexity, and variance
        self.deltas_ent = delta(self.entropy_vals, 2)
        self.deltas_deltas_ent = delta(self.deltas_ent, 2)
      
    
        # Flatten the arrays
        self.entropy_vals = self.entropy_vals.flatten()
        self.deltas_ent = self.deltas_ent.flatten()
        self.deltas_deltas_ent = self.deltas_deltas_ent.flatten()
        
        
        
    def scal_log_esp_power(self, sp, method='welch', nperseg=None, normalize=True):
        
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
    
        # Initialize vectors for accumulating spectral entropy values
        self.scal_log_esp_power_vals = []
    
        # Iterate over the windows and calculate the spectral entropy
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            frame_data = self.da[:, start_index:end_index]
    
            # Choose between fft or Welch
            if method == 'fft':
                _, psd = periodogram(frame_data[sp], fs=self.srate)
            elif method == 'welch':
                _, psd = welch(frame_data[sp], fs=self.srate, nperseg=nperseg)
    
            # Calculate spectral entropy
            entropy = -np.sum(psd * np.log2(psd + 1e-10))  # Agregar pequeño valor para evitar log(0)
    
            # Normalize if necessary
            if normalize:
                entropy /= np.log2(psd.size)
    
            # Store entropy values in the list
            self.scal_log_esp_power_vals.append([entropy])
            

        # Convert lists to numpy arrays
        self.scal_log_esp_power_vals = np.array(self.scal_log_esp_power_vals)
     
    
        # Apply delta function to mobility, complexity, and variance
        self.deltas_scal_log_esp_power = delta(self.scal_log_esp_power_vals, 2)
        self.deltas_deltas_scal_log_esp_power = delta(self.deltas_scal_log_esp_power, 2)
      
    
        # Flatten the arrays
        self.scal_log_esp_power_vals = self.scal_log_esp_power_vals.flatten()
        self.deltas_scal_log_esp_power = self.deltas_scal_log_esp_power.flatten()
        self.deltas_deltas_scal_log_esp_power = self.deltas_deltas_scal_log_esp_power.flatten()

      
    
    
        


    def approximate_entropy(self, sp, order=2, metric='chebyshev'):
            
            # Calculate the number of points in each window
            points_per_frame = int(window_duration * self.srate)
            overlap_points = int(overlap_duration * self.srate)
        
            # Create an array of times for the x-axis
            times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
        
            # Make sure that the times are in the correct format
            times = [self.stime + datetime.timedelta(seconds=t) for t in times]
        
        
            # Initialize vectors for accumulating approximate entropy values
            self.approx_entropy_vals = []
        
            # Iterate over the windows and calculate the approximate entropy
            for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
                start_index = index
                end_index = start_index + points_per_frame
                data_current_window = self.da[:, start_index:end_index]
        
                # Calculate approximate entropy using ant.app_entropy
                approx_entropy = ant.app_entropy(data_current_window[sp], order=order, metric=metric)
        
                # Store entropy values in the list
                self.approx_entropy_vals.append([approx_entropy])
                
    
            # Convert lists to numpy arrays
            self.approx_entropy_vals = np.array(self.approx_entropy_vals)
         
        
            # Apply delta function to mobility, complexity, and variance
            self.deltas_ent_app = delta(self.approx_entropy_vals, 2)
            self.deltas_deltas_ent_app = delta(self.deltas_ent_app, 2)
            
            # Flatten the arrays
            self.approx_entropy_vals = self.approx_entropy_vals.flatten()
            self.deltas_ent_app = self.deltas_ent_app.flatten()
            self.deltas_deltas_ent_app = self.deltas_deltas_ent_app.flatten()
        
        
          
        
    
        
        
    def calculate_top_fft_amplitudes_and_frequencies(self, sp, num_top=3):
        
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
        # Initialize vectors for accumulating top FFT amplitudes and frequencies values
        self.top_fft_amplitudes_vals = np.zeros((len(times), num_top))
        self.top_fft_frequencies_vals = np.zeros((len(times), num_top))
        
    
        # Iterate over the windows and calculate the top FFT amplitudes
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            data_current_window = self.da[:, start_index:end_index]
            
            # Crear una copia de data_current_window
            data_window_with_hamming_fft = np.copy(data_current_window)
            
            # Aplicar la ventana de Hamming a la copia de los datos
            hamming_window = np.hamming(points_per_frame)
            data_window_with_hamming_fft *= hamming_window[:, np.newaxis].T  # Transponer para que las dimensiones coincidan
    
            # Calculate FFT of the signal
            fft_vals = np.fft.fft(data_window_with_hamming_fft[sp], n=256)
    
            # Calculate amplitude spectrum
            amplitude_spectrum = np.abs(fft_vals)[:len(fft_vals)//2]
    
            # Find the indices of the top num_top amplitudes
            top_indices = np.argsort(amplitude_spectrum)[::-1][:num_top]
    
            # Get the corresponding amplitudes
            top_amplitudes = amplitude_spectrum[top_indices]
            top_frequencies = np.fft.fftfreq(len(amplitude_spectrum), d=1/self.srate)[top_indices]
            
            
            # Almacenar los top 3 amplitudes y frecuencias en matriz
            self.top_fft_amplitudes_vals[i, :] = top_amplitudes
            self.top_fft_frequencies_vals[i, :] = top_frequencies

            

        # Convert lists to numpy arrays    
        self.top_fft_amplitudes_vals = np.array(self.top_fft_amplitudes_vals)
        self.top_fft_frequencies_vals = np.array(self.top_fft_frequencies_vals)

        
        # Calcular las derivadas de amplitudes y frecuencias
        self.deltas_fft_amp = delta(self.top_fft_amplitudes_vals, 2)
        self.deltas_deltas_fft_amp = delta(self.deltas_fft_amp, 2)
        
        self.deltas_fft_frec = delta(self.top_fft_frequencies_vals, 2)
        self.deltas_deltas_fft_frec = delta(self.deltas_fft_frec, 2)
        
        # Transponer las matrices para cambiar las dimensiones a 3x713
        self.top_fft_amplitudes_vals = self.top_fft_amplitudes_vals.T
        self.deltas_fft_amp = self.deltas_fft_amp.T
        self.deltas_deltas_fft_amp = self.deltas_deltas_fft_amp.T 
        self.top_fft_frequencies_vals = self.top_fft_frequencies_vals.T
        self.deltas_fft_frec = self.deltas_fft_frec.T
        self.deltas_deltas_fft_frec = self.deltas_deltas_fft_frec.T         
    
  

        
        
   
        
    def percentiles(self, sp):
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
        # Initialize matrix for percentiles
        self.percentiles_matrix = np.zeros((len(times), 3))  # 3 rows for 20th, 50th, and 80th percentiles
    
        # Iterate over the windows and calculate the percentiles
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            data_current_window = self.da[:, start_index:end_index]
    
            # Calculate cumulative sum
            cumsum = np.cumsum(np.abs(data_current_window[sp]))
    
            # Calculate percentiles
            percentile_20_val = np.percentile(cumsum, 20)
            percentile_50_val = np.percentile(cumsum, 50)
            percentile_80_val = np.percentile(cumsum, 80)
    
            # Store percentiles in matrix
            self.percentiles_matrix[i, :] = [percentile_20_val, percentile_50_val, percentile_80_val]
            
            

        # Convert lists to numpy arrays    
        self.percentiles_matrix = np.array(self.percentiles_matrix)
        
        # Calcular las derivadas de los percentiles
        self.deltas_perc = delta(self.percentiles_matrix, 2)
        self.deltas_deltas_perc = delta(self.deltas_perc, 2)
        
        # Transponer las matrices para cambiar las dimensiones a 3x713
        self.percentiles_matrix = self.percentiles_matrix.T
        self.deltas_perc = self.deltas_perc.T
        self.deltas_deltas_perc = self.deltas_deltas_perc.T 
    
    
    
        
    def lpc(self, sp, order=8):
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)

        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt

        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]

        # Initialize matrix for accumulating LPC coefficients
        self.lpc_matrix = np.zeros((len(times), order))
    
        # Iterate over the windows and calculate the LPC coefficients
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            data_current_window = self.da[:, start_index:end_index]

            # Calculate LPC coefficients using librosa.lpc
            lpc_coefficients = librosa.lpc(data_current_window[sp], order=order)
            
            
            # Discard the first coefficient (always 1)
            lpc_coefficients = lpc_coefficients[1:]
    
            # Store LPC coefficients in the matrix            
            self.lpc_matrix[i, :] = lpc_coefficients
             

        # Convert lists to numpy arrays    
        self.lpc_matrix = np.array(self.lpc_matrix)
        
        # Calcular las derivadas de los coeficientes LPC
        self.deltas_LPC = delta(self.lpc_matrix, 2)
        self.deltas_deltas_LPC = delta(self.deltas_LPC, 2)        
        
        # Transponer las matrices para cambiar las dimensiones a 8x713
        self.lpc_matrix = self.lpc_matrix.T
        self.deltas_LPC = self.deltas_LPC.T
        self.deltas_deltas_LPC = self.deltas_deltas_LPC.T 
    


    


    def lpc_5(self, sp, coefficient_number=6, order=8):
            
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
    
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
    
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
    
        # Initialize list to store the fifth LPC coefficient values
        self.fifth_lpc_coefficients = []
    
        # Iterate over the windows and calculate the fifth LPC coefficient
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            data_current_window = self.da[:, start_index:end_index]
    
            # Calculate LPC coefficients using librosa.lpc
            lpc_librosa = librosa.lpc(data_current_window[sp], order=order)
    
            # Append the fifth LPC coefficient to the list
            self.fifth_lpc_coefficients.append(lpc_librosa[coefficient_number])
    
 
        
        
  
        
        
    def calculate_top_lpc_amplitudes_and_frequencies(self, sp, num_top=3):
                
        # Calculate the number of points in each window
        points_per_frame = int(window_duration * self.srate)
        overlap_points = int(overlap_duration * self.srate)
        
        # Create an array of times for the x-axis
        times = np.arange(0, self.nsamp - points_per_frame + 1, overlap_points) * self.dt
        
        # Make sure that the times are in the correct format
        times = [self.stime + datetime.timedelta(seconds=t) for t in times]
        
        
        # Initialize matrices to store top LPC amplitudes and frequencies for each window
        self.top_lpc_amplitudes_vals = np.zeros((len(times), num_top))
        self.top_lpc_frequencies_vals = np.zeros((len(times), num_top))
      
        
        # Iterate over the windows and calculate the top LPC amplitudes and frequencies
        for i, index in enumerate(range(0, self.nsamp - points_per_frame + 1, overlap_points)):
            start_index = index
            end_index = start_index + points_per_frame
            data_current_window = self.da[:, start_index:end_index]
            
            # Crear una copia de data_current_window
            data_window_with_hamming = np.copy(data_current_window)
            
            # Aplicar la ventana de Hamming a la copia de los datos
            hamming_window = np.hamming(points_per_frame)
            data_window_with_hamming *= hamming_window[:, np.newaxis].T  # Transponer para que las dimensiones coincidan
        
            # Calculate LPC coefficients
            lpc_coeffs = librosa.lpc(data_window_with_hamming[sp], order=8)
            
            # Generate LPC filter using LPC coefficients
            w, h = signal.freqz(1, lpc_coeffs, 256, fs=self.srate)   
            
            # Calculate LPC spectrum
            lpc_spectrum = np.abs(h)
            
            # Find the indices of the top num_top amplitudes
            top_indices = np.argsort(lpc_spectrum)[::-1][:num_top]
            
            # Get the corresponding amplitudes and frequencies
            top_amplitudes = lpc_spectrum[top_indices]
            
            # Calculate the frequency corresponding to each top index using fftfreq
            fft_freqs = np.fft.fftfreq(len(lpc_spectrum), d=1/self.srate)
            top_frequencies = fft_freqs[top_indices]
            
  
            # Almacenar los top 3 amplitudes y frecuencias en matriz
            self.top_lpc_amplitudes_vals[i, :] = top_amplitudes
            self.top_lpc_frequencies_vals[i, :] = top_frequencies

            

        # Convert lists to numpy arrays    
        self.top_lpc_amplitudes_vals = np.array(self.top_lpc_amplitudes_vals)
        self.top_lpc_frequencies_vals = np.array(self.top_lpc_frequencies_vals)

        
        # Calcular las derivadas de amplitudes y frecuencias
        self.deltas_lpc_amp = delta(self.top_lpc_amplitudes_vals, 2)
        self.deltas_deltas_lpc_amp = delta(self.deltas_lpc_amp, 2)
        
        self.deltas_lpc_frec = delta(self.top_lpc_frequencies_vals, 2)
        self.deltas_deltas_lpc_frec = delta(self.deltas_lpc_frec, 2)
        
        # Transponer las matrices para cambiar las dimensiones a 3x713
        self.top_lpc_amplitudes_vals = self.top_lpc_amplitudes_vals.T
        self.deltas_lpc_amp = self.deltas_lpc_amp.T
        self.deltas_deltas_lpc_amp = self.deltas_deltas_lpc_amp.T 
        self.top_lpc_frequencies_vals = self.top_lpc_frequencies_vals.T
        self.deltas_lpc_frec = self.deltas_lpc_frec.T
        self.deltas_deltas_lpc_frec = self.deltas_deltas_lpc_frec.T   
        
    
  

   