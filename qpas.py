import os
import scipy
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import h5py
import time
import glob
from scipy.signal import hilbert, tukey
from scipy.fftpack import fft, rfft, irfft, rfftfreq
from scipy.optimize import least_squares
from numba import njit, prange
from skimage.transform import resize
from lightgbm import LGBMRegressor

def tukeybandpass(signal, sample_rate, axis=1, alpha=1):
    """Applies a bandpass using a Tukey window.
    TODO
    """
    frequency = rfftfreq(signal.shape[axis],d=1/sample_rate)
    print(max(frequency))
    window = tukey(signal.shape[axis], alpha=alpha)
    bp_signal = np.zeros_like(signal)
    if signal.ndim == 3:
        dims = [0,1,2]
        dims.remove(axis)
        for i in range(signal.shape[dims[0]]):
            for j in range(signal.shape[dims[1]]):
                w_f_s = rfft(signal[i,:,j].copy())*window
                bp_signal[i,:,j] = irfft(w_f_s)

    if signal.ndim == 4:
        dims = [0,1,2,3]
        dims.remove(axis)
        for i in range(signal.shape[dims[0]]):
            for j in range(signal.shape[dims[1]]):
                for k in range(signal.shape[dims[2]]):
                    w_f_s = rfft(signal[i,:,j,k].copy())*window
                    bp_signal[i,:,j,k] = irfft(w_f_s)
                
    if signal.ndim == 5:
        dims = [0,1,2,3,4]
        dims.remove(axis)
        for i in range(signal.shape[dims[0]]):
            for j in range(signal.shape[dims[1]]):
                for k in range(signal.shape[dims[2]]):
                    for l in range(signal.shape[dims[3]]):
                        w_f_s = rfft(signal[i,:,j,k,l].copy())*window
                        bp_signal[i,:,j,k,l] = irfft(w_f_s)

    if signal.ndim == 2 or signal.ndim == 1:
        w_f_s = rfft(signal.copy())*window
        bp_signal = irfft(w_f_s)
    if signal.ndim > 5:
        print("invalid input dimensions")

    return bp_signal

def bmode(bf_image, axis=1):
    """Applies a B-Mode filter.
    Using a Hilbert transform for envelope detection along axis.
    """
    analytic_signal = np.zeros_like(bf_image)
    if bf_image.ndim == 3:
        for i in range(bf_image.shape[2]):
            analytic_signal[:,:,i] = hilbert(bf_image[:,:,i], axis=axis)
    if bf_image.ndim == 4:
        for i in range(bf_image.shape[2]):
            for j in range(bf_image.shape[3]):
                analytic_signal[:,:,i,j] = hilbert(bf_image[:,:,i,j], axis=axis)
    if bf_image.ndim == 5:
        for i in range(bf_image.shape[2]):
            for j in range(bf_image.shape[3]):
                for k in range(bf_image.shape[4]):
                    analytic_signal[:,:,i,j,k] = hilbert(bf_image[:,:,i,j,k], axis=axis)
    if bf_image.ndim == 1:
        analytic_signal = hilbert(bf_image, axis=axis)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

class beamformer():
    
    def __init__(self, sample_rate, speed_of_sound, 
                 modality="PA", algorithm="DAS"):
        """Initilizing Beamformer.
        Default is delay and sum for photoacoustics.

        Keyword arguments:
        speed_of_sound -- assumed speed of sound in the medium [m/s]
        sample_rate -- sample rate of the input data in [Hz]
        modality -- photoacoustics ("PA") or plane wave ultrasound for 
                    linear trasducers ("plane wave US") is implemented so far 
                    (default: "PA")
        algorithm -- only delay and sum is implemeted (default: "DAS")
        """
        self._sample_rate = sample_rate
        self._speed_of_sound = speed_of_sound
        self._modality = None
        if modality=="PA" or modality=="plane wave US":
            self._modality = modality
        else:
            raise ValueError('modality is set to an invalid value. Photoacoustics (PA) or plane wave ultrasound for linear trasducers (plane wave US) is implemented so far')
        self._algorithm = algorithm
        
    def set_linear_sensor(self, n_elements=128, sensor_width=38.1):
        """Define a linear sensor geometry. 
        Default is a L4-7 linear array transducer or similar.
        Custom (e.g. curved or arbitrarily shaped) sensors can currently be
        defined directly by modifying the self.sensor_pos array but 
        apodization is not correctly implemented for that case.
        
        Keyword arguments:
        n_elements -- number of elements (default: 128)
        sensor_width -- linear array width (default: 38.4)
        """
        self.sensor_geometry = "linear"
        self.sensor_elements = n_elements if n_elements is not None else 128
        self.sensor_width = sensor_width if sensor_width is not None else 38.4
        self.sensor_pos = np.zeros([2, self.sensor_elements])
        self.sensor_pos[0,:] = np.linspace(0,
                                           self.sensor_width,
                                           self.sensor_elements)
        # relevant for apodisation - surface normal vector on sensor
        # TODO use this information for the apodization
        self.sensor_alignment = np.zeros([2, self.sensor_elements])
        self.sensor_alignment[1,:] = np.ones(self.sensor_elements)
        
    def set_output_area(self, x0, x1, 
                        min_depth, max_depth, 
                        pixel_count_x, pixel_count_y):
        """TODO
        """
        self._x_pos = np.linspace(x0,x1,pixel_count_x)
        self._y_pos = np.linspace(min_depth,max_depth,pixel_count_y)
        self._spacing = np.array([self._x_pos[1]-self._x_pos[0], 
                                  self._y_pos[1]-self._y_pos[0]])
        
        self._output = np.zeros([pixel_count_x, 
                                 pixel_count_y])
        
    def set_apodization(self, func = None, angle = 90):  
        """Setting the apodization function and angle for beamforming. 
        func -- apodization function, "Box" or "Hann" are implemented 
        angle -- maximum angle from the normal on the sensor surface [deg]
                 (default: 90)
        """
        self._apodization_function = func
        self._apodization_angle = angle
        
    def beamform(self, signal):
        """Perform beamforming. 
        Wrapper for the numba beamforming function.
        Assumes that axis 0 is transducers and axis 1 is time. 
        Further axis are all treated as seperate frames.
        Needed for the numba function to compile.
        signal -- input signal
        """

        if signal.ndim == 2:
            output = self._beamform_numba(
                signal=signal,
                output=self._output,
                sensor_pos=self.sensor_pos,
                spacing=self._spacing,
                sos=self._speed_of_sound,
                sample_rate=self._sample_rate,
                mode=self._modality,
                apod_angle=self._apodization_angle,
                apod_func=self._apodization_function,
                y_min=self._y_pos[0]/self._spacing[1])

            
        if signal.ndim == 3: 
            output = np.zeros([self._output.shape[0],
                               self._output.shape[1],
                               signal.shape[2]])
            for i in range(signal.shape[2]):
                output[:,:,i] = self._beamform_numba(
                    signal=signal[:,:,i],
                    output=self._output,
                    sensor_pos=self.sensor_pos,
                    spacing=self._spacing,
                    sos=self._speed_of_sound,
                    sample_rate=self._sample_rate,
                    mode=self._modality,
                    apod_angle=self._apodization_angle,
                    apod_func=self._apodization_function,
                    y_min=self._y_pos[0]/self._spacing[1])
                
        if signal.ndim == 4: 
            output = np.zeros([self._output.shape[0],
                               self._output.shape[1],
                               signal.shape[2],
                               signal.shape[3]])
            for i in range(signal.shape[2]):
                for j in range(signal.shape[3]):
                    output[:,:,i,j] = self._beamform_numba(
                        signal=signal[:,:,i,j],
                        output=self._output,
                        sensor_pos=self.sensor_pos,
                        spacing=self._spacing,
                        sos=self._speed_of_sound,
                        sample_rate=self._sample_rate,
                        mode=self._modality,
                        apod_angle=self._apodization_angle,
                        apod_func=self._apodization_function,
                        y_min=self._y_pos[0]/self._spacing[1])

        if signal.ndim == 5: 
            output = np.zeros([self._output.shape[0],
                               self._output.shape[1],
                               signal.shape[2],
                               signal.shape[3],
                               signal.shape[4]])
            for i in range(signal.shape[2]):
                for j in range(signal.shape[3]):
                    for k in range(signal.shape[4]):
                        output[:,:,i,j,k] = self._beamform_numba(
                            signal=signal[:,:,i,j,k],
                            output=self._output,
                            sensor_pos=self.sensor_pos,
                            spacing=self._spacing,
                            sos=self._speed_of_sound,
                            sample_rate=self._sample_rate,
                            mode=self._modality,
                            apod_angle=self._apodization_angle,
                            apod_func=self._apodization_function,
                            y_min=self._y_pos[0]/self._spacing[1])
        return output

    @staticmethod
    @njit(parallel = True)
    def _beamform_numba(signal, output, 
                        sensor_pos, spacing, sos, sample_rate, mode, 
                        apod_angle, apod_func, y_min):
        """Beamforming routine for one frame. 
        numba for faster-than-python computation
        directionality is not implemented and curved arrays are not tested.
        return: one beamformed image.
        """
        for x in prange(output.shape[0]):
            for y in range(output.shape[1]):
                _tmp = 0
                for i in range(signal.shape[0]):
                    dx = sensor_pos[0,i] - spacing[0]*(x)
                    dy = sensor_pos[1,i] - spacing[1]*(y+y_min)
                    deg = np.abs(np.arctan(dx/dy)/np.pi*180)
                    apod = 0
                    if deg <= apod_angle:
                        if apod_func == "Hann": ### oonly for linear trans rn
                            apod = 0.5-0.5*np.cos(deg/apod_angle*np.pi+np.pi)
                        if apod_func == "Box":
                            apod = 1                            
                        if mode == "PA":
                            delay = np.sqrt(dx**2 + dy**2)/sos/1000
                        if mode == "plane wave US":
                            delay = (np.sqrt(dx**2 + dy**2)
                                     - dy) / sos / 1000
                        _tmp += apod * signal[i, int(np.round(delay * 
                                                              sample_rate, 
                                                              0))]
                output[x,y] = _tmp
        return output

def create_vid(inputData, filename, titel, resolution, vmin, vmax,
               fps = 20, cmap = cm.viridis, 
               normalize_frame = 'True', aspect = 1):
    """Rendering a video of US or PA inputData.
    """
    plt.style.use('default')
    dpi = 300
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if(normalize_frame == 'One'):
        _cur_data_norm = (inputData[:,:,0]/np.nanmax(inputData[:,:,0]))
        vmin = 0
        vmax = 1
    if(normalize_frame == 'Log'):
        _cur_data_norm = 20 * np.log10(inputData[:,:,0]/np.nanmax(inputData))
        vmin = vmin
        vmax = 0
    if(normalize_frame == 'True'):
        _cur_data_norm = inputData[:,:,0]/np.nanmax(inputData)
        vmin = vmin
        vmax = vmax
    if(normalize_frame == 'False'):
        _cur_data_norm = inputData[:,:,0]
        vmin = vmin
        vmax = vmax
    extent = [0, inputData.shape[0]*resolution, inputData.shape[1]*resolution, 0]
    im = ax.imshow(np.rot90(_cur_data_norm[:,:], k=-1), 
                   cmap = cmap, 
                   vmin = vmin, vmax = vmax,
                   interpolation = 'nearest', 
                   extent = extent)
    fig.set_size_inches([8,4])
    fig.colorbar(im, ax = ax)
    plt.title(titel)
    plt.xlabel("x [mm]")
    plt.ylabel("depth [mm]")
    plt.tight_layout()

    def update_img(n):
        if(normalize_frame == 'One'):
            _cur_data_norm = (inputData[:,:,n]/np.nanmax(inputData[:,:,n]))
        if(normalize_frame == 'Log'):
            _cur_data_norm = 20 * np.log10(inputData[:,:,n]/np.nanmax(inputData[:,:,:]))
        if(normalize_frame == 'True'):
            _cur_data_norm = inputData[:,:,n]/np.nanmax(inputData)
        if(normalize_frame == 'False'):
            _cur_data_norm = inputData[:,:,n]

        im.set_data(np.rot90(_cur_data_norm[:,:], k=-1))
        if n%100==0:
            print("vid progress: "+str(n)+"/"+str(inputData.shape[2])+" frames")
        return im

    ani = animation.FuncAnimation(fig, update_img, 
                                  inputData[0,0,:].size, 
                                  interval = 20)
    #FFMpegWriter Pipe-based ffmpeg writer.
    writer = animation.FFMpegWriter(fps = fps, bitrate = None)
    
    testpath = filename + '.mp4'
    print(testpath)
    ani.save(testpath, writer = writer, dpi = dpi)
    return ani

def svd_pi_lu_oxy(input_data, wavelength_axis, hemoglobin_df_path, wavelenghs):
    """Fast linear unmixing.
    using "Fast Linear Unmixing for PhotoAcoustic Imaging (FLUPAI)" frei nach Niklas Holtzwarth 
    - SVD with a pseudo inverse matrix
    - equivalent to a least squares ansatz for linear spectral unmixing of multi-spectral photoacoustic images
    return: oxygenation
    """
    # reshape input
    input_data = np.swapaxes(input_data, wavelength_axis, 0)
    shape_mem = np.asarray(input_data.shape)
    input_data = np.reshape(input_data, (len(wavelenghs), -1))
    
    # load hemoglobin spectra
    df_sO2 = pd.read_csv(hemoglobin_df_path+"absorption_spectra_hemoglobin.csv", header=0)
    df_sO2 = df_sO2[df_sO2["lambda/nm"].isin(wavelenghs)]

    # create piv absorption matrix for Hb and HbO endmembers
    endmember_matrix = np.zeros((len(wavelenghs), 2))

    # write absorption data for each chromophore and the corresponding wavelength into a matrix
    for i_chrom, key in enumerate(['mua(HbO)//cmM', 'mua(Hb)//cmM']):
        for i_wl, wl in enumerate(wavelenghs):
            foo = df_sO2[df_sO2["lambda/nm"].isin([wl])]
            endmember_matrix[i_wl,i_chrom] = foo[key]
            piv = np.linalg.pinv(endmember_matrix)

    # matmul of abundances = piv * input_data with chromophore abundances, piv pseudo inverse matrix with absorber information
    abundances = np.matmul(piv, input_data)
    oxygenation = abundances[0,:]/np.sum(abundances, axis=0)
    thb = np.sum(abundances, axis=0)
    
    # reshape back to input shape
    shape_mem[0] = 1 # oxygenation result is 1D
    oxygenation = np.reshape(oxygenation, (shape_mem))
    oxygenation = np.swapaxes(oxygenation, wavelength_axis, 0)
    thb = np.reshape(thb, (shape_mem))
    thb = np.swapaxes(thb, wavelength_axis, 0)
    return np.squeeze(oxygenation), np.squeeze(thb)