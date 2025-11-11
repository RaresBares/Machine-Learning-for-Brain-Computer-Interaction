import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import detrend


def preprocess_time_to_freq(input_time_array, dt, fc=40):
    def getFourier(array, dt):
        N = array.size
        f = np.fft.rfftfreq(N, d=dt)
        X = np.fft.rfft(array)
        df = 1/(dt*N)
        A = np.abs(X) / N
        if N % 2 == 0:
            A[1:-1] *= 2.0
        else:
            A[1:] *= 2.0
        return A, f, df
    def normalize(array):
        x = np.asarray(array, dtype=float)
        m = np.max(np.abs(x))
        y = x / m if m != 0 else np.zeros_like(x)
        return y
    def deTrend(array):
        return detrend(array, type='constant')
    def lowPass(fourier, freqs, fc):
        fourier[freqs > fc] = 0
        return fourier

    def cutOff(fourier, freqs, fc):
        mask = freqs <= fc
        return fourier[mask], freqs[mask]

    def interpolate(fourier, freqs, fc):
        fd = float(freqs[0])
        f_new = np.linspace(fd, fc, 128)
        y_128 = interp1d(freqs, fourier, kind='cubic', bounds_error=False, fill_value='extrapolate')(f_new)
        df_new = (f_new[-1] - f_new[0]) / (len(f_new) - 1) if len(f_new) > 1 else 0.0
        return y_128, f_new, df_new
    def deMean(y):
        return y - np.mean(y)

    input_time_array = deMean(input_time_array)
    input_time_array = deTrend(input_time_array)
    input_time_array, freqs, df = getFourier(input_time_array, dt)
    input_time_array = normalize(input_time_array)

    input_time_array = lowPass(input_time_array, freqs, fc)

    cut_fourier, cut_freqs = cutOff(input_time_array, freqs, fc)

    result_fourier, result_freqs, result_df = interpolate(cut_fourier, cut_freqs, fc)
    result_fourier = normalize(result_fourier)
    return result_fourier, result_freqs, result_df



def dummy(t):
    y  = np.cos(2*np.pi*t*8)
    y += np.cos(2*np.pi*t*6)
    y += np.cos(2*np.pi*t*10)
    y += np.cos(2*np.pi*t*25)
    y += np.cos(2*np.pi*t*5)
    return y

