import numpy as np
import scipy.signal
from scipy.special._precompute.expn_asy import x

from data import fetcher




import numpy as np
import matplotlib.pyplot as plt

def plot_signal_and_spectrum(x, t):
    """
    x: Signal im Zeitbereich, shape (N,)
    t: Zeitstempel in Sekunden, shape (N,)
    Plottet:
      1) x(t) über der Zeit
      2) |FFT(x)| über der Frequenz
    """

    # Zeitsignal
    dt = t[1] - t[0]
    N = len(x)

    # Frequenzachse und Spektrum
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(X)

    # Plot Zeitbereich
    fig_time, ax_time = plt.subplots(1, 1, figsize=(10, 3))
    ax_time.plot(t, x)
    ax_time.set_xlabel("t [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Signal im Zeitbereich")
    plt.tight_layout()
    plt.show()

    # Plot Frequenzbereich
    fig_freq, ax_freq = plt.subplots(1, 1, figsize=(10, 3))
    ax_freq.semilogy(freqs, mag)
    ax_freq.set_xlabel("f [Hz]")
    ax_freq.set_ylabel("|X(f)|")
    ax_freq.set_title("Spektrum (Betrag der rFFT)")
    plt.tight_layout()
    plt.show()


def detrend(data):
    return scipy.signal.detrend(data, type="constant")

def filter(data, timestambs):
    dt = timestambs[1] - timestambs[0]
    N = len(data)

    X = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=dt)

    mask = (freqs >= 3.0) & (freqs <= 30.0)

    band_fft_mag = np.abs(X)[mask]
    band_freqs = freqs[mask]

    return band_fft_mag, band_freqs

def normalize(data):
    min = np.min(data)
    max = np.max(data)
    return [(elem - min)/(max - min) for elem in data]

def discretize(data):
    arr = np.asarray(data, dtype=float)
    return np.round(arr / 0.1) * 0.1
fs = 256
t, raw_channels = fetcher.generate_bci_dummy_with_peaks(channels=2, seconds=5.0, fs=fs, seed=0)
data =raw_channels[1].clean
timestambs = t

data = detrend(data)
band_fft_mag, freq = filter(data, timestambs)
norm_fft_mag = normalize(band_fft_mag)
display_fft_mag = discretize(norm_fft_mag)
plt.plot( freq,discretize(norm_fft_mag))
plt.show()


def process(data, timestambs):
    data = detrend(data)
    band_fft_mag, freq = filter(data, timestambs)
    norm_fft_mag = normalize(band_fft_mag)
    display_fft_mag = discretize(norm_fft_mag)
    return freq, display_fft_mag
