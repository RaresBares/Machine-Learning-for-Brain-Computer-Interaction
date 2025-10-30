import numpy as np
from scipy.signal import detrend, welch

class ProcChannel:
    def __init__(
        self,
        raw,
        fs,
        clean,
        t,
        freqs_fft,
        fft_mag,
        freqs_psd,
        psd,
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
        blink_power,
        energy
    ):
        self.raw = raw
        self.fs = fs
        self.clean = clean
        self.t = t
        self.fft_freqs = freqs_fft
        self.fft_mag = fft_mag
        self.psd_freqs = freqs_psd
        self.psd = psd
        self.delta = delta_power
        self.theta = theta_power
        self.alpha = alpha_power
        self.beta = beta_power
        self.blink = blink_power
        self.energy = energy

    @classmethod
    def preprocess(cls, x, duration_s):
        x = np.asarray(x, dtype=float).ravel()
        n = x.shape[0]
        fs = n / float(duration_s)

        x_dt = detrend(x, type="constant")

        freqs_full = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_full = np.fft.rfft(x_dt)
        mask = (freqs_full >= 3.0) & (freqs_full <= 40.0)
        fft_full_filt = fft_full * mask
        x_filt = np.fft.irfft(fft_full_filt, n=n)

        t = np.arange(n) / fs

        freqs_fft, fft_mag = _fft_mag(x_filt, fs)
        freqs_psd, psd = _psd(x_filt, fs)

        delta_power = _band_power(freqs_psd, psd, 0.5, 4.0)
        theta_power = _band_power(freqs_psd, psd, 4.0, 8.0)
        alpha_power = _band_power(freqs_psd, psd, 8.0, 13.0)
        beta_power  = _band_power(freqs_psd, psd, 13.0, 30.0)

        blink_power = delta_power
        energy = np.mean(x_filt ** 2)

        return cls(
            raw=x,
            fs=fs,
            clean=x_filt,
            t=t,
            freqs_fft=freqs_fft,
            fft_mag=fft_mag,
            freqs_psd=freqs_psd,
            psd=psd,
            delta_power=delta_power,
            theta_power=theta_power,
            alpha_power=alpha_power,
            beta_power=beta_power,
            blink_power=blink_power,
            energy=energy,
        )

def _normalize(x):
    m = np.mean(x)
    s = np.std(x) + 1e-9
    return (x - m) / s

def _fft_mag(x, fs):
    n = len(x)
    window = np.hanning(n)
    xw = x * window
    F = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(F)
    return freqs, mag

def _psd(x, fs):
    freqs, pxx = welch(
        x,
        fs=fs,
        nperseg=min(256, len(x)),
        noverlap=None,
        detrend="constant",
        return_onesided=True,
        scaling="density"
    )
    return freqs, pxx

def _band_power(freqs, spec, f_lo, f_hi):
    idx = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(idx):
        return 0.0
    return np.trapz(spec[idx], freqs[idx])

def build_from_channel(chan):
    """
    Build a ProcChannel from an existing Channel.

    Uses chan.clean (time domain), chan.fs, and derives frequency/feature info.
    No additional filtering, notch, or normalization is applied here.
    """
    x = np.asarray(chan.clean, dtype=float).ravel()
    fs = float(chan.fs)
    n = x.shape[0]
    t = np.arange(n) / fs

    # FFT magnitude from the clean signal
    freqs_fft, fft_mag = _fft_mag(x, fs)

    # PSD estimate
    freqs_psd, psd = _psd(x, fs)

    # Band powers / features
    delta_power = _band_power(freqs_psd, psd, 0.5, 4.0)
    theta_power = _band_power(freqs_psd, psd, 4.0, 8.0)
    alpha_power = _band_power(freqs_psd, psd, 8.0, 13.0)
    beta_power = _band_power(freqs_psd, psd, 13.0, 30.0)

    blink_power = delta_power
    energy = np.mean(x ** 2)

    return ProcChannel(
        raw=x,
        fs=fs,
        clean=x,
        t=t,
        freqs_fft=freqs_fft,
        fft_mag=fft_mag,
        freqs_psd=freqs_psd,
        psd=psd,
        delta_power=delta_power,
        theta_power=theta_power,
        alpha_power=alpha_power,
        beta_power=beta_power,
        blink_power=blink_power,
        energy=energy
    )