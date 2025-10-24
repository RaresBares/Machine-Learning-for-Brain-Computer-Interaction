import numpy as np
from scipy import signal

def preprocess_eeg_fft(eeg, fs, band=(1.0, 40.0), notch=None, notch_bw=1.0, reref="avg", zscore=True):
    """
    Parameters
    ----------
    eeg : numpy.ndarray
        Shape (n_channels, n_samples), dtype float. Raw EEG.
    fs : float or int
        Sampling rate in Hz.
    band : tuple[float, float]
        (low_hz, high_hz) passband for FFT-domain bandpass.
    notch : float | None
        Mains frequency in Hz to notch (e.g., 50 or 60). If None, disabled.
    notch_bw : float
        Notch bandwidth in Hz (± notch_bw/2 around the notch; also applies to 2× notch if in Nyquist).
    reref : str | None
        If "avg", apply common average reference; otherwise no rereferencing.
    zscore : bool
        If True, per-channel z-normalization after filtering.

    Returns
    -------
    numpy.ndarray
        Shape (n_channels, n_samples), dtype float. Preprocessed EEG with same shape as input.
    """
    x = np.asarray(eeg, dtype=float)
    x = signal.detrend(x, axis=1, type="constant")
    n = x.shape[1]
    f = np.fft.rfftfreq(n, d=1.0/fs)
    keep = (f >= band[0]) & (f <= band[1])
    X = np.fft.rfft(x, axis=1)
    mask = keep.copy()
    if notch is not None:
        nb = notch_bw * 0.5
        mask &= ~((f >= notch - nb) & (f <= notch + nb))
        h = 2.0 * notch
        if h < fs * 0.49:
            mask &= ~((f >= h - nb) & (f <= h + nb))
    X *= mask[None, :]
    x = np.fft.irfft(X, n=n, axis=1)
    if reref == "avg":
        x = x - x.mean(axis=0, keepdims=True)
    if zscore:
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - m) / s
    return x

def epoch_bandpower_fft(eeg, fs, events, tmin, tmax, bands={"mu": (8, 12), "beta": (13, 30)}, window="hann"):
    """
    Parameters
    ----------
    eeg : numpy.ndarray
        Shape (n_channels, n_samples), dtype float. Preprocessed or raw EEG.
    fs : float or int
        Sampling rate in Hz.
    events : numpy.ndarray | list[int]
        Event onsets as sample indices. Shape (n_events,), dtype int.
    tmin : float
        Start time relative to each event in seconds (can be negative).
    tmax : float
        End time relative to each event in seconds (must be > tmin).
    bands : dict[str, tuple[float, float]]
        Mapping of band name -> (low_hz, high_hz). Order of values defines feature order.
    window : str
        Window name for FFT (e.g., 'hann').

    Returns
    -------
    feats : numpy.ndarray
        Bandpower features per epoch. Shape (n_kept_epochs, n_channels * n_bands), dtype float.
    ev : numpy.ndarray
        Events that were within bounds. Shape (n_kept_epochs,), dtype int.
    """
    x = np.asarray(eeg, dtype=float)
    events = np.asarray(events, dtype=int)
    pre = int(round(tmin * fs))
    post = int(round(tmax * fs))
    keep = (events + pre >= 0) & (events + post <= x.shape[1])
    ev = events[keep]
    feats = []
    for e in ev:
        seg = x[:, e + pre : e + post]
        n = seg.shape[1]
        w = signal.get_window(window, n, fftbins=True)
        U = (w**2).sum() / n
        f = np.fft.rfftfreq(n, d=1.0/fs)
        S = np.fft.rfft(seg * w, axis=1)
        Pxx = (np.abs(S) ** 2) / (fs * n * U)
        fb = []
        for lo, hi in bands.values():
            idx = (f >= lo) & (f <= hi)
            df = np.diff(f).mean() if f.size > 1 else 0.0
            bp = Pxx[:, idx].sum(axis=1) * df
            fb.append(np.log10(bp + 1e-12))
        fb = np.concatenate(fb, axis=0)
        feats.append(fb)
    feats = np.vstack(feats) if len(feats) else np.empty((0, x.shape[0] * len(bands)))
    return feats, ev

