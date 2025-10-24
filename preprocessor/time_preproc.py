import numpy as np
from scipy import signal

def preprocess_eeg(eeg, fs, band=(1.0, 40.0), notch=None, reref="avg", zscore=True):
    """
    Parameters
    ----------
    eeg : numpy.ndarray
        Shape (n_channels, n_samples), dtype float. Raw EEG.
    fs : float or int
        Sampling rate in Hz.
    band : tuple[float, float]
        (low_hz, high_hz) passband for time-domain Butterworth filter.
    notch : float | None
        Mains frequency in Hz to notch (e.g., 50 or 60). If None, disabled. Also removes 2× notch if < Nyquist.
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
    if notch is not None:
        b, a = signal.iirnotch(notch / (fs / 2.0), 30.0)
        x = signal.filtfilt(b, a, x, axis=1)
        h = 2.0 * notch
        if h < fs * 0.49:
            b, a = signal.iirnotch(h / (fs / 2.0), 30.0)
            x = signal.filtfilt(b, a, x, axis=1)
    wp = (band[0] / (fs / 2.0), band[1] / (fs / 2.0))
    b, a = signal.butter(4, wp, btype="band")
    x = signal.filtfilt(b, a, x, axis=1)
    if reref == "avg":
        x = x - x.mean(axis=0, keepdims=True)
    if zscore:
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - m) / s
    return x

def epoch_bandpower(eeg, fs, events, tmin, tmax, bands={"mu": (8, 12), "beta": (13, 30)}, nperseg=256):
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
    nperseg : int
        Segment length for Welch PSD; clipped to epoch length.

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
        f, pxx = signal.welch(seg, fs=fs, nperseg=min(nperseg, seg.shape[1]), axis=1, average="median")
        fb = []
        for lo, hi in bands.values():
            idx = (f >= lo) & (f <= hi)
            bp = np.trapz(pxx[:, idx], f[idx], axis=1)
            fb.append(np.log10(bp + 1e-12))
        fb = np.concatenate(fb, axis=0)
        feats.append(fb)
    feats = np.vstack(feats) if len(feats) else np.empty((0, x.shape[0] * len(bands)))
    return feats, ev