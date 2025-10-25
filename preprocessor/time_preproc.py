import numpy as np
from scipy import signal

def preprocess_eeg(eeg, fs, band=(1.0, 40.0), notch=None, reref="avg", zscore=True):
    """
    Parameters
    ----------
    eeg : np.ndarray of Channel
        Array of Channel objects; each `Channel.clean` is read and overwritten in-place.
    fs : float | int | None
        Sampling rate in Hz. If None, inferred from channels' `fs` (must all match).
    band : tuple[float, float]
    notch : float | None
    reref : str | None
    zscore : bool
    Returns
    -------
    np.ndarray of Channel
        The same array of Channel objects with updated `clean`.
    """
    chs = np.asarray(eeg, dtype=object).ravel()
    # Collect time-domain arrays from channels
    arrs = []
    for ch in chs:
        sig = getattr(ch, "clean", None)
        if sig is None:
            raise ValueError("All channels must have a non-None `clean` array")
        arrs.append(np.asarray(sig, dtype=float).ravel())
    nset = {a.shape[0] for a in arrs}
    if len(nset) != 1:
        raise ValueError("All channels must have the same number of samples")
    x = np.vstack(arrs)

    # Sampling rate
    if fs is None:
        fss = {float(getattr(ch, "fs")) for ch in chs}
        if len(fss) != 1:
            raise ValueError("Channels must share the same sampling rate or pass `fs` explicitly")
        fs_use = fss.pop()
    else:
        fs_use = float(fs)

    # Detrend, notch (optional), bandpass, reref, zscore
    x = signal.detrend(x, axis=1, type="constant")
    if notch is not None:
        b, a = signal.iirnotch(notch / (fs_use / 2.0), 30.0)
        x = signal.filtfilt(b, a, x, axis=1)
        h = 2.0 * notch
        if h < fs_use * 0.49:
            b, a = signal.iirnotch(h / (fs_use / 2.0), 30.0)
            x = signal.filtfilt(b, a, x, axis=1)
    wp = (band[0] / (fs_use / 2.0), band[1] / (fs_use / 2.0))
    b, a = signal.butter(4, wp, btype="band")
    x = signal.filtfilt(b, a, x, axis=1)
    if reref == "avg":
        x = x - x.mean(axis=0, keepdims=True)
    if zscore:
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - m) / s

    # Write back to the channels
    for i, ch in enumerate(chs):
        ch.update_clean(x[i], fs_use)
    return eeg

def epoch_bandpower(eeg, fs, events, tmin, tmax, bands={"mu": (8, 12), "beta": (13, 30)}, nperseg=256):
    """
    Parameters
    ----------
    eeg : np.ndarray of Channel
        Array of Channel objects; each `Channel.clean` is used.
    fs : float | int | None
        Sampling rate in Hz. If None, inferred from channels' `fs` (must all match).
    events : numpy.ndarray | list[int]
    tmin : float
    tmax : float
    bands : dict[str, tuple[float, float]]
    nperseg : int
    Returns
    -------
    feats : numpy.ndarray
        Bandpower per epoch, shape (n_kept_epochs, n_channels * n_bands), dtype float.
    ev : numpy.ndarray
        Kept events (within bounds), shape (n_kept_epochs,), dtype int.
    """
    chs = np.asarray(eeg, dtype=object).ravel()
    # Stack clean signals
    arrs = []
    for ch in chs:
        sig = getattr(ch, "clean", None)
        if sig is None:
            raise ValueError("All channels must have a non-None `clean` array")
        arrs.append(np.asarray(sig, dtype=float).ravel())
    nset = {a.shape[0] for a in arrs}
    if len(nset) != 1:
        raise ValueError("All channels must have the same number of samples")
    x = np.vstack(arrs)

    # Sampling rate
    if fs is None:
        fss = {float(getattr(ch, "fs")) for ch in chs}
        if len(fss) != 1:
            raise ValueError("Channels must share the same sampling rate or pass `fs` explicitly")
        fs_use = fss.pop()
    else:
        fs_use = float(fs)

    events = np.asarray(events, dtype=int)
    pre = int(round(tmin * fs_use))
    post = int(round(tmax * fs_use))
    keep = (events + pre >= 0) & (events + post <= x.shape[1])
    ev = events[keep]

    feats = []
    for e in ev:
        seg = x[:, e + pre : e + post]
        f, pxx = signal.welch(seg, fs=fs_use, nperseg=min(nperseg, seg.shape[1]), axis=1, average="median")
        fb = []
        for lo, hi in bands.values():
            idx = (f >= lo) & (f <= hi)
            bp = np.trapz(pxx[:, idx], f[idx], axis=1)
            fb.append(np.log10(bp + 1e-12))
        fb = np.concatenate(fb, axis=0)
        feats.append(fb)
    feats = np.vstack(feats) if len(feats) else np.empty((0, x.shape[0] * len(bands)))
    return feats, ev