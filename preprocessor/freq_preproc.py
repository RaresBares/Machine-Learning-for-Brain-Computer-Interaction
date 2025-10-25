
import numpy as np
from scipy import signal

def preprocess_eeg_fft(eeg, fs=None, band=(1.0, 40.0), notch=None, notch_bw=1.0, reref="avg", zscore=True):
    """
    Parameters
    ----------
    eeg : numpy.ndarray | list[Channel]
        If ndarray: shape (n_channels, n_samples), dtype float.
        If list of Channel: each Channel.clean is used and updated.
    fs : float or int | None
        Sampling rate in Hz. If eeg is a list of Channel, fs may be None and will be inferred; otherwise required.
    band : tuple[float, float]
    notch : float | None
    notch_bw : float
    reref : str | None
    zscore : bool
    Returns
    -------
    numpy.ndarray | list[Channel]
        Same container type as input. If channels were passed, the same Channel objects are returned with `clean` updated.
    """
    chs_arr = np.asarray(eeg, dtype=object)
    if chs_arr.dtype == object:
        flat = chs_arr.ravel()
        arrs = []
        for ch in flat:
            sig = getattr(ch, "clean", None)
            if sig is None:
                raise ValueError("All channels must have a non-None `clean` array")
            arrs.append(np.asarray(sig, dtype=float).ravel())
        nset = {a.shape[0] for a in arrs}
        if len(nset) != 1:
            raise ValueError("All channels must have the same number of samples")
        x = np.vstack(arrs)
        if fs is None:
            fss = {float(getattr(ch, "fs")) for ch in flat}
            if len(fss) != 1:
                raise ValueError("Channels must share the same sampling rate or pass `fs` explicitly")
            fs_use = fss.pop()
        else:
            fs_use = float(fs)
    else:
        x = np.asarray(eeg, dtype=float)
        if fs is None:
            raise ValueError("fs must be provided when eeg is a numeric array")
        fs_use = float(fs)


    x = signal.detrend(x, axis=1, type="constant")
    n = x.shape[1]
    f = np.fft.rfftfreq(n, d=1.0 / fs_use)
    keep = (f >= band[0]) & (f <= band[1])
    X = np.fft.rfft(x, axis=1)
    mask = keep.copy()
    if notch is not None:
        nb = notch_bw * 0.5
        mask &= ~((f >= notch - nb) & (f <= notch + nb))
        h = 2.0 * notch
        if h < fs_use * 0.49:
            mask &= ~((f >= h - nb) & (f <= h + nb))
    X *= mask[None, :]
    x = np.fft.irfft(X, n=n, axis=1)
    if reref == "avg":
        x = x - x.mean(axis=0, keepdims=True)
    if zscore:
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - m) / s

    if chs_arr.dtype == object:
        flat = chs_arr.ravel()
        for i, ch in enumerate(flat):
            ch.update_clean(x[i], fs_use)
        return chs_arr
    return x

def epoch_bandpower_fft(eeg, fs, events, tmin, tmax, bands={"mu": (8, 12), "beta": (13, 30)}, window="hann"):
    """
    Parameters
    ----------
    eeg : numpy.ndarray | list[Channel]
        If ndarray: shape (n_channels, n_samples), dtype float.
        If list of Channel: each Channel.clean is used.
    fs : float or int | None
        Sampling rate in Hz. If eeg is a list of Channel and fs is None, it is inferred from channels (must all match).
    events : numpy.ndarray | list[int]
    tmin : float
    tmax : float
    bands : dict[str, tuple[float, float]]
    window : str
    Returns
    -------
    feats : numpy.ndarray
        Shape (n_kept_epochs, n_channels * n_bands), dtype float.
    ev : numpy.ndarray
        Events that were within bounds. Shape (n_kept_epochs,), dtype int.
    """
    chs = np.asarray(eeg, dtype=object).ravel()
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
        n = seg.shape[1]
        w = signal.get_window(window, n, fftbins=True)
        U = (w**2).sum() / n
        f = np.fft.rfftfreq(n, d=1.0 / fs_use)
        S = np.fft.rfft(seg * w, axis=1)
        Pxx = (np.abs(S) ** 2) / (fs_use * n * U)
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
