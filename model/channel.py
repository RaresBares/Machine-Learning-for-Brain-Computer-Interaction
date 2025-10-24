import numpy as np

class Channel:
    """
    Lightweight container for one EEG/BCI channel.

    Attributes
    ----------
    fs : float
        Sampling rate in Hz.
    name : str | None
        Optional display name.
    clean : np.ndarray | None
        Time-domain signal, shape (n,), dtype float.
    freqs : np.ndarray | None
        One-sided frequency axis for rFFT, shape (n_rfft,), dtype float.
    fft : np.ndarray | None
        One-sided complex FFT (rFFT) of `clean`, shape (n_rfft,), dtype complex.
    """

    def __init__(self, fs, clean=None, name=None):
        """
        Parameters
        ----------
        fs : float | int
            Sampling rate in Hz.
        clean : array-like | None
            Optional initial time-domain signal, shape (n,). If provided, `clean`, `fft`, and `freqs` are computed.
        name : str | None
            Optional channel name.
        """
        self.fs = float(fs)
        self.name = name
        self.clean = None
        self.freqs = None
        self.fft = None
        if clean is not None:
            self.update_clean(clean, fs)

    def update(self, signal, fs=None):
        """
        Alias for `update_clean`.

        Parameters
        ----------
        signal : array-like
            Time-domain signal, shape (n,).
        fs : float | int | None
            Sampling rate in Hz. If None, keep current `fs`.

        Returns
        -------
        Channel
            Self.
        """
        self.update_clean(signal, fs)
        return self

    def update_clean(self, signal, fs=None):
        """
        Update from a time-domain signal and recompute FFT and frequency axis.

        Parameters
        ----------
        signal : array-like
            Time-domain signal, shape (n,), dtype cast to float.
        fs : float | int | None
            Sampling rate in Hz. If provided, overrides existing `fs`.

        Sets
        ----
        clean : np.ndarray
            Shape (n,), dtype float.
        fft : np.ndarray
            rFFT of `clean`, shape (n_rfft,), dtype complex.
        freqs : np.ndarray
            rFFT frequency axis, shape (n_rfft,), dtype float.

        Returns
        -------
        Channel
            Self.
        """
        if fs is not None:
            self.fs = float(fs)
        x = np.asarray(signal, dtype=float).ravel()
        self.clean = x
        n = x.shape[0]
        self.freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        self.fft = np.fft.rfft(x)
        return self

    def update_fft(self, fft_values, fs=None, n=None):
        """
        Update from a one-sided rFFT vector; reconstruct the time signal.

        Parameters
        ----------
        fft_values : array-like
            One-sided complex FFT (np.fft.rfft convention), shape (n_rfft,), dtype complex.
        fs : float | int | None
            Sampling rate in Hz. If provided, overrides existing `fs`.
        n : int | None
            Target time-domain length. If None, uses len(clean) if available, else `2*(len(fft_values)-1)`.

        Sets
        ----
        fft : np.ndarray
            Stored rFFT.
        freqs : np.ndarray
            rFFT frequency axis, shape (n_rfft,), dtype float.
        clean : np.ndarray
            Reconstructed time-domain signal via irFFT, shape (n,), dtype float.

        Returns
        -------
        Channel
            Self.
        """
        if fs is not None:
            self.fs = float(fs)
        self.fft = np.asarray(fft_values)
        if n is None:
            if self.clean is not None:
                n = self.clean.shape[0]
            else:
                n = 2 * (len(self.fft) - 1)
        self.freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        self.clean = np.fft.irfft(self.fft, n=n)
        return self

    def set_clean(self, signal):
        """
        Alias for `update_clean`.

        Parameters
        ----------
        signal : array-like
            Time-domain signal, shape (n,).

        Returns
        -------
        Channel
            Self.
        """
        return self.update_clean(signal)

    def set_fft(self, fft_values, fs=None, n=None):
        """
        Alias for `update_fft`.

        Parameters
        ----------
        fft_values : array-like
            One-sided complex FFT, shape (n_rfft,).
        fs : float | int | None
            Sampling rate in Hz.
        n : int | None
            Time-domain length for reconstruction.

        Returns
        -------
        Channel
            Self.
        """
        return self.update_fft(fft_values, fs, n)

    def get_psd(self):
        """
        Return Welch-equivalent single-segment Hann-window PSD of `clean`.

        Returns
        -------
        (freqs, psd) : tuple[np.ndarray, np.ndarray] | (None, None)
            freqs: shape (n_rfft,), Hz; psd: shape (n_rfft,), V^2/Hz. Returns (None, None) if `clean` is None.
        """
        if self.clean is None:
            return None, None
        n = self.clean.shape[0]
        w = np.hanning(n)
        xw = self.clean * w
        f = np.fft.rfft(xw)
        psd = (np.abs(f) ** 2) / (np.sum(w ** 2) * self.fs)
        return self.freqs, psd