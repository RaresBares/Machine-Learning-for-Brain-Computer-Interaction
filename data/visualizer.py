import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    Minimal plotting helper for EEG/BCI or generic multichannel data.

    Attributes
    ----------
    figsize : tuple[float, float]
        Figure size for matplotlib plots.
    """
    def __init__(self, figsize=(10, 4)):
        """
        Parameters
        ----------
        figsize : tuple[float, float]
            Default figure size (width, height) for generated plots.
        """
        self.figsize = figsize

    def _to_2d(self, x):
        """
        Convert input array to 2D.
        Parameters
        ----------
        x : array-like of shape (n,) or (c, n)
            Input data. 1D arrays are expanded to shape (1, n).
        Returns
        -------
        np.ndarray of shape (c, n)
            2D float array representation.
        """
        X = np.asarray(x)
        if X.ndim == 1:
            X = X[None, :]
        return X

    def time(self, data, fs=None, t=None, labels=None, title=None, show=True):
        """
        Plot time-domain signals.

        Parameters
        ----------
        data : array-like of shape (n,) or (c, n)
            Signal data, where c is the number of channels and n is samples per channel.
        fs : float or int, optional
            Sampling rate in Hz. If None, x-axis is sample index.
        t : array-like of shape (n,), optional
            Optional custom time axis.
        labels : list[str], optional
            Labels for each channel (length c).
        title : str, optional
            Plot title.
        show : bool, default=True
            If True, calls plt.show() after plotting.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        X = self._to_2d(data)
        n = X.shape[1]
        if t is None:
            t = np.arange(n) if fs is None else np.arange(n) / float(fs)
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        if labels is None or X.shape[0] == 1:
            for i in range(X.shape[0]):
                ax.plot(t, X[i])
        else:
            for i in range(X.shape[0]):
                lab = str(labels[i]) if i < len(labels) else None
                ax.plot(t, X[i], label=lab)
            ax.legend(loc="best")
        ax.set_xlabel("t" + ("" if fs is None else " [s]"))
        ax.set_ylabel("amplitude")
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def freq(self, data, fs, labels=None, db=True, average=None, title=None, show=True):
        """
        Plot one-sided power spectral density (PSD) via Hann window and rFFT.

        Parameters
        ----------
        data : array-like of shape (n,) or (c, n)
            Time-domain signal(s).
        fs : float or int
            Sampling rate in Hz.
        labels : list[str], optional
            Labels for each channel.
        db : bool, default=True
            Whether to plot in dB scale (10*log10(PSD)).
        average : {"mean", "median", None}, optional
            If set, reduces multiple channels into one PSD by mean or median.
        title : str, optional
            Plot title.
        show : bool, default=True
            Whether to display the plot.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        X = self._to_2d(data)
        n = X.shape[1]
        window = np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        S = []
        for i in range(X.shape[0]):
            xw = X[i] * window
            fft = np.fft.rfft(xw)
            psd = (np.abs(fft) ** 2) / (np.sum(window**2) * fs)
            S.append(psd)
        S = np.vstack(S)
        if average == "mean":
            S = S.mean(axis=0, keepdims=True)
        elif average == "median":
            S = np.median(S, axis=0, keepdims=True)
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        if db:
            Y = 10.0 * np.log10(np.maximum(S, 1e-20))
            yl = "PSD [dB/Hz]"
        else:
            Y = S
            yl = "PSD [V²/Hz]"
        if labels is None or Y.shape[0] == 1:
            for i in range(Y.shape[0]):
                ax.plot(freqs, Y[i])
        else:
            for i in range(Y.shape[0]):
                lab = str(labels[i]) if i < len(labels) else None
                ax.plot(freqs, Y[i], label=lab)
            ax.legend(loc="best")
        ax.set_xlim(0, fs / 2)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(yl)
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def any(self, y, x=None, labels=None, title=None, xlabel=None, ylabel=None, show=True):
        """
        Generic line plot (y vs x).

        Parameters
        ----------
        y : array-like of shape (n,) or (c, n)
            Signal data.
        x : array-like of shape (n,), optional
            X-axis values.
        labels : list[str], optional
            Channel labels.
        title : str, optional
            Plot title.
        xlabel : str, optional
            Label for x-axis.
        ylabel : str, optional
            Label for y-axis.
        show : bool, default=True
            Whether to display the plot.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        Y = self._to_2d(y)
        if x is None:
            x = np.arange(Y.shape[1])
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        if labels is None or Y.shape[0] == 1:
            for i in range(Y.shape[0]):
                ax.plot(x, Y[i])
        else:
            for i in range(Y.shape[0]):
                lab = str(labels[i]) if i < len(labels) else None
                ax.plot(x, Y[i], label=lab)
            ax.legend(loc="best")
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if show:
            plt.show()
        return fig, ax

    def plot(self, data, fs=None, kind="time", **kwargs):
        """
        Unified plot entrypoint.

        Parameters
        ----------
        data : array-like of shape (n,) or (c, n)
            Input data.
        fs : float or int, optional
            Sampling rate, required for "freq".
        kind : {"time", "freq", "any"}, default="time"
            Plot type selector.
        **kwargs : dict
            Passed through to submethods.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        if kind == "time":
            return self.time(data, fs=fs, **kwargs)
        if kind == "freq":
            if fs is None:
                raise ValueError("fs required for freq")
            return self.freq(data, fs=fs, **kwargs)
        return self.any(data, **kwargs)

    def time_channels(self, channels, title=None, show=True):
        """
        Plot time-domain signals from Channel objects or numeric arrays.

        Parameters
        ----------
        channels : list[Channel] or array-like
            Each Channel must have `.clean` (shape (n,)) and `.fs` (float).
            Or numeric array (c,n) or (n,).
        title : str, optional
            Plot title.
        show : bool, default=True
            Whether to display the plot.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        arr = np.asarray(channels, dtype=object)
        # Case A: array/list of Channel objects
        if arr.dtype == object:
            flat = arr.ravel().tolist()
            xs = []
            labels = []
            for ch in flat:
                try:
                    sig = ch.clean
                except AttributeError:
                    sig = None
                if sig is not None:
                    xs.append(np.asarray(sig, dtype=float).ravel())
                    name = ch.name if hasattr(ch, "name") else None
                    labels.append(name)
            X = np.vstack(xs) if len(xs) else np.empty((0, 0))
            fs = flat[0].fs if len(flat) else None
            return self.time(X, fs=fs, labels=labels, title=title, show=show)
        # Case B: numeric array (c,n) or (n,)
        X = self._to_2d(np.asarray(channels, dtype=float))
        return self.time(X, fs=None, labels=None, title=title, show=show)

    def freq_channels(self, channels, db=True, title=None, show=True):
        """
        Plot PSDs from Channel objects.

        Parameters
        ----------
        channels : list[Channel]
            Each must have `.clean` (shape (n,)) and `.fs` (float).
        db : bool, default=True
            Plot in dB if True, else linear PSD.
        title : str, optional
            Plot title.
        show : bool, default=True
            Whether to display the plot.

        Returns
        -------
        (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Created matplotlib figure and axes.
        """
        arr = np.asarray(channels, dtype=object)
        if arr.dtype != object:
            raise TypeError("freq_channels expects Channel objects with .fs; for numeric arrays use Visualizer.freq(data, fs=...) directly")
        flat = arr.ravel().tolist()
        xs = []
        labels = []
        fs = flat[0].fs if len(flat) else None
        for ch in flat:
            if ch.clean is not None:
                xs.append(np.asarray(ch.clean, dtype=float).ravel())
                labels.append(ch.name if hasattr(ch, "name") else None)
        X = np.vstack(xs) if len(xs) else np.empty((0, 0))
        return self.freq(X, fs=fs, labels=labels, db=db, title=title, show=show)