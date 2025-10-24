import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, figsize=(10, 4)):
        self.figsize = figsize

    def _to_2d(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        return x

    def plot_time(self, data, fs=None, t=None, ch_names=None, overlay=False, title=None):
        X = self._to_2d(data)
        n = X.shape[1]
        if t is None:
            if fs is None:
                t = np.arange(n)
            else:
                t = np.arange(n) / float(fs)
        fig = plt.figure(figsize=self.figsize)
        if overlay:
            ax = fig.add_subplot(111)
            for i in range(X.shape[0]):
                label = None if ch_names is None else str(ch_names[i])
                ax.plot(t, X[i], label=label)
            if ch_names is not None:
                ax.legend(loc="best")
            ax.set_xlabel("t" + ("" if fs is None else " [s]"))
            ax.set_ylabel("amplitude")
            if title is not None:
                ax.set_title(title)
            return fig, ax
        axes = []
        for i in range(X.shape[0]):
            ax = fig.add_subplot(X.shape[0], 1, i + 1)
            ax.plot(t, X[i])
            ax.set_xlabel("t" + ("" if fs is None else " [s]"))
            ax.set_ylabel("ch {}".format(i if ch_names is None else ch_names[i]))
            axes.append(ax)
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axes

    def plot_freq(self, data, fs, overlay=False, db=True, average=None, title=None):
        X = self._to_2d(data)
        n = X.shape[1]
        window = np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        spectra = []
        for i in range(X.shape[0]):
            xw = X[i] * window
            fft = np.fft.rfft(xw)
            psd = (np.abs(fft) ** 2) / (np.sum(window**2) * fs)
            spectra.append(psd)
        S = np.vstack(spectra)
        if average == "mean":
            S = S.mean(axis=0, keepdims=True)
        elif average == "median":
            S = np.median(S, axis=0, keepdims=True)
        fig = plt.figure(figsize=self.figsize)
        if db:
            Splot = 10.0 * np.log10(np.maximum(S, 1e-20))
            ylabel = "PSD [dB/Hz]"
        else:
            Splot = S
            ylabel = "PSD [V²/Hz]"
        if overlay or Splot.shape[0] == 1:
            ax = fig.add_subplot(111)
            for i in range(Splot.shape[0]):
                ax.plot(freqs, Splot[i])
            ax.set_xlim(0, fs/2)
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title)
            return fig, ax
        axes = []
        for i in range(Splot.shape[0]):
            ax = fig.add_subplot(Splot.shape[0], 1, i + 1)
            ax.plot(freqs, Splot[i])
            ax.set_xlim(0, fs/2)
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(ylabel)
            axes.append(ax)
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axes

    def plot_any(self, x, y=None, title=None, xlabel=None, ylabel=None):
        x = np.asarray(x)
        if y is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(x)
        else:
            y = np.asarray(y)
            if y.ndim == 1:
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(x, y)
            else:
                fig, ax = plt.subplots(figsize=self.figsize)
                for i in range(y.shape[0]):
                    ax.plot(x, y[i])
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return fig, ax


if __name__ == "__main__":
    fs = 256
    seconds = 5.0
    channels = 3

    rng = np.random.RandomState(0)
    n = int(seconds * fs)
    t = np.arange(n, dtype=np.float32) / float(fs)
    data = np.zeros((channels, n), dtype=np.float32)

    base_freqs = np.linspace(6.0, 30.0, channels).astype(np.float32)
    phi8 = rng.uniform(0, 2*np.pi, size=channels).astype(np.float32)
    phi12 = rng.uniform(0, 2*np.pi, size=channels).astype(np.float32)

    for ch in range(channels):
        f = float(base_freqs[ch])
        s_main = 12e-6 * np.sin(2*np.pi * f * t)
        s_harm = 5e-6 * np.sin(2*np.pi * (f + 0.7) * t)
        s8 = 30e-6 * np.sin(2*np.pi * 8.0 * t + phi8[ch])
        s12 = 24e-6 * np.sin(2*np.pi * 12.0 * t + phi12[ch])
        drift = np.cumsum(rng.normal(0.0, 1e-8, size=n)).astype(np.float32)
        noise = rng.normal(0.0, 3e-6, size=n).astype(np.float32)
        data[ch] = s_main + s_harm + s8 + s12 + drift + noise

    viz = Visualizer(figsize=(12, 5))
    viz.plot_time(data, fs=fs, overlay=True, title="Time domain")
    plt.show()

    viz.plot_freq(data, fs=fs, overlay=True, db=True, title="Frequency domain")
    plt.show()