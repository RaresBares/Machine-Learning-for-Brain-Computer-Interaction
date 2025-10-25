import numpy as np
from scipy import signal

class EEGFeaturizer:
    def __init__(self, bands=None):
        # Default canonical EEG bands
        self.bands = bands or {
            "delta": (1, 4),
            "theta": (4, 7),
            "alpha": (8, 12),
            "beta": (13, 30),
            "gamma": (31, 45),
        }

    # ---------- helpers ----------

    def _stack_channels(self, channels):
        """
        Accepts either Channel objects (with .clean, .fs, .name) or a numeric ndarray (C,N).
        Returns: (chs_or_none, X, names)
        """
        arr = np.asarray(channels, dtype=object).ravel()
        if arr.size == 0:
            raise ValueError("Empty channel list")

        # Try Channel path without getattr
        channel_like = True
        try:
            _ = arr[0].clean  # attribute existence check via direct access
        except AttributeError:
            channel_like = False

        if channel_like:
            chs = arr
            xs, names = [], []
            for i, ch in enumerate(chs):
                if ch.clean is None:
                    raise ValueError("Channel.clean is None")
                xs.append(np.asarray(ch.clean, dtype=float).ravel())
                nm = ch.name if isinstance(ch.name, str) and len(ch.name) > 0 else f"ch{i}"
                names.append(nm)
            nset = {a.shape[0] for a in xs}
            if len(nset) != 1:
                raise ValueError("All channels must have the same number of samples")
            X = np.vstack(xs)
            return chs, X, names

        # Numeric matrix path
        X = np.asarray(channels, dtype=float)
        if X.ndim != 2:
            raise ValueError("Numeric EEG input must be 2D (n_channels, n_samples)")
        names = [f"ch{i}" for i in range(X.shape[0])]
        return None, X, names

    def _fs(self, chs, fs):
        if fs is not None:
            return float(fs)
        if chs is None:
            raise ValueError("`fs` must be provided when passing numeric arrays")
        fss = {float(ch.fs) for ch in chs}
        if len(fss) != 1:
            raise ValueError("Channels must share the same sampling rate or pass fs explicitly")
        return fss.pop()

    # ---------- public API ----------

    def feature_map(self, channels, fs=None, events=None, tmin=0.0, tmax=None, nperseg=256):
        """
        Compute bandpower features over the **entire signal duration** (no epochs).

        Parameters
        ----------
        channels : list[Channel] | np.ndarray
            Channels (with .clean/.fs) or numeric matrix (C,N).
        fs : float | None
            Sampling rate in Hz. Required for numeric arrays; inferred from Channel objects.
        events, tmin, tmax : ignored
            Kept only for backward compatibility; the full trace is always used.
        nperseg : int
            Welch segment length.

        Returns dict with 1D/2D full-trace features (no epoch dimension):
          - channel_names: list[str]
          - bands_order: list[str]
          - labels: list[str]  # order matches bandpower_log vector
          - bandpower_per_band_lin: dict[band] -> (n_channels,)  # linear power
          - bandpower_per_band_log: dict[band] -> (n_channels,)  # log10 power
          - bandpower_log: (n_channels * n_bands,)  # flattened in labels order
          - alpha_beta_ratio: (n_channels,) or None
          - theta_beta_ratio: (n_channels,) or None
        """
        chs, X, names = self._stack_channels(channels)
        fs_use = self._fs(chs, fs)

        # Welch PSD over the **full trace** (no epoching)
        f, pxx = signal.welch(
            X, fs=fs_use,
            nperseg=min(nperseg, X.shape[1]),
            axis=1, average="median"
        )

        band_names = list(self.bands.keys())
        n_ch = X.shape[0]

        # Per-band power per channel (no epoch dimension)
        per_band_lin = {}
        per_band_log = {}
        for b in band_names:
            lo, hi = self.bands[b]
            idx = (f >= lo) & (f <= hi)
            bp_lin = np.trapz(pxx[:, idx], f[idx], axis=1)  # (n_channels,)
            per_band_lin[b] = bp_lin
            per_band_log[b] = np.log10(bp_lin + 1e-12)

        # Flattened vector (labels order: for each band, all channels)
        labels = [f"{names[ch]}_{b}" for b in band_names for ch in range(n_ch)]
        bandpower_log_vec = np.concatenate([per_band_log[b] for b in band_names], axis=0)

        alpha_beta_ratio = None
        if "alpha" in per_band_lin and "beta" in per_band_lin:
            alpha_beta_ratio = per_band_lin["alpha"] / (per_band_lin["beta"] + 1e-12)

        theta_beta_ratio = None
        if "theta" in per_band_lin and "beta" in per_band_lin:
            theta_beta_ratio = per_band_lin["theta"] / (per_band_lin["beta"] + 1e-12)

        return {
            "channel_names": names,
            "bands_order": band_names,
            "labels": labels,
            "bandpower_per_band_log": per_band_log,
            "bandpower_per_band_lin": per_band_lin,
            "bandpower_log": bandpower_log_vec,
            "alpha_beta_ratio": alpha_beta_ratio,
            "theta_beta_ratio": theta_beta_ratio,
        }

    def select(self, fmap, key):
        return fmap.get(key, None)

# ------------------------------------------------------------
# Example usages (no epochs; full-trace)
# ------------------------------------------------------------
# 1) Full-trace features from Channel objects
# feat = EEGFeaturizer()
# fmap = feat.feature_map(channels=chs, fs=None)  # fs inferred
# print(fmap["labels"])                 # e.g. ['ch0_delta','ch1_delta',...,'ch0_theta',...]
# print(fmap["bandpower_log"].shape)    # (n_channels * n_bands,)
# print(fmap["alpha_beta_ratio"].shape) # (n_channels,)
# alpha_ch0 = fmap["bandpower_per_band_lin"]["alpha"][0]

# 2) Numeric matrix input (no Channel objects)
# X = np.vstack([ch.clean for ch in chs])  # (C,N)
# fs = 256.0
# feat = EEGFeaturizer({"mu": (8,12), "beta": (13,30)})
# fmap = feat.feature_map(channels=X, fs=fs)
# mu_ch1 = fmap["bandpower_per_band_lin"]["mu"][1]

# 3) Flattened lookup by label
# labels = fmap["labels"]
# vals = fmap["bandpower_log"]
# i = labels.index("ch0_alpha")
# alpha_log_ch0 = vals[i]