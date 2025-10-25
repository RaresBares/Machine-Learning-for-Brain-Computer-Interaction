from data.fetcher import generate_bci_dummy_with_peaks
from data.visualizer import Visualizer
from preprocessor.featurizer import EEGFeaturizer
from preprocessor.freq_preproc import preprocess_eeg_fft, epoch_bandpower_fft
from preprocessor.time_preproc import preprocess_eeg
import numpy as np

def to_2d_array(chs):
    if isinstance(chs, np.ndarray):
        return chs if chs.ndim == 2 else chs[None, :]
    return np.vstack([c for c in chs])

if __name__ == "__main__":
    _, chs = generate_bci_dummy_with_peaks(channels=8, seconds=5.0, fs=256, seed=0)
    viz = Visualizer()
    sel = to_2d_array([chs[0], chs[1],chs[2]])
    viz.time_channels(sel)
    viz.freq_channels(sel)

    sel = preprocess_eeg(sel, fs=256)

    viz.time_channels(sel, title="Time")
    viz.freq_channels(sel, title="PSD")

    sel = preprocess_eeg_fft(sel, fs=256)

    viz.time_channels(sel, title="Time")
    viz.freq_channels(sel, title="PSD")

    feat = EEGFeaturizer();

    fmap = feat.feature_map(sel, fs=256, events=[0], tmin=0.0, tmax=5.0)

    alpha_lin_ch1 = fmap["bandpower_per_band_lin"]["alpha"][0]  # linear
    alpha_log_ch1 = fmap["bandpower_per_band_log"]["alpha"][0]  # log10
    print("alpha_lin_ch1:", alpha_lin_ch1)
    print("alpha_log_ch1:", alpha_log_ch1)

    # --- More detailed outputs for channel 0 ---
    ch_idx = 0
    names = fmap.get("channel_names", [f"ch{i}" for i in range(sel.shape[0])])
    ch_name = names[ch_idx]
    bands = fmap["bands_order"]

    print(f"\nCHANNEL {ch_idx} ({ch_name}) — FULL-TRACE FEATURES")

    # Per-band linear and log powers
    for b in bands:
        val_lin = float(fmap["bandpower_per_band_lin"][b][ch_idx])
        val_log = float(fmap["bandpower_per_band_log"][b][ch_idx])
        print(f"  {b}: lin={val_lin:.6g}, log10={val_log:.6g}")

    # Ratios if available
    abr = fmap.get("alpha_beta_ratio")
    if abr is not None:
        print(f"  alpha/beta: {float(abr[ch_idx]):.6g}")
    tbr = fmap.get("theta_beta_ratio")
    if tbr is not None:
        print(f"  theta/beta: {float(tbr[ch_idx]):.6g}")

    # Flattened vector lookup by label
    labels = fmap["labels"]
    values = fmap["bandpower_log"]
    print("\nFlattened (bandpower_log) entries for this channel:")
    for b in bands:
        label = f"{ch_name}_{b}"
        try:
            idx = labels.index(label)
            print(f"  {label}: {float(values[idx]):.6g}")
        except ValueError:
            pass