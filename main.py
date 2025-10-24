from data.fetcher import generate_bci_dummy_with_peaks
from data.visualizer import Visualizer

if __name__ == "__main__":
    _, chs = generate_bci_dummy_with_peaks(channels=8, seconds=5.0, fs=256, seed=0)
    viz = Visualizer()
    sel = [chs[i] for i in (0, 2, 4)]
    viz.time_channels(sel, title="Time")
    viz.freq_channels(sel, title="PSD")
