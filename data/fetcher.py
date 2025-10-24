import numpy as np

def generate_bci_dummy_with_peaks(channels=12, seconds=5.0, fs=256, seed=0):
    rng = np.random.RandomState(seed)
    n = int(seconds * fs)
    t = np.arange(n, dtype=np.float32) / float(fs)
    data = np.zeros((channels, n), dtype=np.float32)
    base_freqs = np.linspace(6.0, 30.0, channels).astype(np.float32)
    phi8 = rng.uniform(0, 2*np.pi, size=channels).astype(np.float32)
    phi12 = rng.uniform(0, 2*np.pi, size=channels).astype(np.float32)
    for ch in range(channels):
        f = float(base_freqs[ch])
        s_main = 12e-6*np.sin(2*np.pi*f*t)
        s_harm = 5e-6*np.sin(2*np.pi*(f+0.7)*t)
        s8 = 30e-6*np.sin(2*np.pi*8.0*t + phi8[ch])
        s12 = 24e-6*np.sin(2*np.pi*12.0*t + phi12[ch])
        drift = np.cumsum(rng.normal(0.0, 1e-8, size=n)).astype(np.float32)
        noise = rng.normal(0.0, 3e-6, size=n).astype(np.float32)
        data[ch] = s_main + s_harm + s8 + s12 + drift + noise
    return t, data