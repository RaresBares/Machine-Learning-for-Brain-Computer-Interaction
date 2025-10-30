import numpy as np
import matplotlib.pyplot as plt

from data import fetcher
from model.ProcChannel import ProcChannel, build_from_channel


def viz_channel(ch_raw, ch_proc, idx):
    # vor Proc (rohes Channel-Objekt)
    x_pre = np.asarray(ch_raw.clean, float)
    fs_pre = float(ch_raw.fs)
    t_pre = np.arange(x_pre.shape[0]) / fs_pre
    freqs_pre = np.asarray(ch_raw.freqs, float)
    mag_pre = np.abs(np.asarray(ch_raw.fft))

    # nach Proc (ProcChannel)
    x_post = np.asarray(ch_proc.clean, float)
    fs_post = float(ch_proc.fs)
    t_post = np.arange(x_post.shape[0]) / fs_post
    freqs_post = np.asarray(ch_proc.fft_freqs, float)
    mag_post = np.asarray(ch_proc.fft_mag, float)

    # Plot vor Proc: Zeit + Frequenz
    fig_pre, ax_pre = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    ax_pre[0].plot(t_pre, x_pre)
    ax_pre[0].set_xlabel('t [s]')
    ax_pre[0].set_ylabel('Amplitude')
    ax_pre[0].set_title(f'Channel {idx} vor Proc – Zeitbereich')

    ax_pre[1].semilogy(freqs_pre, mag_pre)
    ax_pre[1].set_xlabel('f [Hz]')
    ax_pre[1].set_ylabel('|FFT|')
    ax_pre[1].set_title(f'Channel {idx} vor Proc – Frequenzbereich')

    plt.tight_layout()

    # Plot nach Proc: Zeit + Frequenz
    fig_post, ax_post = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    ax_post[0].plot(t_post, x_post)
    ax_post[0].set_xlabel('t [s]')
    ax_post[0].set_ylabel('Amplitude (proc)')
    ax_post[0].set_title(f'Channel {idx} nach Proc – Zeitbereich')

    ax_post[1].semilogy(freqs_post, mag_post)
    ax_post[1].set_xlabel('f [Hz]')
    ax_post[1].set_ylabel('|FFT| (proc)')
    ax_post[1].set_title(f'Channel {idx} nach Proc – Frequenzbereich')

    plt.tight_layout()
    plt.show()
    # Skalarwerte in Console
    print(f'Channel {idx}')
    print(f'  delta_power: {ch_proc.delta:.6f}')
    print(f'  theta_power: {ch_proc.theta:.6f}')
    print(f'  alpha_power: {ch_proc.alpha:.6f}')
    print(f'  beta_power:  {ch_proc.beta:.6f}')
    print(f'  blink_power: {ch_proc.blink:.6f}')
    print(f'  energy:      {ch_proc.energy:.6f}')

def main():
    fs = 256
    t, raw_channels = fetcher.generate_bci_dummy_with_peaks(channels=3, seconds=5.0, fs=fs, seed=0)

    proc_channels = [build_from_channel(ch) for ch in raw_channels]

    for idx, (ch_raw, ch_proc) in enumerate(zip(raw_channels, proc_channels)):
        viz_channel(ch_raw, ch_proc, idx)

if __name__ == "__main__":
    main()