

import numpy as np

from preprocessor import time_preproc, freq_preproc, featurizer
from preprocessor.normalizer import Normalizer


def _apply_time_filters(channels, bp=None, notch=None, **kwargs):
    out = []
    for ch in channels:
        x = ch.clean
        fs = ch.fs
        y = x
        if hasattr(time_preproc, 'preprocess'):
            y = time_preproc.preprocess(x, fs=fs, bandpass=bp, notch=notch, **kwargs)
        else:
            if bp is not None and hasattr(time_preproc, 'bandpass'):
                y = time_preproc.bandpass(y, fs, bp[0], bp[1], **{k: v for k, v in kwargs.items() if k in ('order','ripple','atten')} )
            if notch is not None and hasattr(time_preproc, 'notch'):
                y = time_preproc.notch(y, fs, notch, **{k: v for k, v in kwargs.items() if k in ('q','order')} )
        ch.update_clean(y, fs)
        out.append(ch)
    return out


def _apply_freq_preproc(spec, freqs, **kwargs):
    if spec is None:
        return None
    if hasattr(freq_preproc, 'preprocess'):
        return freq_preproc.preprocess(spec, freqs=freqs, **kwargs)
    if hasattr(freq_preproc, 'postime_preprocrocess'):
        return freq_preproc.postime_preprocrocess(spec, freqs=freqs, **kwargs)
    return spec


def _fallback_mi_features(channels, spectrum, freqs):
    names = [getattr(c, 'name', None) for c in channels]
    idx_c3 = names.index('C3') if 'C3' in names else (0 if len(names) > 0 else None)
    idx_c4 = names.index('C4') if 'C4' in names else (1 if len(names) > 1 else None)
    if spectrum is None or freqs is None or idx_c3 is None or idx_c4 is None:
        return {}
    f = freqs
    def band_ix(a, b):
        return np.where((f >= a) & (f < b))[0]
    bix_mu = band_ix(8.0, 13.0)
    bix_be = band_ix(13.0, 30.0)
    psd = np.asarray(spectrum)
    def bp(ch, bix):
        v = psd[ch, bix]
        s = np.sum(v)
        return np.log1p(max(s, 0.0))
    mu_c3 = bp(idx_c3, bix_mu)
    mu_c4 = bp(idx_c4, bix_mu)
    be_c3 = bp(idx_c3, bix_be)
    be_c4 = bp(idx_c4, bix_be)
    return {
        'mu_c3': float(mu_c3),
        'mu_c4': float(mu_c4),
        'beta_c3': float(be_c3),
        'beta_c4': float(be_c4),
        'delta_mu': float(mu_c3 - mu_c4),
        'delta_beta': float(be_c3 - be_c4),
        'ratio_mu': float((mu_c3 - mu_c4) / (mu_c3 + mu_c4 + 1e-6)),
        'ratio_beta': float((be_c3 - be_c4) / (be_c3 + be_c4 + 1e-6)),
    }


def preprocess(channels, bp=(0.5, 40.0), notch=None, normalize=True, log_spectrum=True, log_features=True, use_raw=True, use_psd=True, **kwargs):
    ch_filt = _apply_time_filters(list(channels), bp=bp, notch=notch, **kwargs)
    norm = channels##Normalizer.normalize(ch_filt, use_raw=use_raw, use_psd=use_psd, eps=1e-6, log_spectrum=log_spectrum, log_features=log_features)
    Xr, Xs, freqs = norm.transform_channels(ch_filt, use_raw=use_raw, use_psd=use_psd, return_freqs=True)
    Xs = _apply_freq_preproc(Xs, freqs, **kwargs)
    feats = None
    if hasattr(featurizer, 'extract_mi_features'):
        try:
            feats = featurizer.extract_mi_features(channels=ch_filt, spectrum=Xs, freqs=freqs, raw=Xr)
        except Exception:
            feats = None
    if feats is None and hasattr(featurizer, 'featurize'):
        try:
            feats = featurizer.featurize(channels=ch_filt, spectrum=Xs, freqs=freqs, raw=Xr)
        except Exception:
            feats = None
    if feats is None:
        feats = _fallback_mi_features(ch_filt, Xs, freqs)
    out = {
        'channels': ch_filt,
        'raw': Xr,
        'spectrum': Xs,
        'freqs': freqs,
        'features': feats,
        'normalizer_state': norm.get_state(),
    }
    return out