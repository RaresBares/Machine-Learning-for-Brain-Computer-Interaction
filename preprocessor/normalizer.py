

import numpy as np

class Normalizer(object):
    def __init__(self, eps=1e-6, log_spectrum=True, log_features=True, spectrum_layout='auto'):
        self.eps = float(eps)
        self.log_spectrum = bool(log_spectrum)
        self.log_features = bool(log_features)
        self.spectrum_layout = spectrum_layout
        self.raw_mean = None
        self.raw_std = None
        self.spec_mean = None
        self.spec_std = None
        self.feat_mean = None
        self.feat_std = None
        self.feature_keys = None
        self.freq_axis = None

    def _nan_to_num(self, x):
        return np.nan_to_num(np.asarray(x))

    def _extract_from_channels(self, channels, compute_raw=True, compute_psd=True):
        xs = []
        fs = None
        for ch in channels or []:
            if ch is None:
                continue
            x = getattr(ch, 'clean', None)
            if x is None:
                continue
            if fs is None:
                fs = float(getattr(ch, 'fs', 0.0))
            x = np.asarray(x, dtype=float).ravel()
            xs.append(x)
        if not xs:
            return None, None, None
        n = min(len(x) for x in xs)
        Xr = np.stack([x[-n:] for x in xs], axis=0) if compute_raw else None
        Xs = None
        freqs = None
        if compute_psd:
            w = np.hanning(n)
            s = 1.0 / (np.sum(w ** 2) * fs) if fs and fs > 0 else 1.0
            freqs = np.fft.rfftfreq(n, d=1.0 / fs) if fs and fs > 0 else None
            spec = []
            for x in [x[-n:] for x in xs]:
                f = np.fft.rfft(x * w)
                psd = (np.abs(f) ** 2) * s
                spec.append(psd)
            Xs = np.stack(spec, axis=0)
        self.freq_axis = freqs
        return Xr, Xs, freqs

    def _infer_spec_layout(self, x):
        if self.spectrum_layout in ('CF', 'NF'):
            return self.spectrum_layout
        if x.ndim == 1:
            return 'NF'
        if x.ndim >= 2:
            if x.shape[-1] >= 8:
                if x.shape[-2] <= 64:
                    return 'CF'
                return 'NF'
        return 'NF'

    def fit_from_channels(self, channels, use_raw=True, use_psd=True):
        Xr, Xs, _ = self._extract_from_channels(channels, compute_raw=use_raw, compute_psd=use_psd)
        if use_raw and Xr is not None:
            self._fit_raw(Xr)
        if use_psd and Xs is not None:
            self._fit_spectrum(Xs)
            self.spectrum_layout = 'CF'
        return self

    def transform_channels(self, channels, use_raw=True, use_psd=True, return_freqs=False):
        Xr, Xs, freqs = self._extract_from_channels(channels, compute_raw=use_raw, compute_psd=use_psd)
        Yr = self._transform_raw(Xr) if use_raw and Xr is not None else None
        Ys = self._transform_spectrum(Xs) if use_psd and Xs is not None else None
        if return_freqs:
            return Yr, Ys, freqs
        return Yr, Ys

    @classmethod
    def normalize(cls, channels, use_raw=True, use_psd=True, eps=1e-6, log_spectrum=True, log_features=True):
        obj = cls(eps=eps, log_spectrum=log_spectrum, log_features=log_features, spectrum_layout='CF' if use_psd else 'auto')
        obj.fit_from_channels(channels, use_raw=use_raw, use_psd=use_psd)
        return obj

    def _fit_raw(self, raw):
        x = self._nan_to_num(raw)
        if x.ndim == 1:
            x = x[np.newaxis, np.newaxis, :]
        elif x.ndim == 2:
            x = x[np.newaxis, :, :]
        else:
            b = int(np.prod(x.shape[:-2]))
            x = x.reshape(b, x.shape[-2], x.shape[-1])
        c = x.shape[1]
        xr = x.transpose(1, 0, 2).reshape(c, -1)
        m = xr.mean(axis=1)
        s = xr.std(axis=1)
        s[s == 0] = 1.0
        self.raw_mean = m
        self.raw_std = s

    def _fit_spectrum(self, spec):
        x = self._nan_to_num(spec)
        if self.log_spectrum:
            x = np.log1p(np.maximum(x, 0.0))
        layout = self._infer_spec_layout(x)
        if layout == 'CF':
            if x.ndim == 2:
                x = x[np.newaxis, :, :]
            else:
                b = int(np.prod(x.shape[:-2]))
                x = x.reshape(b, x.shape[-2], x.shape[-1])
            m = x.mean(axis=0)
            s = x.std(axis=0)
        else:
            if x.ndim == 1:
                x = x[np.newaxis, :]
            else:
                b = int(np.prod(x.shape[:-1]))
                x = x.reshape(b, x.shape[-1])
            m = x.mean(axis=0)
            s = x.std(axis=0)
        s[s == 0] = 1.0
        self.spec_mean = m
        self.spec_std = s
        self.spectrum_layout = layout

    def _features_to_array(self, features, fit_mode=False):
        if features is None:
            return None
        if isinstance(features, dict):
            if fit_mode or self.feature_keys is None:
                keys = sorted(features.keys())
                self.feature_keys = keys
            else:
                keys = self.feature_keys
            vals = [features[k] if k in features else 0.0 for k in keys]
            x = np.asarray(vals, dtype=float)
            return x
        return self._nan_to_num(features)

    def _array_to_features(self, arr, template):
        if isinstance(template, dict):
            out = {}
            keys = self.feature_keys if self.feature_keys is not None else sorted(template.keys())
            for i, k in enumerate(keys):
                out[k] = float(arr[i])
            return out
        return arr

    def _fit_features(self, features):
        x = self._features_to_array(features, fit_mode=True)
        if x is None:
            return
        if self.log_features:
            x = np.log1p(np.maximum(x, 0.0))
        if x.ndim == 1:
            x = x[np.newaxis, :]
        else:
            b = int(np.prod(x.shape[:-1]))
            x = x.reshape(b, x.shape[-1])
        m = x.mean(axis=0)
        s = x.std(axis=0)
        s[s == 0] = 1.0
        self.feat_mean = m
        self.feat_std = s

    def fit(self, raw=None, spectrum=None, features=None):
        if raw is not None:
            self._fit_raw(raw)
        if spectrum is not None:
            self._fit_spectrum(spectrum)
        if features is not None:
            self._fit_features(features)
        return self

    def partial_fit(self, raw=None, spectrum=None, features=None, momentum=0.1):
        if raw is not None:
            x = self._nan_to_num(raw)
            if x.ndim == 1:
                x = x[np.newaxis, np.newaxis, :]
            elif x.ndim == 2:
                x = x[np.newaxis, :, :]
            else:
                b = int(np.prod(x.shape[:-2]))
                x = x.reshape(b, x.shape[-2], x.shape[-1])
            c = x.shape[1]
            xr = x.transpose(1, 0, 2).reshape(c, -1)
            m = xr.mean(axis=1)
            v = xr.var(axis=1)
            if self.raw_mean is None:
                self.raw_mean = m
                self.raw_std = np.sqrt(v)
                self.raw_std[self.raw_std == 0] = 1.0
            else:
                self.raw_mean = (1.0 - momentum) * self.raw_mean + momentum * m
                new_std = np.sqrt((1.0 - momentum) * (self.raw_std ** 2) + momentum * v)
                new_std[new_std == 0] = 1.0
                self.raw_std = new_std
        if spectrum is not None:
            x = self._nan_to_num(spectrum)
            if self.log_spectrum:
                x = np.log1p(np.maximum(x, 0.0))
            layout = self._infer_spec_layout(x)
            if layout == 'CF':
                if x.ndim == 2:
                    x = x[np.newaxis, :, :]
                else:
                    b = int(np.prod(x.shape[:-2]))
                    x = x.reshape(b, x.shape[-2], x.shape[-1])
                m = x.mean(axis=0)
                v = x.var(axis=0)
            else:
                if x.ndim == 1:
                    x = x[np.newaxis, :]
                else:
                    b = int(np.prod(x.shape[:-1]))
                    x = x.reshape(b, x.shape[-1])
                m = x.mean(axis=0)
                v = x.var(axis=0)
            if self.spec_mean is None:
                self.spec_mean = m
                self.spec_std = np.sqrt(v)
                self.spec_std[self.spec_std == 0] = 1.0
                self.spectrum_layout = layout
            else:
                self.spec_mean = (1.0 - momentum) * self.spec_mean + momentum * m
                new_std = np.sqrt((1.0 - momentum) * (self.spec_std ** 2) + momentum * v)
                new_std[new_std == 0] = 1.0
                self.spec_std = new_std
                self.spectrum_layout = layout
        if features is not None:
            x = self._features_to_array(features, fit_mode=True)
            if x is not None:
                if self.log_features:
                    x = np.log1p(np.maximum(x, 0.0))
                if x.ndim == 1:
                    x = x[np.newaxis, :]
                else:
                    b = int(np.prod(x.shape[:-1]))
                    x = x.reshape(b, x.shape[-1])
                m = x.mean(axis=0)
                v = x.var(axis=0)
                if self.feat_mean is None:
                    self.feat_mean = m
                    self.feat_std = np.sqrt(v)
                    self.feat_std[self.feat_std == 0] = 1.0
                else:
                    self.feat_mean = (1.0 - momentum) * self.feat_mean + momentum * m
                    new_std = np.sqrt((1.0 - momentum) * (self.feat_std ** 2) + momentum * v)
                    new_std[new_std == 0] = 1.0
                    self.feat_std = new_std
        return self

    def _transform_raw(self, raw):
        if self.raw_mean is None or self.raw_std is None or raw is None:
            return raw
        x = self._nan_to_num(raw)
        if x.ndim == 1:
            x = x[np.newaxis, np.newaxis, :]
            orig = '1d'
        elif x.ndim == 2:
            x = x[np.newaxis, :, :]
            orig = '2d'
        else:
            b = int(np.prod(x.shape[:-2]))
            x = x.reshape(b, x.shape[-2], x.shape[-1])
            orig = 'nd'
        m = self.raw_mean.reshape(1, -1, 1)
        s = self.raw_std.reshape(1, -1, 1)
        y = (x - m) / (s + self.eps)
        if orig == '1d':
            return y.reshape(y.shape[-1])
        if orig == '2d':
            return y.reshape(y.shape[1], y.shape[2])
        return y.reshape(raw.shape)

    def _transform_spectrum(self, spec):
        if self.spec_mean is None or self.spec_std is None or spec is None:
            return spec
        x = self._nan_to_num(spec)
        if self.log_spectrum:
            x = np.log1p(np.maximum(x, 0.0))
        layout = self._infer_spec_layout(x)
        if layout == 'CF':
            if x.ndim == 2:
                x = x[np.newaxis, :, :]
                orig = '2d'
            else:
                b = int(np.prod(x.shape[:-2]))
                x = x.reshape(b, x.shape[-2], x.shape[-1])
                orig = 'nd'
            m = self.spec_mean.reshape(1, self.spec_mean.shape[0], self.spec_mean.shape[1])
            s = self.spec_std.reshape(1, self.spec_std.shape[0], self.spec_std.shape[1])
            y = (x - m) / (s + self.eps)
            if orig == '2d':
                return y.reshape(y.shape[1], y.shape[2])
            return y.reshape(spec.shape)
        else:
            if x.ndim == 1:
                x = x[np.newaxis, :]
                orig = '1d'
            else:
                b = int(np.prod(x.shape[:-1]))
                x = x.reshape(b, x.shape[-1])
                orig = 'nd'
            m = self.spec_mean.reshape(1, self.spec_mean.shape[0])
            s = self.spec_std.reshape(1, self.spec_std.shape[0])
            y = (x - m) / (s + self.eps)
            if orig == '1d':
                return y.reshape(y.shape[-1])
            return y.reshape(spec.shape)

    def _transform_features(self, features):
        if features is None or self.feat_mean is None or self.feat_std is None:
            return features
        arr = self._features_to_array(features, fit_mode=False)
        if self.log_features:
            arr = np.log1p(np.maximum(arr, 0.0))
        if arr.ndim == 1:
            x = arr[np.newaxis, :]
            orig = '1d'
        else:
            b = int(np.prod(arr.shape[:-1]))
            x = arr.reshape(b, arr.shape[-1])
            orig = 'nd'
        m = self.feat_mean.reshape(1, self.feat_mean.shape[0])
        s = self.feat_std.reshape(1, self.feat_std.shape[0])
        y = (x - m) / (s + self.eps)
        y = y.reshape(arr.shape) if orig != '1d' else y.reshape(y.shape[-1])
        return self._array_to_features(y, features)

    def transform(self, raw=None, spectrum=None, features=None):
        return self._transform_raw(raw), self._transform_spectrum(spectrum), self._transform_features(features)

    def fit_transform(self, raw=None, spectrum=None, features=None):
        self.fit(raw=raw, spectrum=spectrum, features=features)
        return self.transform(raw=raw, spectrum=spectrum, features=features)

    def get_state(self):
        return {
            'eps': self.eps,
            'log_spectrum': self.log_spectrum,
            'log_features': self.log_features,
            'spectrum_layout': self.spectrum_layout,
            'raw_mean': None if self.raw_mean is None else self.raw_mean.tolist(),
            'raw_std': None if self.raw_std is None else self.raw_std.tolist(),
            'spec_mean': None if self.spec_mean is None else self._tolist(self.spec_mean),
            'spec_std': None if self.spec_std is None else self._tolist(self.spec_std),
            'feat_mean': None if self.feat_mean is None else self.feat_mean.tolist(),
            'feat_std': None if self.feat_std is None else self.feat_std.tolist(),
            'feature_keys': self.feature_keys,
        }

    def set_state(self, state):
        self.eps = float(state.get('eps', 1e-6))
        self.log_spectrum = bool(state.get('log_spectrum', True))
        self.log_features = bool(state.get('log_features', True))
        self.spectrum_layout = state.get('spectrum_layout', 'auto')
        self.raw_mean = self._fromlist(state.get('raw_mean'))
        self.raw_std = self._fromlist(state.get('raw_std'))
        self.spec_mean = self._fromlist(state.get('spec_mean'))
        self.spec_std = self._fromlist(state.get('spec_std'))
        self.feat_mean = self._fromlist(state.get('feat_mean'))
        self.feat_std = self._fromlist(state.get('feat_std'))
        self.feature_keys = state.get('feature_keys')
        return self

    def _tolist(self, x):
        if x is None:
            return None
        return np.asarray(x).tolist()

    def _fromlist(self, x):
        if x is None:
            return None
        return np.asarray(x, dtype=float)