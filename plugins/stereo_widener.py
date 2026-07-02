"""OmegaFM plugin: Stereo Widener.

Mid/side width control with a bass-mono guard (side content below the
MONO< frequency stays centred - essential for FM, where excessive LF
side energy wastes modulation and can break on mono radios).

Drop this file in the `plugins/` folder; it appears in the PLUGINS
window, inserts right after the input stage, and exposes WIDTH and
MONO< knobs plus a live L/R correlation meter.
"""

import numpy as np

PLUGIN = {
    "id": "stereo_widener",
    "name": "Stereo Widener",
    "version": "1.0",
    "insert": "post_input",
    "params": [
        {"key": "width", "label": "WIDTH", "min": 0.0, "max": 2.0,
         "default": 1.25, "fmt": "{:.2f}", "suffix": "x"},
        {"key": "mono_hz", "label": "MONO<", "min": 0.0, "max": 300.0,
         "default": 120.0, "fmt": "{:.0f}", "suffix": "Hz"},
    ],
    "meters": [
        {"key": "corr", "label": "CORR", "mode": "bipolar",
         "lo": -1.0, "hi": 1.0},
        {"key": "side_db", "label": "SIDE", "mode": "level",
         "lo": -42.0, "hi": 3.0},
    ],
}


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.width = 1.25
        self.mono_hz = 120.0
        self._a = 0.0
        self._set_split()
        self._lp = 0.0                       # one-pole state on the side
        self._corr = 1.0
        self.meters = {"corr": 1.0, "side_db": -80.0}

    def _set_split(self):
        hz = max(self.mono_hz, 1.0)
        self._a = float(np.exp(-2.0 * np.pi * hz / self.fs))

    def set_params(self, width=None, mono_hz=None):
        if width is not None:
            self.width = float(np.clip(width, 0.0, 2.0))
        if mono_hz is not None:
            self.mono_hz = float(np.clip(mono_hz, 0.0, 300.0))
            self._set_split()

    def process(self, x):
        L = x[:, 0]
        R = x[:, 1]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)

        if self.mono_hz >= 10.0:
            # one-pole LP extracts the low side; only the highs widen,
            # the low side is discarded (bass forced to mono)
            a = self._a
            b0 = 1.0 - a
            lo = np.empty_like(S)
            z = self._lp
            # scipy-free tiny recursion (S is a short block)
            for i in range(len(S)):
                z = b0 * S[i] + a * z
                lo[i] = z
            self._lp = z
            S = (S - lo) * self.width
        else:
            S = S * self.width

        L2 = M + S
        R2 = M - S

        # block correlation, smoothed for the meter
        el = float(np.dot(L2, L2)) + 1e-12
        er = float(np.dot(R2, R2)) + 1e-12
        c = float(np.dot(L2, R2)) / np.sqrt(el * er)
        self._corr = 0.8 * self._corr + 0.2 * c
        s_rms = float(np.sqrt(np.mean(S * S)) + 1e-12)
        self.meters["corr"] = self._corr
        self.meters["side_db"] = 20.0 * np.log10(s_rms)

        return np.stack([L2, R2], axis=1)

    def reset(self):
        self._lp = 0.0
        self._corr = 1.0
        self.meters = {"corr": 1.0, "side_db": -80.0}
