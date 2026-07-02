"""
omegafm.dsp.resample
====================

Streaming 4x up / down converters bridging the 48 kHz multiband front
end and the 192 kHz emphasis / limiting / MPX back end.

Both use ~96-tap Kaiser FIRs (pass 20 kHz, stop ~26 kHz, >70 dB) run
through scipy.lfilter with preserved state so block boundaries are
seamless.  The upsampler zero-stuffs then filters (taps pre-scaled x4
to restore amplitude); the decimator filters then keeps every 4th
sample.  Phase alignment is enforced by requiring block sizes that are
multiples of the factor.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

FACTOR = 4


def _design_taps(fs_high: float) -> np.ndarray:
    # cutoff between 20k audio and 24k image, at the HIGH rate
    return firwin(97, 21600.0, window=("kaiser", 8.0), fs=fs_high)


class Upsampler4:
    def __init__(self, fs_in: float = 48000.0, channels: int = 2):
        self.channels = channels
        self.taps = _design_taps(fs_in * FACTOR) * FACTOR
        self.zi = np.zeros((len(self.taps) - 1, channels))

    @property
    def delay_out(self) -> int:
        return (len(self.taps) - 1) // 2

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        n = len(x)
        up = np.zeros((n * FACTOR, self.channels))
        up[::FACTOR, :] = x
        y, self.zi = lfilter(self.taps, [1.0], up, axis=0, zi=self.zi)
        return y

    def reset(self):
        self.zi[:] = 0.0


class Decimator4:
    def __init__(self, fs_in: float = 192000.0, channels: int = 2):
        self.channels = channels
        self.taps = _design_taps(fs_in)
        self.zi = np.zeros((len(self.taps) - 1, channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        assert len(x) % FACTOR == 0, "decimator blocks must be multiples of 4"
        y, self.zi = lfilter(self.taps, [1.0], x, axis=0, zi=self.zi)
        return y[::FACTOR, :]

    def reset(self):
        self.zi[:] = 0.0
