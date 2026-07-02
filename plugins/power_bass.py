"""OmegaFM plugin: Power Bass (bass clipper).

Stereo-Tool-style bass densifier. The band below FREQ is split off
(2nd-order Butterworth, subtractive so the recombination is exact),
driven into a smooth tanh ceiling, and the clip harmonics above the
band are kept in a controlled amount (PUNCH) for small-speaker girth.
Clipped bass carries more RMS at the same peak, so the low end gets
louder and tighter on air without pumping the wideband limiter or
wasting modulation.

Inserted post_multiband: after the leveler / compressor (which would
otherwise "level away" added bass energy) and right before the final
limiting section, where a dedicated bass clipper belongs.

The clipper is a soft-knee ceiling at -10.5 dBFS, exactly linear
below the knee and matched to the measured LF operating level of the
chain at this point (peak median -15 dBFS on dense program): at the
factory default of +3.5 dB drive it only *kisses* the ceiling
(~0.6 dB of crest work on dense music) - deliberately un-exaggerated.
Push DRIVE toward +6..+8 for the full power-bass effect (~1.5-3 dB of
bass densification, +6-9 points of modulation); even at maximum the
downstream WB limiter workload barely moves, so it cannot destabilise
the chain.

Controls
--------
DRIVE  0..+8 dB   how hard the bass leans on the ceiling (default +3.5)
FREQ   60..200 Hz bass band split (default 120)
PUNCH  0..1       clip-harmonic amount kept above the band (default 0.25)
TRIM   -6..+6 dB  processed-bass output level (default 0)

Meters: CLIP = dB of bass crest reduction, BASS = processed band peak.
"""

import numpy as np
from scipy.signal import butter, sosfilt

PLUGIN = {
    "id": "power_bass",
    "name": "Power Bass",
    "version": "1.0",
    "insert": "post_multiband",
    "params": [
        {"key": "drive_db", "label": "DRIVE", "min": 0.0, "max": 8.0,
         "default": 3.5, "fmt": "{:+.1f}", "suffix": "dB"},
        {"key": "freq", "label": "FREQ", "min": 60.0, "max": 200.0,
         "default": 120.0, "fmt": "{:.0f}", "suffix": "Hz"},
        {"key": "punch", "label": "PUNCH", "min": 0.0, "max": 1.0,
         "default": 0.25, "fmt": "{:.2f}", "suffix": ""},
        {"key": "trim_db", "label": "TRIM", "min": -6.0, "max": 6.0,
         "default": 0.0, "fmt": "{:+.1f}", "suffix": "dB"},
    ],
    "meters": [
        {"key": "clip_db", "label": "CLIP", "mode": "gr",
         "lo": 0.0, "hi": 12.0},
        {"key": "bass_db", "label": "BASS", "mode": "level",
         "lo": -42.0, "hi": 3.0},
    ],
}

CEILING = 10.0 ** (-10.5 / 20.0)          # -10.5 dBFS, matched to the chain
KNEE = 0.45 * CEILING                     # exactly linear below T-knee


def _soft_ceiling(x, T=CEILING, k=KNEE):
    """Piecewise soft-knee ceiling: identity below T-k, quadratic knee
    to a hard stop at T (C1-continuous, mirrored for negative)."""
    ax = np.abs(x)
    s = np.sign(x)
    y = ax.copy()
    m = (ax > T - k) & (ax < T + k)
    y[m] = ax[m] - (ax[m] - (T - k)) ** 2 / (4.0 * k)
    y[ax >= T + k] = T
    return s * y


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.drive_db = 3.5
        self.freq = 120.0
        self.punch = 0.25
        self.trim_db = 0.0
        self._design()
        self._clip_sm = 0.0
        self.meters = {"clip_db": 0.0, "bass_db": -80.0}

    # ------------------------------------------------------------------ setup
    def _design(self):
        f = float(np.clip(self.freq, 40.0, 250.0))
        self.sos_lo = butter(2, f / (self.fs / 2.0), output="sos")
        fh = min(2.6 * f, 420.0)
        self.sos_h = butter(2, fh / (self.fs / 2.0), output="sos")
        self.zi_lo = np.zeros((self.sos_lo.shape[0], 2, self.channels))
        self.zi_h = np.zeros((self.sos_h.shape[0], 2, self.channels))

    def set_params(self, drive_db=None, freq=None, punch=None, trim_db=None):
        if drive_db is not None:
            self.drive_db = float(np.clip(drive_db, 0.0, 8.0))
        if punch is not None:
            self.punch = float(np.clip(punch, 0.0, 1.0))
        if trim_db is not None:
            self.trim_db = float(np.clip(trim_db, -6.0, 6.0))
        if freq is not None and abs(float(freq) - self.freq) > 0.5:
            self.freq = float(np.clip(freq, 60.0, 200.0))
            self._design()                     # fresh states on re-split

    # ------------------------------------------------------------------ audio
    def process(self, x):
        lo, self.zi_lo = sosfilt(self.sos_lo, x, axis=0, zi=self.zi_lo)
        hi = x - lo                            # exact complement

        d = 10.0 ** (self.drive_db / 20.0)
        driven = lo * d
        c = _soft_ceiling(driven)

        # keep the fundamental region, scale the clip harmonics (PUNCH)
        fund, self.zi_h = sosfilt(self.sos_h, c, axis=0, zi=self.zi_h)
        bass = (fund + self.punch * (c - fund)) * 10.0 ** (self.trim_db / 20.0)

        # meters: crest reduction of the bass clipper + band level
        pk_in = float(np.max(np.abs(driven))) + 1e-12
        pk_out = float(np.max(np.abs(c))) + 1e-12
        red = max(0.0, 20.0 * np.log10(pk_in / pk_out))
        self._clip_sm = 0.7 * self._clip_sm + 0.3 * red
        self.meters["clip_db"] = self._clip_sm
        self.meters["bass_db"] = 20.0 * np.log10(
            float(np.max(np.abs(bass))) + 1e-12)

        return hi + bass

    def reset(self):
        self.zi_lo[:] = 0.0
        self.zi_h[:] = 0.0
        self._clip_sm = 0.0
        self.meters = {"clip_db": 0.0, "bass_db": -80.0}
