"""OmegaFM plugin: Multipath / Stereo Energy Governor.

The grown-up companion to the stereo widener. Excessive L-R (side)
energy is the FM troublemaker: it loads the fragile 23-53 kHz
subcarrier region - which is what multipath reflections chew on in a
moving car - and it spends modulation budget that mono listeners never
hear. Stereo Tool obsesses over this for good reason.

This module is a *ratio limiter*, not a width knob: it watches the
side-to-mid energy ratio with 40 ms energy-scale envelopes and, only when the ratio
exceeds CEILING, ducks the side channel just enough to hold it -
soft-kneed, fast attack, musical release. Two properties make it
broadcast-grade:

  * mono is mathematically untouched - only S is scaled, so the L+R
    sum (what mono radios and the loudness meter hear) is preserved
    sample-for-sample;
  * below the ceiling it is bit-exact - normal stereo production
    (calibrated: dense program peaks around -5 dB S/M even after the
    factory widener) passes completely free. Only hyper-wide moments -
    anti-phase synths, huge reverbs, azimuth junk - get pulled to
    legal.

Inserted post_multiband, after all tonal processing and right before
the final stages: it governs the stereo image that actually reaches
the MPX generator.

Controls
--------
CEILING -12..+3 dB  maximum S/M energy ratio (default -3)
RELEASE 50..500 ms  recovery after a wide burst (default 250)
RANGE   0..18 dB    maximum side reduction (default 12)

Meters: S/M = live side/mid ratio (red at/above the danger zone),
GR = side gain reduction being applied.
"""

import numpy as np
from scipy.signal import lfilter

PLUGIN = {
    "id": "stereo_governor",
    "name": "Multipath Governor",
    "version": "1.0",
    "insert": "post_multiband",
    "params": [
        {"key": "ceiling_db", "label": "CEILING", "min": -12.0, "max": 3.0,
         "default": -3.0, "fmt": "{:+.1f}", "suffix": "dB"},
        {"key": "release_ms", "label": "RELEASE", "min": 50.0, "max": 500.0,
         "default": 250.0, "fmt": "{:.0f}", "suffix": "ms"},
        {"key": "range_db", "label": "RANGE", "min": 0.0, "max": 18.0,
         "default": 12.0, "fmt": "{:.0f}", "suffix": "dB"},
    ],
    "meters": [
        {"key": "sm_db", "label": "S/M", "mode": "level",
         "lo": -18.0, "hi": 6.0},
        {"key": "gr_db", "label": "GR", "mode": "gr",
         "lo": 0.0, "hi": 12.0},
    ],
}

CTRL = 32
KNEE = 2.0
GATE_DB = -70.0          # both channels silent: relax to unity


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.ceiling_db = -3.0
        self.release_ms = 250.0
        self.range_db = 12.0
        a = float(np.exp(-1.0 / (0.040 * self.fs)))   # energy scale:
        self._a_env = a                # multipath cares about sustained
        self._zi_m = np.zeros(1)       # side loading, not 3 ms blips
        self._zi_s = np.zeros(1)
        self._att = float(np.exp(-(CTRL / self.fs) / 0.010))
        self._warm = 0                 # envelope settle guard
        self._recalc()
        self._g = 0.0
        self._a_sm = float(np.exp(-1.0 / (0.0007 * self.fs)))
        self._zi_g = np.zeros(1)
        self._a_disp = float(np.exp(-(CTRL / self.fs) / 0.1))
        self._disp = -18.0
        self.meters = {"sm_db": -18.0, "gr_db": 0.0}

    def _recalc(self):
        self._rel = float(np.exp(-(CTRL / self.fs)
                                 / (self.release_ms * 1e-3)))

    def set_params(self, ceiling_db=None, release_ms=None, range_db=None):
        if ceiling_db is not None:
            self.ceiling_db = float(np.clip(ceiling_db, -12.0, 3.0))
        if release_ms is not None:
            self.release_ms = float(np.clip(release_ms, 50.0, 500.0))
            self._recalc()
        if range_db is not None:
            self.range_db = float(np.clip(range_db, 0.0, 18.0))

    # ------------------------------------------------------------------ audio
    def process(self, x):
        n = len(x)
        m = 0.5 * (x[:, 0] + x[:, 1])
        s = 0.5 * (x[:, 0] - x[:, 1])
        pm, self._zi_m = lfilter([1 - self._a_env], [1, -self._a_env],
                                 m * m, zi=self._zi_m)
        ps, self._zi_s = lfilter([1 - self._a_env], [1, -self._a_env],
                                 s * s, zi=self._zi_s)
        ratio = 10.0 * np.log10(np.maximum(ps, 1e-14)
                                / np.maximum(pm, 1e-14))
        total = 10.0 * np.log10(pm + ps + 1e-14)
        over = ratio - self.ceiling_db
        k = KNEE
        want = np.where(over <= -k / 2, 0.0,
               np.where(over >= k / 2, over,
                        (over + k / 2) ** 2 / (2 * k)))
        want = np.where(total < GATE_DB, 0.0,
                        np.minimum(want, self.range_db))
        # envelope warm-up after silence/reset: don't judge unsettled
        warm_n = int(0.08 * self.fs)
        if self._warm < warm_n:
            k0 = min(n, warm_n - self._warm)
            want[:k0] = 0.0
        self._warm = min(self._warm + n, warm_n) \
            if total[-1] >= GATE_DB else 0

        g = self._g
        disp = self._disp
        gains = np.empty(n)
        i = 0
        any_gr = False
        while i < n:
            j = min(i + CTRL, n)
            t = float(want[i:j].max())
            if t > g:
                g = self._att * g + (1 - self._att) * t
            else:
                g = self._rel * g + (1 - self._rel) * t
            if g < 1e-3:
                g = 0.0
            else:
                any_gr = True
            gains[i:j] = g
            disp = self._a_disp * disp + (1 - self._a_disp) \
                * float(ratio[j - 1])
            i = j
        self._g = g
        self._disp = disp
        self.meters["sm_db"] = disp
        self.meters["gr_db"] = g

        if not any_gr:
            return x                            # below ceiling: bit-exact

        gsm, self._zi_g = lfilter([1 - self._a_sm], [1, -self._a_sm],
                                  gains, zi=self._zi_g)
        s2 = s * (10.0 ** (-gsm / 20.0))
        return np.stack([m + s2, m - s2], axis=1)

    def reset(self):
        self._zi_m[:] = 0.0
        self._zi_s[:] = 0.0
        self._zi_g[:] = 0.0
        self._g = 0.0
        self._warm = 0
        self._disp = -18.0
        self.meters = {"sm_db": -18.0, "gr_db": 0.0}
