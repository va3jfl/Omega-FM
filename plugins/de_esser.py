"""OmegaFM plugin: De-Esser.

Split-band sibilance controller for the harsh 's/sh/t' splatter that
75 us pre-emphasis turns into clipper distortion. Three design choices
keep it transparent rather than aggressive:

  * split-band - only the band above FREQ is gain-reduced through a
    complementary split (below the split is bit-untouched, so vowels
    and music body never duck: no lisping, no pumping);
  * relative detection - sibilance is measured against the program's
    own level (a steeper 2nd-order detector vs a 15 ms program
    reference), so the threshold is a spectral *tilt*: it adapts to
    quiet and loud, dull and bright material instead of hammering
    whatever happens to be hot;
  * event-scoped ballistics - sub-millisecond attack authority, full
    release between esses (a de-esser must let go completely, the
    opposite of the chain's riding limiters), and depth hard-capped by
    RANGE so it can never crush the top end.

Inserted post_agc: levels are AGC-normalised there, and esses are
removed *before* the leveler / compressor / HF limiter react to them,
so every downstream stage works less on sibilance.

Factory defaults are calibrated from measured tilt statistics
(music p99 = -4.0 dB, vowels p90 = -8.6, esses median = -1.4): the
default THRESH of -4.0 makes music and vowels essentially invisible
(< 0.5 dB) while a real ess draws a gentle 3-6 dB.

Controls
--------
FREQ    3k..9k Hz   split / detector corner (default 6000)
THRESH  -12..+6 dB  relative tilt threshold (default -4.0)
RANGE   0..10 dB    maximum reduction depth (default 6)
RELEASE 20..200 ms  recovery after the ess (default 60)

Meters: GR = live sibilance reduction, SIB = detector band level.
"""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter

PLUGIN = {
    "id": "de_esser",
    "name": "De-Esser",
    "version": "1.0",
    "insert": "post_agc",
    "params": [
        {"key": "freq", "label": "FREQ", "min": 3000.0, "max": 9000.0,
         "default": 6000.0, "fmt": "{:.0f}", "suffix": "Hz"},
        {"key": "thresh_db", "label": "THRESH", "min": -12.0, "max": 6.0,
         "default": -4.0, "fmt": "{:+.1f}", "suffix": "dB"},
        {"key": "range_db", "label": "RANGE", "min": 0.0, "max": 10.0,
         "default": 6.0, "fmt": "{:.1f}", "suffix": "dB"},
        {"key": "release_ms", "label": "RELEASE", "min": 20.0, "max": 200.0,
         "default": 60.0, "fmt": "{:.0f}", "suffix": "ms"},
    ],
    "meters": [
        {"key": "gr_db", "label": "GR", "mode": "gr", "lo": 0.0, "hi": 10.0},
        {"key": "sib_db", "label": "SIB", "mode": "level",
         "lo": -42.0, "hi": 3.0},
    ],
}

CTRL = 32                       # gain recursion granularity (samples)
SLOPE = 1.5                     # dB of GR per dB over threshold
KNEE = 3.0                      # soft knee width (dB)


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.freq = 6000.0
        self.thresh_db = -4.0
        self.range_db = 6.0
        self.release_ms = 60.0
        self._design()
        # envelope states (power domain)
        self._a_sib = float(np.exp(-1.0 / (0.003 * self.fs)))
        self._a_ref = float(np.exp(-1.0 / (0.015 * self.fs)))
        self._zi_sib = np.zeros(1)
        self._zi_ref = np.zeros(1)
        self._g = 0.0                            # gain state, dB <= 0
        self._recalc_release()
        # gain edge smoother (~0.7 ms)
        self._a_sm = float(np.exp(-1.0 / (0.0007 * self.fs)))
        self._zi_sm = np.zeros(1)
        self.meters = {"gr_db": 0.0, "sib_db": -80.0}

    # ------------------------------------------------------------------ setup
    def _design(self):
        f = float(np.clip(self.freq, 2500.0, 10000.0))
        a = float(np.exp(-2.0 * np.pi * f / self.fs))
        self._split_a = a                        # one-pole complementary split
        self._zi_lo = np.zeros((1, self.channels))
        self.sos_det = butter(2, f / (self.fs / 2.0), btype="high",
                              output="sos")
        self._zi_det = np.zeros((self.sos_det.shape[0], 2))

    def _recalc_release(self):
        self._rel_a = float(np.exp(-(CTRL / self.fs)
                                   / (max(self.release_ms, 5.0) * 1e-3)))

    def set_params(self, freq=None, thresh_db=None, range_db=None,
                   release_ms=None):
        if thresh_db is not None:
            self.thresh_db = float(np.clip(thresh_db, -12.0, 6.0))
        if range_db is not None:
            self.range_db = float(np.clip(range_db, 0.0, 10.0))
        if release_ms is not None:
            self.release_ms = float(np.clip(release_ms, 20.0, 200.0))
            self._recalc_release()
        if freq is not None and abs(float(freq) - self.freq) > 0.5:
            self.freq = float(np.clip(freq, 3000.0, 9000.0))
            self._design()

    # ------------------------------------------------------------------ audio
    def process(self, x):
        n = len(x)
        mono = 0.5 * (x[:, 0] + x[:, 1])

        # complementary split (processed path)
        a = self._split_a
        lo, zf = lfilter([1.0 - a], [1.0, -a], x, axis=0,
                         zi=self._zi_lo * np.ones((1, self.channels)))
        self._zi_lo = zf[:1]
        hp = x - lo

        # detection: steep HP band vs program reference, power envelopes
        det, self._zi_det = sosfilt(self.sos_det, mono, zi=self._zi_det)
        p_sib, self._zi_sib = lfilter([1 - self._a_sib], [1, -self._a_sib],
                                      det * det, zi=self._zi_sib)
        p_ref, self._zi_ref = lfilter([1 - self._a_ref], [1, -self._a_ref],
                                      mono * mono, zi=self._zi_ref)
        tilt = (5.0 * np.log10(np.maximum(p_sib, 1e-14))
                - 5.0 * np.log10(np.maximum(p_ref, 1e-14)))

        # soft-knee static curve on the tilt overage
        over = tilt - self.thresh_db
        k = KNEE
        gr = np.where(over <= -k / 2, 0.0,
             np.where(over >= k / 2, SLOPE * over,
                      SLOPE * (over + k / 2) ** 2 / (2 * k)))
        target = -np.minimum(gr, self.range_db)         # dB, <= 0

        # control-rate ballistics: instant attack (chunk min),
        # one-pole release toward 0 - a de-esser lets go completely
        g = self._g
        gains = np.empty(n)
        i = 0
        while i < n:
            j = min(i + CTRL, n)
            t = float(target[i:j].min())
            if t < g:
                g = t
            else:
                w = self._rel_a if (j - i) == CTRL \
                    else float(self._rel_a ** ((j - i) / CTRL))
                g = g * w
            gains[i:j] = g
            i = j
        self._g = g

        gsm, self._zi_sm = lfilter([1 - self._a_sm], [1, -self._a_sm],
                                   gains, zi=self._zi_sm)
        lin = 10.0 ** (gsm / 20.0)

        self.meters["gr_db"] = max(0.0, float(-np.min(gsm)))
        self.meters["sib_db"] = float(
            5.0 * np.log10(np.max(p_sib) + 1e-14))

        return lo + hp * lin[:, None]

    def reset(self):
        self._zi_lo[:] = 0.0
        self._zi_det[:] = 0.0
        self._zi_sib[:] = 0.0
        self._zi_ref[:] = 0.0
        self._zi_sm[:] = 0.0
        self._g = 0.0
        self.meters = {"gr_db": 0.0, "sib_db": -80.0}
