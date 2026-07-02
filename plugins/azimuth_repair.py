"""OmegaFM plugin: Azimuth / Stereo Repair.

The Stereo Tool signature source-repair trick: auto-detects and
corrects L/R time misalignment - the classic disease of misaligned
tape-head azimuth, vinyl rips and old cart transfers. A skew of even a
few samples notches the mono sum (comb filtering), smears the image
and wastes L-R modulation; on FM that is real, measurable damage.

Detection: windowed cross-correlation over +/-500 us with parabolic
peak refinement (sub-sample precision), gated by a confidence
threshold - the tracker only trusts frames where L and R are clearly
the *same* signal displaced in time. Intentionally wide or
decorrelated production freezes the tracker: true stereo is not a
defect and is never "corrected".

Correction: the measured skew is removed by a slowly-slewed split
fractional delay (+/- skew/2 on each channel, cubic interpolation,
0.33 ms fixed base latency). When the tracked skew is zero the delays
sit on integer taps and the audio passes *bit-exact* (just the fixed
latency, identical on both channels).

Polarity: a sustained strongly-negative correlation (an inverted
channel) trips a repair after ~1.7 s of evidence, applied through a
30 ms crossfade so the fix never clicks.

Inserted post_input - repair before anything measures or processes.

Controls
--------
CORRECT 0..10   fraction of detected skew removed (0 = meter-only,
                default 10 = full repair)
MAXLAG  100..500 us   detection / correction range (default 400)
POLFIX  0..1    polarity repair enable (default 1)

Meters: SKEW = applied correction in microseconds (bipolar),
CORR = signed peak correlation (quality + polarity at a glance).
"""

import numpy as np

PLUGIN = {
    "id": "azimuth_repair",
    "name": "Azimuth / Stereo Repair",
    "version": "1.0",
    "insert": "post_input",
    "params": [
        {"key": "correct", "label": "CORRECT", "min": 0.0, "max": 10.0,
         "default": 10.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "maxlag_us", "label": "MAXLAG", "min": 100.0, "max": 500.0,
         "default": 400.0, "fmt": "{:.0f}", "suffix": "us"},
        {"key": "polfix", "label": "POLFIX", "min": 0.0, "max": 1.0,
         "default": 1.0, "fmt": "{:.0f}", "suffix": ""},
    ],
    "meters": [
        {"key": "skew_us", "label": "SKEW", "mode": "bipolar",
         "lo": -500.0, "hi": 500.0},
        {"key": "corr", "label": "CORR", "mode": "bipolar",
         "lo": -1.0, "hi": 1.0},
    ],
}

ANA = 4096                # analysis frame (85 ms @ 48k)
D0 = 16                   # base delay, samples (0.33 ms both channels)
CONF = 0.55               # correlation needed before the tracker trusts
SLEW = 4.0                # samples of correction slewed per second
POL_BLOCKS = 20           # sustained frames before a polarity repair
XFADE_S = 0.03            # polarity crossfade


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.correct = 10.0
        self.maxlag_us = 400.0
        self.polfix = 1.0
        self._maxlag = self._lag_samples()
        # analysis accumulator
        self._abuf = np.zeros((ANA, 2))
        self._afill = 0
        self._win = np.hanning(ANA)
        # tracked / applied state
        self._est = 0.0                 # trusted skew estimate (samples)
        self._applied = 0.0             # slewed, actually applied
        self._corr = 0.0
        self._pol = 1.0                 # polarity multiplier on R
        self._pol_votes = 0
        self._xfade = None              # (target, pos, total) during flips
        # delay history (per channel)
        self._hist = np.zeros((D0 + 32, 2))
        self.meters = {"skew_us": 0.0, "corr": 0.0}

    def _lag_samples(self):
        return int(np.clip(round(self.maxlag_us * 1e-6 * self.fs / 2) * 2,
                           4, 28))

    def set_params(self, correct=None, maxlag_us=None, polfix=None):
        if correct is not None:
            self.correct = float(np.clip(correct, 0.0, 10.0))
        if maxlag_us is not None:
            self.maxlag_us = float(np.clip(maxlag_us, 100.0, 500.0))
            self._maxlag = self._lag_samples()
        if polfix is not None:
            self.polfix = float(np.clip(polfix, 0.0, 1.0))

    # ------------------------------------------------------------------ est
    def _analyse(self):
        l = self._abuf[:, 0] * self._win
        r = self._abuf[:, 1] * self._win
        m = self._maxlag
        lc = l[m:-m]
        el = float(np.dot(lc, lc))
        if el < ANA * 1e-9:                       # level gate (~ -70 dBFS)
            return
        lags = np.arange(-m, m + 1)
        cc = np.empty(len(lags))
        for idx, k in enumerate(lags):
            rs = r[m + k: m + k + len(lc)]
            cc[idx] = float(np.dot(lc, rs)) / (
                np.sqrt(el * float(np.dot(rs, rs))) + 1e-12)
        i = int(np.argmax(np.abs(cc)))
        pk = float(cc[i])
        coef = abs(pk)
        self._corr = 0.7 * self._corr + 0.3 * pk
        if coef < CONF:
            return                                # true stereo: hands off
        # parabolic sub-sample refinement on |cc|
        if 0 < i < len(cc) - 1:
            y0, y1, y2 = abs(cc[i - 1]), coef, abs(cc[i + 1])
            denom = (y0 - 2 * y1 + y2)
            frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            frac = float(np.clip(frac, -0.5, 0.5))
        else:
            frac = 0.0
        lag = float(lags[i]) + frac               # R arrives `lag` after L
        self._est = 0.75 * self._est + 0.25 * lag
        # polarity voting
        if pk < -CONF:
            self._pol_votes += 1
        else:
            self._pol_votes = max(0, self._pol_votes - 1)
        if (self.polfix >= 0.5 and self._pol_votes >= POL_BLOCKS
                and self._pol > 0 and self._xfade is None):
            self._xfade = (-1.0, 0, int(XFADE_S * self.fs))
        if (pk > CONF and self._pol < 0 and self._xfade is None
                and self._pol_votes == 0):
            self._xfade = (1.0, 0, int(XFADE_S * self.fs))

    # ------------------------------------------------------------------ audio
    def _frac_delay(self, work, delay, n):
        """Cubic (Catmull-Rom) read of `n` samples delayed by `delay`
        from `work` (history + block). Integer delay = exact taps."""
        base = len(work) - n
        pos = base + np.arange(n) - delay
        i0 = np.floor(pos).astype(int)
        f = pos - i0
        if np.all(f < 1e-9):                       # integer: bit-exact path
            return work[i0]
        xm1 = work[i0 - 1]
        x0 = work[i0]
        x1 = work[i0 + 1]
        x2 = work[np.minimum(i0 + 2, len(work) - 1)]
        # Catmull-Rom coefficients
        c0 = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
        c1 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
        c2 = -0.5 * xm1 + 0.5 * x1
        return ((c0 * f + c1) * f + c2) * f + x0

    def process(self, x):
        n = len(x)
        # feed the analyser
        take = min(ANA - self._afill, n)
        self._abuf[self._afill:self._afill + take] = x[:take]
        self._afill += take
        if self._afill >= ANA:
            self._analyse()
            self._afill = 0
        # slew the applied correction toward the trusted estimate
        want = self._est * (self.correct / 10.0)
        lim = np.clip(want, -self._maxlag, self._maxlag)
        step = SLEW * n / self.fs
        d = self._applied + float(np.clip(lim - self._applied, -step, step))
        if abs(d) < 1e-4:
            d = 0.0
        self._applied = d

        work = np.concatenate([self._hist, x], axis=0)
        self._hist = work[-len(self._hist):].copy()
        # R arrives d after L -> delay L by d (split across channels)
        yl = self._frac_delay(work[:, 0], D0 + d / 2.0, n)
        yr = self._frac_delay(work[:, 1], D0 - d / 2.0, n)

        # polarity (with click-free crossfade on repair)
        if self._xfade is not None:
            tgt, ppos, tot = self._xfade
            ramp = np.arange(ppos, ppos + n) / tot
            m = self._pol + (tgt - self._pol) * np.clip(ramp, 0, 1)
            yr = yr * m
            if ppos + n >= tot:
                self._pol = tgt
                self._pol_votes = 0
                self._xfade = None
            else:
                self._xfade = (tgt, ppos + n, tot)
        elif self._pol < 0:
            yr = -yr

        self.meters["skew_us"] = self._applied / self.fs * 1e6
        self.meters["corr"] = self._corr
        return np.stack([yl, yr], axis=1)

    def reset(self):
        self._abuf[:] = 0.0
        self._afill = 0
        self._est = 0.0
        self._applied = 0.0
        self._corr = 0.0
        self._pol = 1.0
        self._pol_votes = 0
        self._xfade = None
        self._hist[:] = 0.0
        self.meters = {"skew_us": 0.0, "corr": 0.0}
