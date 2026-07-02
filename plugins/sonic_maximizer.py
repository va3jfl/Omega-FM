"""OmegaFM plugin: Sonic Maximizer (BBE-style).

A clone of the classic two-knob rack enhancer, engineered from what
the box actually does rather than the brochure:

  * phase alignment - a cascade of four first-order allpass sections
    (~550 Hz) gives low frequencies ~2.3 ms of group delay smoothly
    tapering to zero at the top: bass fundamentals and their harmonic
    transients arrive time-aligned (the "definition" effect).  Allpass
    means |H| = 1 everywhere - the alignment is magnitude-transparent
    by construction, unlike naive band-delay implementations which
    comb at the crossovers;

  * PROCESS - the signature *program-dependent* HF enhancement: the
    treble lift is envelope-controlled against the mid band, applying
    the full amount to dull material and backing smoothly off as the
    top end gets hot (fast duck, relaxed bloom), so it adds air and
    attack without ever turning harshness up further;

  * LO CONTOUR - the gentle static low shelf (< 150 Hz) of the
    original.

Inserted post_input: exactly where a rack BBE sits in a station -
between the console and the processor input - with the AGC, de-esser
point, mix-coupled leveler and HF limiter all downstream as
guardrails.

Factory defaults are deliberately modest (PROCESS 3, LO CONTOUR 2,
about +3 dB dynamic air and +1.6 dB warmth at most); crank toward 10
for the full late-night-infomercial experience.

Controls
--------
PROCESS  0..10   dynamic HF enhancement amount (default 3)
CONTOUR  0..10   low shelf warmth (default 2)
OUT      -6..+6  output trim (default 0)

Meters: HF+ = live dynamic treble lift (dB), OUT = output level.
"""

import numpy as np
from scipy.signal import lfilter, butter, sosfilt

PLUGIN = {
    "id": "sonic_maximizer",
    "name": "Sonic Maximizer",
    "version": "1.0",
    "insert": "post_input",
    "params": [
        {"key": "proc", "label": "PROCESS", "min": 0.0, "max": 10.0,
         "default": 3.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "contour", "label": "CONTOUR", "min": 0.0, "max": 10.0,
         "default": 2.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "out_db", "label": "OUT", "min": -6.0, "max": 6.0,
         "default": 0.0, "fmt": "{:+.1f}", "suffix": "dB"},
    ],
    "meters": [
        {"key": "hf_boost", "label": "HF+", "mode": "level",
         "lo": 0.0, "hi": 12.0},
        {"key": "out_lvl", "label": "OUT", "mode": "level",
         "lo": -42.0, "hi": 3.0},
    ],
}

AP_FREQ = 550.0          # allpass corner: ~2.3 ms LF group delay
AP_N = 4
F_LO = 150.0             # lo contour shelf region
F_HI = 2400.0            # enhancement band
TILT_FULL = -8.0         # detector tilt at/below which boost is full
TILT_OFF = 3.0           # tilt at/above which boost is zero
CTRL = 32


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.proc_amt = 3.0
        self.contour = 2.0
        self.out_db = 0.0
        # allpass cascade
        t = np.tan(np.pi * AP_FREQ / self.fs)
        self._ap_a = (t - 1.0) / (t + 1.0)
        self._ap_zi = [np.zeros((1, self.channels)) for _ in range(AP_N)]
        # complementary splits
        self._a_lo = float(np.exp(-2.0 * np.pi * F_LO / self.fs))
        self._a_hi = float(np.exp(-2.0 * np.pi * F_HI / self.fs))
        self._zi_lo = np.zeros((1, self.channels))
        self._zi_hi = np.zeros((1, self.channels))
        # steep detectors (the gain legs stay gentle & authentic;
        # detection must be honest)
        self._sos_dh = butter(2, F_HI / (self.fs / 2.0), btype="high",
                              output="sos")
        self._sos_dm = butter(2, [300.0 / (self.fs / 2.0),
                                  2000.0 / (self.fs / 2.0)],
                              btype="band", output="sos")
        self._zi_dh = np.zeros((self._sos_dh.shape[0], 2))
        self._zi_dm = np.zeros((self._sos_dm.shape[0], 2))
        # envelopes (power, ~6 ms)
        self._a_env = float(np.exp(-1.0 / (0.006 * self.fs)))
        self._zi_eh = np.zeros(1)
        self._zi_em = np.zeros(1)
        # dynamic boost ballistics (control rate)
        self._up = float(np.exp(-(CTRL / self.fs) / 0.025))   # bloom 25 ms
        self._dn = float(np.exp(-(CTRL / self.fs) / 0.008))   # duck 8 ms
        self._boost = 0.0
        self._a_sm = float(np.exp(-1.0 / (0.001 * self.fs)))
        self._zi_sm = np.zeros(1)
        self.meters = {"hf_boost": 0.0, "out_lvl": -80.0}

    # ------------------------------------------------------------------ setup
    def set_params(self, proc=None, contour=None, out_db=None):
        if proc is not None:
            self.proc_amt = float(np.clip(proc, 0.0, 10.0))
        if contour is not None:
            self.contour = float(np.clip(contour, 0.0, 10.0))
        if out_db is not None:
            self.out_db = float(np.clip(out_db, -6.0, 6.0))

    # ------------------------------------------------------------------ audio
    def process_ap(self, x):
        a = self._ap_a
        for k in range(AP_N):
            x, zf = lfilter([a, 1.0], [1.0, a], x, axis=0,
                            zi=self._ap_zi[k])
            self._ap_zi[k] = zf
        return x

    def process(self, x):
        n = len(x)
        x = self.process_ap(x)

        a1 = self._a_lo
        lo, self._zi_lo = lfilter([1 - a1], [1, -a1], x, axis=0,
                                  zi=self._zi_lo)
        rest = x - lo
        a2 = self._a_hi
        mid, self._zi_hi = lfilter([1 - a2], [1, -a2], rest, axis=0,
                                   zi=self._zi_hi)
        hi = rest - mid

        # program-dependent HF lift: full on dull, off on bright
        mono = 0.5 * (x[:, 0] + x[:, 1])
        dh, self._zi_dh = sosfilt(self._sos_dh, mono, zi=self._zi_dh)
        dm, self._zi_dm = sosfilt(self._sos_dm, mono, zi=self._zi_dm)
        ph, self._zi_eh = lfilter([1 - self._a_env], [1, -self._a_env],
                                  dh * dh, zi=self._zi_eh)
        pm, self._zi_em = lfilter([1 - self._a_env], [1, -self._a_env],
                                  dm * dm, zi=self._zi_em)
        tilt = (5.0 * np.log10(np.maximum(ph, 1e-14))
                - 5.0 * np.log10(np.maximum(pm, 1e-14)))
        hf_max = self.proc_amt                      # knob = dB, 1:1
        want = hf_max * np.clip((TILT_OFF - tilt) / (TILT_OFF - TILT_FULL),
                                0.0, 1.0)

        b = self._boost
        boosts = np.empty(n)
        i = 0
        while i < n:
            j = min(i + CTRL, n)
            t = float(want[i:j].mean())
            w = self._up if t > b else self._dn
            if (j - i) != CTRL:
                w = float(w ** ((j - i) / CTRL))
            b = w * b + (1.0 - w) * t
            boosts[i:j] = b
            i = j
        self._boost = b
        bsm, self._zi_sm = lfilter([1 - self._a_sm], [1, -self._a_sm],
                                   boosts, zi=self._zi_sm)

        g_hi = 10.0 ** (bsm / 20.0)
        g_lo = 10.0 ** (0.8 * self.contour / 20.0)
        g_out = 10.0 ** (self.out_db / 20.0)
        y = (lo * g_lo + mid + hi * g_hi[:, None]) * g_out

        self.meters["hf_boost"] = float(np.mean(bsm))
        self.meters["out_lvl"] = 20.0 * np.log10(
            float(np.max(np.abs(y))) + 1e-12)
        return y

    def reset(self):
        for z in self._ap_zi:
            z[:] = 0.0
        self._zi_dh[:] = 0.0
        self._zi_dm[:] = 0.0
        self._zi_lo[:] = 0.0
        self._zi_hi[:] = 0.0
        self._zi_eh[:] = 0.0
        self._zi_em[:] = 0.0
        self._zi_sm[:] = 0.0
        self._boost = 0.0
        self.meters = {"hf_boost": 0.0, "out_lvl": -80.0}
