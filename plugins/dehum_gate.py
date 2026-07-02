"""OmegaFM plugin: Dehummer + Noise Gate.

The utility module that saves the day someone feeds you a ground loop.

DEHUMMER
    A fine-resolution detector (0.2 Hz steps, 45-65 Hz, 1 s windows)
    hunts for a *stationary* tonal line - mains hum at 50 or 60 Hz,
    including drifted supplies. Two guards make it safe: it only
    *judges* during quiet moments (track intros, gaps, fades - exactly
    where a ground loop is exposed and audible; loud program freezes
    the verdict rather than confusing it), and confidence only accrues
    while the same frequency (+/-0.4 Hz) stays tonally dominant across
    consecutive quiet windows - bass NOTES, which move, can never
    trip it. Once locked, a chain of
    high-Q notches (fundamental + HARM-1 harmonics, Q 35) is blended
    in over ~1 s, tracks slow drift, and melts away within ~1.5 s of
    the hum disappearing. With no hum the wet path is skipped entirely
    and the audio is bit-exact.

NOISE GATE
    A gentle 2:1 downward expander (not a chopper): below GATE the
    floor is pushed down up to RANGE dB, opening in ~4 ms, holding
    120 ms and closing over ~250 ms, stereo-linked. Above threshold
    the gain is exactly 1.0 - program is never touched. Its job is to
    stop the AGC from hoisting a grotty source's hiss 12 dB into the
    clear. Runs after the dehummer, so the threshold sees the true
    floor.

Inserted post_input - clean the source before anything measures it.

Controls
--------
HUM   0..10    notch depth, ~4.5 dB per unit (default 6 = 27 dB kill)
HARM  1..8     lines notched: fundamental + harmonics (default 5)
GATE  -70..-35 expander threshold, dBFS (default -55)
RANGE 0..24    max floor attenuation, dB (default 12)
TEST  0/1      commissioning aid: injects a quiet ground-loop bed
               (50 Hz complex at -38 dBFS + hiss at -62) into the
               module input so you can watch the whole thing work:
               pause the program and within a few seconds MAINS locks
               50, HUM dives, the hum audibly vanishes and GATE rides
               the remaining hiss. Never leave on for air.

Meters: MAINS = detected line frequency (Hz, '--' when clean),
HUM = current notch depth, GATE = current floor attenuation.
"""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter

PLUGIN = {
    "id": "dehum_gate",
    "name": "Dehummer + Gate",
    "version": "1.2",
    "insert": "post_input",
    "params": [
        {"key": "hum", "label": "HUM", "min": 0.0, "max": 10.0,
         "default": 6.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "harm", "label": "HARM", "min": 1.0, "max": 8.0,
         "default": 5.0, "fmt": "{:.0f}", "suffix": ""},
        {"key": "gate_db", "label": "GATE", "min": -70.0, "max": -35.0,
         "default": -55.0, "fmt": "{:.0f}", "suffix": "dB"},
        {"key": "range_db", "label": "RANGE", "min": 0.0, "max": 24.0,
         "default": 12.0, "fmt": "{:.0f}", "suffix": "dB"},
        {"key": "test", "label": "TEST", "min": 0.0, "max": 1.0,
         "default": 0.0, "fmt": "{:.0f}", "suffix": ""},
    ],
    "meters": [
        {"key": "mains_hz", "label": "MAINS", "mode": "level",
         "lo": 45.0, "hi": 65.0, "unit": "hz"},
        {"key": "hum_db", "label": "HUM", "mode": "gr",
         "lo": 0.0, "hi": 45.0},
        {"key": "gate_gr", "label": "GATE", "mode": "gr",
         "lo": 0.0, "hi": 24.0},
    ],
}

CTRL = 32
DECIM = 32                       # detector runs at fs/32 (1.5 kHz)
SCAN_S = 1.0                     # one detection window per second
F_LO, F_HI, F_STEP = 45.0, 65.0, 0.2
TONALITY_DB = 12.0               # line must stand this far over context
QUIET_DB = -30.0                 # detector judges only below this (LF rms)
CONF_ON = 3                      # windows of stable tone before engaging
CONF_FULL = 5
NOTCH_Q = 35.0
EXP_RATIO = 2.0                  # gate expansion below threshold


def _notch_sos(f0, q, fs, n_harm):
    secs = []
    for k in range(1, n_harm + 1):
        fk = f0 * k
        if fk > 0.45 * fs or fk > 500.0:
            break
        w0 = 2 * np.pi * fk / fs
        alpha = np.sin(w0) / (2 * q)
        b = np.array([1.0, -2 * np.cos(w0), 1.0])
        a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
        secs.append(np.hstack([b / a[0], a / a[0]]))
    return np.array(secs)


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.hum = 6.0
        self.harm = 5.0
        self.gate_db = -55.0
        self.range_db = 12.0
        self.test = 0.0
        self._tn = 0
        self._rng_t = np.random.default_rng(1234)
        # detector: LP + decimate + 1 s ring, cached scan basis
        self.sos_lp = butter(4, 350.0 / (self.fs / 2.0), output="sos")
        self._zi_lp = np.zeros((self.sos_lp.shape[0], 2))
        self._phase = 0
        self._dn = int(SCAN_S * self.fs / DECIM)
        self._ring = np.zeros(self._dn)
        self._rfill = 0
        fd = self.fs / DECIM
        self._freqs = np.arange(F_LO, F_HI + 1e-9, F_STEP)
        t = np.arange(self._dn) / fd
        self._basis = np.exp(-2j * np.pi * np.outer(self._freqs, t)) \
            * np.hanning(self._dn)
        # tracking state
        self._f = 0.0
        self._conf = 0
        self._a = 0.0                      # blended notch amount 0..1
        self._a_slew = float(np.exp(-(CTRL / self.fs) / 0.25))
        self._sos = None
        self._zi_n = None
        self._sos_f = 0.0
        # gate
        self._a_env = float(np.exp(-1.0 / (0.005 * self.fs)))
        self._zi_env = np.zeros(1)
        self._g = 0.0                      # attenuation dB (>= 0)
        self._hold = 0
        self._att = float(np.exp(-(CTRL / self.fs) / 0.004))
        self._rel = float(np.exp(-(CTRL / self.fs) / 0.25))
        self.meters = {"mains_hz": 45.0, "hum_db": 0.0, "gate_gr": 0.0}

    def set_params(self, hum=None, harm=None, gate_db=None, range_db=None,
                   test=None):
        if hum is not None:
            self.hum = float(np.clip(hum, 0.0, 10.0))
        if harm is not None:
            h = float(np.clip(round(harm), 1, 8))
            if h != self.harm:
                self.harm = h
                self._sos_f = 0.0          # force redesign
        if gate_db is not None:
            self.gate_db = float(np.clip(gate_db, -70.0, -35.0))
        if range_db is not None:
            self.range_db = float(np.clip(range_db, 0.0, 24.0))
        if test is not None:
            self.test = float(np.clip(test, 0.0, 1.0))

    # ------------------------------------------------------------------ scan
    def _scan(self):
        # judge only in quiet moments; loud program freezes the verdict
        rms_db = 10.0 * np.log10(float(np.mean(self._ring ** 2)) + 1e-14)
        if rms_db > QUIET_DB:
            return
        mag = np.abs(self._basis @ self._ring)
        i = int(np.argmax(mag))
        f = float(self._freqs[i])
        ctx = np.median(mag[np.abs(self._freqs - f) > 1.5]) + 1e-12
        tonal = 20.0 * np.log10(mag[i] / ctx)
        level_ok = mag[i] > self._dn * 1e-5          # ~ -85 dBFS line
        if tonal > TONALITY_DB and level_ok and \
                (self._conf == 0 or abs(f - self._f) < 0.4):
            self._f = f if self._conf == 0 else 0.7 * self._f + 0.3 * f
            self._conf = min(self._conf + 1, CONF_FULL + 2)
        else:
            self._conf = max(self._conf - 2, 0)

    def _ensure_notches(self):
        if abs(self._f - self._sos_f) > 0.15 or self._sos is None:
            self._sos = _notch_sos(self._f, NOTCH_Q, self.fs,
                                   int(self.harm))
            self._zi_n = np.zeros((self._sos.shape[0], 2, self.channels))
            self._sos_f = self._f

    # ------------------------------------------------------------------ audio
    def process(self, x):
        n = len(x)
        if self.test >= 0.5:                     # commissioning bed
            tt = (self._tn + np.arange(n)) / self.fs
            bed = (10 ** (-38 / 20) * np.sin(2 * np.pi * 50 * tt)
                   + 10 ** (-44 / 20) * np.sin(2 * np.pi * 100 * tt)
                   + 10 ** (-42 / 20) * np.sin(2 * np.pi * 150 * tt)
                   + self._rng_t.standard_normal(n) * 10 ** (-62 / 20))
            x = x + bed[:, None]
            # TEST also seeds the tracker so the response is INSTANT -
            # MAINS locks 50 and the notches engage with program playing
            if self._f == 0.0:
                self._f = 50.0
            self._conf = max(self._conf, CONF_FULL)
        self._tn += n
        mono = 0.5 * (x[:, 0] + x[:, 1])
        lo, self._zi_lp = sosfilt(self.sos_lp, mono, zi=self._zi_lp)
        idx = np.arange(self._phase, n, DECIM)
        self._phase = (self._phase - n) % DECIM
        for v in lo[idx]:
            self._ring[self._rfill] = v
            self._rfill += 1
            if self._rfill >= self._dn:
                self._scan()
                self._rfill = 0

        target = 0.0
        if self.hum > 0 and self._conf >= CONF_ON and self._f > 0:
            frac = min(1.0, (self._conf - CONF_ON + 1)
                       / (CONF_FULL - CONF_ON + 1))
            depth_db = 4.5 * self.hum * frac     # HUM knob = dB of kill
            target = 1.0 - 10.0 ** (-depth_db / 20.0)
        a = self._a
        w = float(np.exp(-(n / self.fs) / 0.25))     # true 250 ms slew
        a = w * a + (1 - w) * target
        self._a = a

        if a > 1e-4:
            self._ensure_notches()
            wet, self._zi_n = sosfilt(self._sos, x, axis=0, zi=self._zi_n)
            y = x * (1.0 - a) + wet * a
        else:
            y = x                            # bit-exact when clean

        # ---- gate (after dehum: sees the true floor) ---------------------
        p = 0.5 * (y[:, 0] ** 2 + y[:, 1] ** 2)
        env, self._zi_env = lfilter([1 - self._a_env], [1, -self._a_env],
                                    p, zi=self._zi_env)
        lv = 10.0 * np.log10(np.maximum(env, 1e-14))
        g = self._g
        hold = self._hold
        hold_blocks = int(0.12 * self.fs / CTRL)
        gains = np.empty(n)
        i = 0
        any_gr = False
        while i < n:
            j = min(i + CTRL, n)
            l = float(lv[i:j].max())
            if l >= self.gate_db:
                g *= self._att               # open fast
                hold = hold_blocks
            elif hold > 0:
                hold -= 1
                g *= self._att if g > 0 else 1.0
            else:
                want = min(self.range_db,
                           (self.gate_db - l) * (EXP_RATIO - 1.0))
                g = self._rel * g + (1 - self._rel) * want \
                    if want > g else g * self._rel + (1 - self._rel) * want
            if g < 1e-3:
                g = 0.0
            else:
                any_gr = True
            gains[i:j] = g
            i = j
        self._g = g
        self._hold = hold
        if any_gr:
            y = y * (10.0 ** (-gains / 20.0))[:, None]

        engaged = self._conf >= CONF_ON and self.hum > 0
        self.meters["mains_hz"] = self._f if engaged else 45.0
        self.meters["hum_db"] = (-20.0 * np.log10(max(1.0 - a, 1e-3))
                                 if a > 1e-4 else 0.0)
        self.meters["gate_gr"] = g
        return y

    def reset(self):
        self._zi_lp[:] = 0.0
        self._phase = 0
        self._ring[:] = 0.0
        self._rfill = 0
        self._f = 0.0
        self._conf = 0
        self._a = 0.0
        self._sos = None
        self._zi_n = None
        self._sos_f = 0.0
        self._zi_env = np.zeros(1)
        self._g = 0.0
        self._hold = 0
        self._tn = 0
        self.meters = {"mains_hz": 45.0, "hum_db": 0.0, "gate_gr": 0.0}
