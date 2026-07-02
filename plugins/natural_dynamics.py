"""OmegaFM plugin: Natural Dynamics (transient restorer).

The missing half of source repair: the de-clipper rebuilds the *peaks*
of loudness-war masters, this rebuilds the *punch*. An upward transient
expander that re-emphasizes attack onsets (kick thump, snare crack)
which bus compression flattened.

Like every transient tool (SPL Transient Designer, Stereo Tool's
Natural Dynamics), this is an **operator tool for percussive
formats**: dance, pop, rock, urban. It adapts itself to how flattened
the hits are - *onset prominence* (squashed masters ~4 dB, healthy
5-10, measured immune to upstream de-clipping) drives the depth, and
SENS sets how eagerly low prominence counts as damage: 0 = hard off
(bit-exact), ~4 = default, 10 = full treatment for any beat.

Format note: orchestral / acoustic program contains genuine bow and
swell micro-onsets that no detector can distinguish from weak drum
hits - on such formats leave this module off or SENS at 0, as you
would any transient designer. AMOUNT 0 and SENS 0 are both
mathematically exact bypasses.

Detection is dual-envelope (2 ms vs 80 ms power) so only genuine
onsets are lifted; each restored hit blooms instantly and decays over
SPEED, stereo-linked so the image never wanders.

Inserted post_input, running right after the de-clipper by rack order:
repair peaks -> repair dynamics -> enhance -> widen.

Controls
--------
AMOUNT  0..10   maximum onset lift in dB, scaled by damage (default 3)
SENS    0..10   how low a hit prominence counts as damage (default 4)
SPEED   20..120 bloom decay per hit, ms (default 60)

Meters: PUNCH+ = onset lift being applied right now,
SQUASH = measured source damage (full scale = fully squashed).
Healthy program passes bit-exact when idle and never sees more than
~0.1 dB even at maximum AMOUNT.
"""

import numpy as np
from scipy.signal import lfilter

PLUGIN = {
    "id": "natural_dynamics",
    "name": "Natural Dynamics",
    "version": "1.0",
    "insert": "post_input",
    "params": [
        {"key": "amount", "label": "AMOUNT", "min": 0.0, "max": 10.0,
         "default": 3.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "sens", "label": "SENS", "min": 0.0, "max": 10.0,
         "default": 4.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "speed_ms", "label": "SPEED", "min": 20.0, "max": 120.0,
         "default": 60.0, "fmt": "{:.0f}", "suffix": "ms"},
    ],
    "meters": [
        {"key": "punch_db", "label": "PUNCH+", "mode": "gr",
         "lo": 0.0, "hi": 8.0},
        {"key": "squash_db", "label": "SQUASH", "mode": "gr",
         "lo": 0.0, "hi": 6.0},
    ],
}

CTRL = 32
GATE_LO = 2.4            # hit gate opens above this (vibrato-proof)
GATE_HI = 4.0            # ... fully open here
PROM_SPAN = 2.0          # dB of prominence over which damage fades out


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.amount = 3.0
        self.sens = 4.0
        self.speed_ms = 60.0
        # onset envelopes (power)
        self._a_f = float(np.exp(-1.0 / (0.002 * self.fs)))
        self._a_s = float(np.exp(-1.0 / (0.080 * self.fs)))
        self._zi_f = np.zeros(1)
        self._zi_s = np.zeros(1)
        # damage tracker (control rate): program-scale hit prominence
        self._prom = 0.0
        self._prom_dec = float(np.exp(-(CTRL / self.fs) / 2.5))
        self._a_dmg = float(np.exp(-(CTRL / self.fs) / 3.0))
        self._damage = 0.0
        # gain state + edge smoother
        self._g = 0.0
        self._a_sm = float(np.exp(-1.0 / (0.0007 * self.fs)))
        self._zi_sm = np.zeros(1)
        self.meters = {"punch_db": 0.0, "squash_db": 0.0}

    def set_params(self, amount=None, sens=None, speed_ms=None):
        if amount is not None:
            self.amount = float(np.clip(amount, 0.0, 10.0))
        if sens is not None:
            self.sens = float(np.clip(sens, 0.0, 10.0))
        if speed_ms is not None:
            self.speed_ms = float(np.clip(speed_ms, 20.0, 120.0))

    # ------------------------------------------------------------------ audio
    def process(self, x):
        n = len(x)
        p = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

        pf, self._zi_f = lfilter([1 - self._a_f], [1, -self._a_f], p,
                                 zi=self._zi_f)
        ps, self._zi_s = lfilter([1 - self._a_s], [1, -self._a_s], p,
                                 zi=self._zi_s)
        delta = 10.0 * np.log10(np.maximum(pf, 1e-14)
                                / np.maximum(ps, 1e-14))
        delta = np.maximum(delta, 0.0)

        dec = np.exp(-(CTRL / self.fs) / (self.speed_ms * 1e-3))
        prom_hi = GATE_LO + 0.6 * self.sens         # SENS 0 == structurally off
        g = self._g
        prom = self._prom
        dmg = self._damage
        gains = np.empty(n)
        i = 0
        while i < n:
            j = min(i + CTRL, n)
            dmax = float(delta[i:j].max())
            # program-scale cue: typical hit prominence (only real hits
            # vote - vibrato never opens the gate, so it never votes)
            if dmax > GATE_LO:
                prom = max(dmax, prom)
            prom = prom * self._prom_dec if prom > 0 else 0.0
            prom_f = float(np.clip((prom_hi - max(prom, GATE_LO))
                                   / PROM_SPAN, 0.0, 1.0))
            dmg = self._a_dmg * dmg + (1.0 - self._a_dmg) * prom_f
            # vibrato-proof onset gate: a detected hit earns the full
            # damage-scaled lift
            gate = float(np.clip((dmax - GATE_LO) / (GATE_HI - GATE_LO),
                                 0.0, 1.0))
            target = self.amount * dmg * gate
            g = target if target > g else max(target, g * dec)
            gains[i:j] = g
            i = j
        self._g = g
        self._prom = prom
        self._damage = dmg

        if self.amount <= 0.0 or (dmg < 0.03 and g < 0.01):
            self._zi_sm[:] = 0.0                 # flush the gain tail
            self._g = 0.0
            self.meters["punch_db"] = 0.0
            self.meters["squash_db"] = dmg * 6.0
            return x                             # bit-exact when idle

        gsm, self._zi_sm = lfilter([1 - self._a_sm], [1, -self._a_sm],
                                   gains, zi=self._zi_sm)
        self.meters["punch_db"] = float(np.max(gsm))
        self.meters["squash_db"] = dmg * 6.0
        return x * (10.0 ** (gsm / 20.0))[:, None]

    def reset(self):
        self._zi_f[:] = 0.0
        self._zi_s[:] = 0.0
        self._zi_sm[:] = 0.0
        self._prom = 0.0
        self._damage = 0.0
        self._g = 0.0
        self.meters = {"punch_db": 0.0, "squash_db": 0.0}
