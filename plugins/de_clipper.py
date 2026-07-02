"""OmegaFM plugin: De-Clipper / De-Lossifier (Omnia "Undo" style).

Source repair - the first thing that should ever happen to program
audio, before any processing touches it. Two engines:

DE-CLIPPER
    Loudness-war masters arrive already hard-clipped: flat-topped
    waveforms whose gritty distortion gets magnified by pre-emphasis
    and the final clipper. This stage tracks the source's own clip
    ceiling, detects flat-top runs, and *rebuilds the missing peak
    arches* with cubic Hermite reconstruction through the healthy
    neighbouring samples (soft-capped at +6 dB over the source
    ceiling). Restored crest means the whole chain works on healthy
    waveforms again. A small 4 ms internal buffer provides the
    right-hand context reconstruction needs.

DE-LOSSIFIER
    Low-bitrate codecs shelve off the top octave. The stage measures
    the actual 11-15 kHz energy against what the 5.5-9 kHz donor band
    predicts for natural material; the measured *deficit* drives a
    harmonic regenerator (rectifier exciter on the donor band,
    band-limited back into 11-15 kHz, level-matched). Deficit-driven
    means a clean full-bandwidth source receives essentially nothing -
    it only restores what was actually removed.

Inserted post_input (before the AGC - exactly where Omnia puts Undo),
and by design it runs ahead of any enhancers at the same point:
repair first, then enhance.

Controls
--------
DECLIP  0..10  reconstruction amount (default 5; 0 = detect only)
LOSSY   0..10  HF restoration amount (default 3)
TEST    0/1    commissioning aid: hard-clips the module's own input at
               half level so loudness-war damage is audible on any
               source - REPAIR lights up and the DECLIP knob audibly
               heals the grit (set DECLIP 0 to hear the raw damage).
               Never leave on for air.

Meters: REPAIR = dB of peak lift being applied to rebuilt samples,
DEFICIT = measured top-octave shortfall vs the donor prediction.
"""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter

PLUGIN = {
    "id": "de_clipper",
    "name": "De-Clipper / De-Lossifier",
    "version": "1.1",
    "insert": "post_input",
    "params": [
        {"key": "declip", "label": "DECLIP", "min": 0.0, "max": 10.0,
         "default": 5.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "lossy", "label": "LOSSY", "min": 0.0, "max": 10.0,
         "default": 3.0, "fmt": "{:.1f}", "suffix": ""},
        {"key": "test", "label": "TEST", "min": 0.0, "max": 1.0,
         "default": 0.0, "fmt": "{:.0f}", "suffix": ""},
    ],
    "meters": [
        {"key": "repair_db", "label": "REPAIR", "mode": "gr",
         "lo": 0.0, "hi": 6.0},
        {"key": "deficit_db", "label": "DEFICIT", "mode": "gr",
         "lo": 0.0, "hi": 18.0},
    ],
}

LAT = 192                 # internal context buffer (4 ms @ 48 kHz)
GUARD = 96                # right-hand healthy context required
MAXRUN = 80               # longest flat-top we attempt to rebuild
CAP_DB = 6.0              # reconstruction ceiling over the source clip level
NAT_TILT = 5.5            # natural donor->target rolloff (dB)


def _hermite(p0, m0, p1, m1, n):
    """Cubic Hermite between p0/p1 with tangents m0/m1 over n points
    (excluding the endpoints)."""
    t = (np.arange(1, n + 1)) / (n + 1)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


class Plugin:
    def __init__(self, fs, channels=2):
        self.fs = float(fs)
        self.channels = int(channels)
        self.declip = 5.0
        self.lossy = 3.0
        self.test = 0.0
        self._test_c = 0.1
        # de-clipper state
        self._pend = np.zeros((LAT, self.channels))
        self._ceil = np.full(self.channels, 0.1)
        self._ceil_decay = float(np.exp(-1.0 / (0.5 * self.fs)))
        self._rep_sm = 0.0
        # de-lossifier filters / states
        self.sos_don = butter(2, [5500 / (self.fs / 2), 9000 / (self.fs / 2)],
                              btype="band", output="sos")
        self.sos_tgt = butter(2, [11000 / (self.fs / 2),
                                  15000 / (self.fs / 2)],
                              btype="band", output="sos")
        self.sos_inj = butter(6, [11200 / (self.fs / 2),
                                  15000 / (self.fs / 2)],
                              btype="band", output="sos")
        self._zi_don = np.zeros((self.sos_don.shape[0], 2, self.channels))
        self._zi_tgt = np.zeros((self.sos_tgt.shape[0], 2, self.channels))
        self._zi_inj = np.zeros((self.sos_inj.shape[0], 2, self.channels))
        a_env = float(np.exp(-1.0 / (0.05 * self.fs)))
        self._a_env = a_env
        self._a_slow = float(np.exp(-1.0 / (0.4 * self.fs)))
        self._zi_pd = np.zeros(1)
        self._zi_pt = np.zeros(1)
        self._zi_pi = np.zeros(1)
        self._zi_pds = np.zeros(1)
        self._zi_pts = np.zeros(1)
        self._a_def = float(np.exp(-1.0 / (0.3 * self.fs)))
        self._zi_def = np.zeros(1)
        self.meters = {"repair_db": 0.0, "deficit_db": 0.0}

    def set_params(self, declip=None, lossy=None, test=None):
        if declip is not None:
            self.declip = float(np.clip(declip, 0.0, 10.0))
        if lossy is not None:
            self.lossy = float(np.clip(lossy, 0.0, 10.0))
        if test is not None:
            self.test = float(np.clip(test, 0.0, 1.0))

    # ------------------------------------------------------------------ declip
    def _declip_channel(self, buf, ch, amt):
        """Repair flat-top runs in buf[:, ch] whose right context ends
        before len(buf) - GUARD. Returns total dB-lift accumulator."""
        c = self._ceil[ch]
        s = buf[:, ch]
        n = len(s)
        # update ceiling from this buffer (decayed running max)
        blk_max = float(np.max(np.abs(s[LAT:]))) if n > LAT else 0.0
        c = max(blk_max, c * self._ceil_decay ** (n - LAT if n > LAT else 1))
        self._ceil[ch] = max(c, 1e-3)
        if c < 0.25 or amt <= 0.0:
            return 0.0
        cap = min(c * 10 ** (CAP_DB / 20.0), 1.995)
        near = np.abs(s) >= 0.985 * c
        lift_acc, lift_n = 0.0, 0
        i = 2
        end = n - GUARD
        while i < end:
            if not near[i]:
                i += 1
                continue
            j = i
            while j < n - 2 and near[j]:
                j += 1
            run = j - i
            if 3 <= run <= MAXRUN and not near[i - 1] and not near[j] \
                    and j + 2 < n:
                seg = s[i:j]
                if np.all(seg > 0) or np.all(seg < 0):
                    flat = (seg.max() - seg.min()) < 0.03 * c
                    if flat:
                        p0, p1 = s[i - 1], s[j]
                        m0 = (s[i - 1] - s[i - 2]) * (run + 1)
                        m1 = (s[j + 1] - s[j]) * (run + 1)
                        arch = _hermite(p0, m0, p1, m1, run)
                        sgn = 1.0 if seg[0] > 0 else -1.0
                        mag = np.abs(arch)
                        over = mag > c
                        soft = c + (cap - c) * np.tanh(
                            (mag - c) / max(cap - c, 1e-6))
                        mag = np.where(over, soft, mag)
                        arch = sgn * np.maximum(mag, np.abs(seg))
                        arch_pk = float(np.max(np.abs(arch)))
                        seg_pk = float(np.max(np.abs(seg))) + 1e-12
                        if arch_pk > 1.02 * seg_pk:
                            # a real flat-top: the rebuilt arch clearly
                            # exceeds the plateau. Natural crests (arch
                            # == true curve) are left bit-untouched.
                            newseg = seg + amt * (sgn * np.abs(arch) - seg)
                            lift = float(np.max(np.abs(newseg)) / seg_pk)
                            if lift > 1.0:
                                lift_acc += 20.0 * np.log10(lift)
                                lift_n += 1
                            s[i:j] = newseg
            i = j + 1
        return lift_acc / lift_n if lift_n else 0.0

    # ------------------------------------------------------------------ audio
    def process(self, x):
        n = len(x)
        if self.test >= 0.5:
            # commissioning: hard-clip the input at ~half its own level so
            # you can HEAR loudness-war damage appear - and DECLIP heal it
            blk = float(np.max(np.abs(x))) if len(x) else 0.0
            self._test_c = max(0.995 * self._test_c, 0.5 * blk, 1e-3)
            x = np.clip(x, -self._test_c, self._test_c)
        buf = np.concatenate([self._pend, x], axis=0)

        amt = self.declip / 10.0
        lifts = []
        for ch in range(self.channels):
            lifts.append(self._declip_channel(buf, ch, amt))
        lift = float(np.mean(lifts))
        self._rep_sm = 0.85 * self._rep_sm + 0.15 * lift
        self.meters["repair_db"] = self._rep_sm

        out = buf[:n].copy()
        self._pend = buf[n:].copy()

        # ---- de-lossifier -------------------------------------------------
        # detection uses mean-of-channel-powers (immune to interchannel
        # phase; a mono sum combs and skews the tilt)
        don, self._zi_don = sosfilt(self.sos_don, out, axis=0,
                                    zi=self._zi_don)
        tgt, self._zi_tgt = sosfilt(self.sos_tgt, out, axis=0,
                                    zi=self._zi_tgt)
        dsq = 0.5 * (don[:, 0] ** 2 + don[:, 1] ** 2)
        tsq = 0.5 * (tgt[:, 0] ** 2 + tgt[:, 1] ** 2)
        pd, self._zi_pd = lfilter([1 - self._a_env], [1, -self._a_env],
                                  dsq, zi=self._zi_pd)
        pds, self._zi_pds = lfilter([1 - self._a_slow], [1, -self._a_slow],
                                    dsq, zi=self._zi_pds)
        pts, self._zi_pts = lfilter([1 - self._a_slow], [1, -self._a_slow],
                                    tsq, zi=self._zi_pts)
        don_db = 10.0 * np.log10(np.maximum(pd, 1e-14))
        don_db_s = 10.0 * np.log10(np.maximum(pds, 1e-14))
        tgt_db_s = 10.0 * np.log10(np.maximum(pts, 1e-14))
        deficit = np.maximum((don_db_s - NAT_TILT) - tgt_db_s, 0.0)
        deficit, self._zi_def = lfilter([1 - self._a_def],
                                        [1, -self._a_def],
                                        deficit, zi=self._zi_def)
        self.meters["deficit_db"] = float(deficit[-1])

        g_knob = self.lossy / 10.0
        if g_knob > 0.0:
            # rectifier exciter on the donor band -> band-limit to target
            exc = np.abs(don)
            exc -= exc.mean(axis=0, keepdims=True)
            inj, self._zi_inj = sosfilt(self.sos_inj, exc, axis=0,
                                        zi=self._zi_inj)
            isq = 0.5 * (inj[:, 0] ** 2 + inj[:, 1] ** 2)
            pi_, self._zi_pi = lfilter([1 - self._a_env],
                                       [1, -self._a_env],
                                       isq, zi=self._zi_pi)
            inj_db = 10.0 * np.log10(np.maximum(pi_, 1e-14))
            frac = np.clip(deficit / 10.0, 0.0, 1.0) * g_knob
            want_db = (don_db - NAT_TILT) + 10.0 * np.log10(
                np.maximum(frac, 1e-6))
            gain = 10.0 ** (np.clip(want_db - inj_db, -80.0, 40.0) / 20.0)
            out = out + inj * gain[:, None]

        return out

    def reset(self):
        self._pend[:] = 0.0
        self._ceil[:] = 0.1
        self._rep_sm = 0.0
        self._zi_don[:] = 0.0
        self._zi_tgt[:] = 0.0
        self._zi_inj[:] = 0.0
        self._zi_pd[:] = 0.0
        self._zi_pt[:] = 0.0
        self._zi_pi[:] = 0.0
        self._zi_pds[:] = 0.0
        self._zi_pts[:] = 0.0
        self._zi_def[:] = 0.0
        self.meters = {"repair_db": 0.0, "deficit_db": 0.0}
