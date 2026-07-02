"""
omegafm.dsp.dynamics
====================

All gain computers in the chain.  Real-time safe: no per-sample Python
loops at audio rate.

Key techniques
--------------
*  **Look-ahead limiter** - fully vectorised: dB gain target -> causal
   running minimum over the look-ahead window (numpy sliding_window_view
   over history+block) -> dB-linear release via a cumulative-minimum
   identity -> one-pole attack smoothing (lfilter w/ state) -> linear
   gain applied to delayed audio.  This is the "brick wall with a
   memory" that keeps 192 kHz peaks under the ceiling with no overshoot
   and no per-sample loop.
*  **Control-rate dynamics** (AGC / leveler / compressor) run at
   CONTROL_BLOCK = 32 samples (0.67 ms @ 48k).  Gains are computed in a
   short Python loop over control blocks (vectorised across bands),
   expanded with np.repeat, then de-zippered with a one-pole in dB.
   This matches how hardware processors update VCA control voltages.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter, sosfilt

CONTROL_BLOCK = 32          # samples per control update @ 48 kHz
_DB_FLOOR = 1e-9


def _to_db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), _DB_FLOOR))


def _release_cummin(gt_db: np.ndarray, prev: float, rate_db_per_sample: float):
    """Instant-attack / linear-dB-release envelope, vectorised.

    g[n] = min(gt[n], g[n-1] + r)  ==  min over k<=n of (gt[k] + (n-k)*r)
    computed with one cumulative minimum:
        aux[n] = gt[n] - n*r ;  g[n] = cummin(prev - r*(-1), aux)[n] + n*r
    """
    n = len(gt_db)
    idx = np.arange(n)
    aux = gt_db - idx * rate_db_per_sample
    carry = prev + rate_db_per_sample          # aligns to virtual index -1
    cm = np.minimum.accumulate(np.concatenate(([carry], aux)))[1:]
    g = cm + idx * rate_db_per_sample
    return np.minimum(g, 0.0), float(g[-1])


class LookaheadLimiter:
    """Stereo-linked look-ahead peak limiter with a *gain-riding* release.

    Broadcast limiters must not pump back to unity between every word or
    beat.  Release here uses the same recipe as the multiband compressor
    so the two stages glide together instead of fighting:

      platform : charges *down* toward the applied gain (~0.25 s EMA)
                 but recovers *up* only via a gentle ~0.8 dB/s leak -
                 the level the limiter rides at through a passage;
      fast     : dB-linear release (~90 dB/s) that recovers transient
                 GR quickly, but only back *down to the platform*.

    (Averaging the instantaneous demand instead - which is 0 between
    words - parks the platform near unity and forces the fast path to
    re-grab several dB on every syllable: audible as the limiter
    "fighting" the smooth stages before it.)

    Attack is instant via the lookahead running-min; a short one-pole
    rounds the gain edge and the downstream safety clip catches its
    sub-0.3 dB overshoot.  The riding recursion runs per 32-sample
    control chunk on top of the per-sample lookahead min.
    """

    CTRL = 32

    def __init__(self, fs: float, channels: int = 2, lookahead_ms: float = 2.0,
                 release_db_s: float = 90.0, ceiling_db: float = -0.5,
                 platform_ms: float = 250.0, platform_floor_db: float = -6.0,
                 platform_leak_db_s: float = 0.8):
        self.fs = fs
        self.channels = channels
        self.la = max(1, int(round(lookahead_ms * 1e-3 * fs)))
        self.release_rate = release_db_s / fs          # dB per sample
        self.ceiling_db = ceiling_db
        self.plat_floor = platform_floor_db
        self.leak_rate = platform_leak_db_s / fs       # dB per sample
        self.win_hist = np.zeros(self.la)              # gain-target history
        self.g_db = 0.0                                # applied gain state
        self.p_db = 0.0                                # platform state
        self.plat_alpha_ms = platform_ms
        self._recalc_plat_alpha()
        self.delay = np.zeros((self.la, channels))     # audio delay line
        tau = max(1.0, self.la / 3.0)                  # attack edge smoother
        self.alpha = float(np.exp(-1.0 / tau))
        self.smooth_zi = np.zeros(1)
        self.gr_meter = 0.0

    def _recalc_plat_alpha(self):
        self.plat_alpha = float(np.exp(-(self.CTRL / self.fs)
                                       / (self.plat_alpha_ms * 1e-3)))

    def set_params(self, ceiling_db=None, release_db_s=None, platform_ms=None):
        if ceiling_db is not None:
            self.ceiling_db = ceiling_db
        if release_db_s is not None:
            self.release_rate = release_db_s / self.fs
        if platform_ms is not None:
            self.plat_alpha_ms = max(platform_ms, 50.0)
            self._recalc_plat_alpha()

    def _ride(self, gmin: np.ndarray) -> np.ndarray:
        """Control-chunk riding recursion over the per-sample demand min."""
        n = len(gmin)
        out = np.empty(n)
        g, p = self.g_db, self.p_db
        pa = self.plat_alpha
        i = 0
        while i < n:
            j = min(i + self.CTRL, n)
            span = j - i
            t = float(gmin[i:j].min())                 # deepest demand, <= 0
            if t < g:
                g = t                                  # instant attack
            else:
                rt = min(t, p)                         # release only to platform
                if rt > g:
                    g = min(rt, g + self.release_rate * span)
            # platform: charges down toward deeper applied gain,
            # recovers up only through the slow leak
            if g < p:
                w = pa if span == self.CTRL else float(pa ** (span / self.CTRL))
                p = w * p + (1.0 - w) * g
            p = min(p + self.leak_rate * span, 0.0)
            p = max(p, self.plat_floor)
            out[i:j] = g
            i = j
        self.g_db, self.p_db = g, p
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        n = len(x)
        peak = np.max(np.abs(x), axis=1)
        gt = np.minimum(self.ceiling_db - _to_db(peak), 0.0)

        # causal running-min over look-ahead window (window = la+1)
        buf = np.concatenate([self.win_hist, gt])
        gmin = sliding_window_view(buf, self.la + 1).min(axis=1)
        self.win_hist = buf[-self.la:].copy()

        # gain-riding attack/release
        g_rel = self._ride(gmin)

        # one-pole attack smoothing in dB
        g_sm, self.smooth_zi = lfilter([1.0 - self.alpha], [1.0, -self.alpha],
                                       g_rel, zi=self.smooth_zi)
        self.gr_meter = max(0.0, float(-np.min(g_sm)))

        lin = 10.0 ** (g_sm / 20.0)

        # delay audio to line up with look-ahead
        joined = np.concatenate([self.delay, x], axis=0)
        self.delay = joined[-self.la:, :].copy()
        return joined[:n, :] * lin[:, None]

    def reset(self):
        self.win_hist[:] = 0
        self.g_db = 0.0; self.p_db = 0.0
        self.delay[:] = 0; self.smooth_zi[:] = 0; self.gr_meter = 0.0


class HFLimiter:
    """Dynamic HF controller after pre-emphasis.

    Complementary one-pole split (default ~4.5 kHz); the HP leg gets an
    instant-attack / exponential-release gain so emphasized sibilance
    and cymbals cannot drive the clipper into gross distortion.

    Two things keep it *inaudible* rather than phasey: the gain edge is
    rounded by a short (~0.4 ms) one-pole so the spectral tilt never
    snaps, and the release is slow enough (default 80 ms) that the tilt
    does not flutter word-to-word.  Staged correctly it only trims the
    occasional 1-3 dB; the WB limiter after it owns the peak guarantee.
    """

    def __init__(self, fs: float, channels: int = 2, split_hz: float = 4500.0,
                 threshold_db: float = -1.0, release_ms: float = 80.0,
                 attack_ms: float = 0.4):
        self.fs = fs
        self.channels = channels
        self.split_hz = None
        self._set_split(split_hz)
        self.lp_state = np.zeros(channels)
        self.threshold_db = threshold_db
        self.rel_alpha = float(np.exp(-1.0 / (release_ms * 1e-3 * fs)))
        self.att_alpha = float(np.exp(-1.0 / (attack_ms * 1e-3 * fs)))
        self.att_zi = np.zeros(1)
        self.g_state = 0.0                       # dB, <= 0
        self.gr_meter = 0.0
        # dynamic-shelf application (8100 style): the control signal
        # drives a gentle high-shelf CUT at 3.2 kHz plus a small (15%)
        # broadband component - it reads as program EQ tilting, not a
        # band being ducked, so the top keeps its sparkle
        self._sh_fs = fs
        self._sh_depth = 0.0
        self._sh_sos = self._design_shelf(0.0)
        self._sh_zi = np.zeros((1, 2, channels))

    def _design_shelf(self, gain_db: float):
        # RBJ high shelf @ 3200 Hz, S ~= 0.9
        fs = self._sh_fs
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * 3200.0 / fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / 2.0 * np.sqrt((A + 1 / A) * (1 / 0.9 - 1) + 2)
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

    def _set_split(self, hz: float):
        hz = float(np.clip(hz, 1000.0, 12000.0))
        if hz != self.split_hz:
            self.split_hz = hz
            self.a = float(np.exp(-2.0 * np.pi * hz / self.fs))

    def set_params(self, threshold_db=None, release_ms=None, split_hz=None):
        if threshold_db is not None:
            self.threshold_db = threshold_db
        if release_ms is not None:
            self.rel_alpha = float(np.exp(-1.0 / (max(release_ms, 5.0)
                                                  * 1e-3 * self.fs)))
        if split_hz is not None:
            self._set_split(split_hz)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        b0 = 1.0 - self.a
        lp, zf = lfilter([b0], [1.0, -self.a], x, axis=0,
                         zi=self.lp_state[None, :])
        self.lp_state = zf[0]
        hp = x - lp

        peak = np.max(np.abs(hp), axis=1)
        gt = np.minimum(self.threshold_db - _to_db(peak), 0.0)

        # instant attack, one-pole release toward 0 dB, loop-free:
        # g[n] = min(gt[n], a*g[n-1])  ->  h[n]=g[n]/a^n = cummin(gt[n]/a^n)
        n = len(gt)
        a = self.rel_alpha
        an = a ** np.arange(n)
        h = np.minimum.accumulate(
            np.concatenate(([self.g_state * a], gt / an)))[1:]
        genv = np.minimum(h * an, 0.0)
        self.g_state = float(genv[-1])

        # round the attack edges so the spectral tilt never snaps
        aa = self.att_alpha
        gsm, self.att_zi = lfilter([1.0 - aa], [1.0, -aa], genv,
                                   zi=self.att_zi)
        self.gr_meter = max(0.0, float(-np.min(gsm)))

        # apply as a chunked dynamic shelf (+15% broadband): the shelf
        # depth follows the deepest cut per ~1.3 ms chunk - still
        # effectively instant for ess protection.  Fast path: with no
        # gain reduction and the shelf parked at 0 dB the stage is a
        # mathematical no-op, so skip the filter entirely (the common
        # case on program between sibilant events).
        if (gsm.min() > -0.05 and self._sh_depth == 0.0
                and abs(self._sh_zi).max() < 1e-9):
            return x
        n = len(x)
        y = np.empty_like(x)
        i = 0
        while i < n:
            j = min(i + 256, n)
            depth = float(np.clip(np.min(gsm[i:j]), -18.0, 0.0))
            if abs(depth - self._sh_depth) > 0.25:
                self._sh_depth = depth
                self._sh_sos = self._design_shelf(depth)
            y[i:j], self._sh_zi = sosfilt(self._sh_sos, x[i:j], axis=0,
                                          zi=self._sh_zi)
            i = j
        lin = 10.0 ** (0.15 * gsm / 20.0)
        return y * lin[:, None]

    def reset(self):
        self.lp_state[:] = 0; self.g_state = 0.0
        self.att_zi[:] = 0; self.gr_meter = 0.0
        self._sh_depth = 0.0
        self._sh_sos = self._design_shelf(0.0)
        self._sh_zi = np.zeros((1, 2, self.channels))


try:                                        # optional acceleration:
    from numba import njit as _njit         # LLVM JIT via the C API -
    _HAVE_NUMBA = True                      # no C files to compile
except Exception:
    _HAVE_NUMBA = False

    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


@_njit(cache=True)
def _rider_loop(levels, gate_lv, g, target, window, rel, att, fast,
                drift, range_db, gate_db, differential, couple,
                gate_hold, out):
    # target/window/range_db/gate_db are per-band float64 arrays
    nblk, nb = levels.shape
    for i in range(nblk):
        m = 0.0
        live = 0
        for b in range(nb):
            if gate_lv[i, b] >= gate_db[b]:
                live += 1
        for b in range(nb):
            e = target[b] - levels[i, b] - g[b]
            out[i, b] = e
        if differential and live > 0:
            s = 0.0
            for b in range(nb):
                if gate_lv[i, b] >= gate_db[b]:
                    s += out[i, b]
            s /= live
            for b in range(nb):
                out[i, b] -= s
        for b in range(nb):
            e = out[i, b]
            mag = abs(e) - window[b]
            if mag < 0.0:
                mag = 0.0
            eff = mag if e > 0 else -mag
            if eff > 0:
                lim = fast if eff > 6.0 else rel
                step = eff if eff < lim else lim
            else:
                step = eff if eff > -att else -att
            if gate_lv[i, b] < gate_db[b]:
                if gate_hold:
                    pass
                else:
                    dstep = -g[b]
                    if dstep > drift:
                        dstep = drift
                    elif dstep < -drift:
                        dstep = -drift
                    g[b] += dstep
            else:
                g[b] += step
            if g[b] > range_db[b]:
                g[b] = range_db[b]
            elif g[b] < -range_db[b]:
                g[b] = -range_db[b]
        if differential:
            s = 0.0
            for b in range(nb):
                s += g[b]
            s /= nb
            for b in range(nb):
                g[b] -= s
        if couple >= 0.0 and nb > 1:
            m = 0.0
            if not differential:
                for b in range(nb):
                    m += g[b]
                m /= nb
            for b in range(nb):
                if g[b] > m + couple:
                    g[b] = m + couple
                elif g[b] < m - couple:
                    g[b] = m - couple
        for b in range(nb):
            out[i, b] = g[b]


class GainRider:
    """Slow gated gain-riding AGC - also used per-band as the leveler.

    The detector is a *true integrated loudness*: control-block power is
    smoothed by a one-pole (default ~300 ms) in the power domain before
    the gain law ever sees it.  That is what makes the ride smooth -
    individual hits and the gaps between them are invisible; only the
    average balance of a passage moves the gain.  (Without integration a
    percussive band's instantaneous level collapses between hits, the
    gain random-walks up every gap and pins at +range - the classic
    "leveler chews the snare tails" artifact.)

    Gain slews at att/rel dB/s toward target; freezes below the gate
    (evaluated on the integrated level) and drifts slowly back to 0 dB.
    Range limited (AGC +/-12 dB, leveler +/-7 dB).  Multi-band capable:
    pass bands as (N, B) level input.

    Multiband mix protection (leveler): with differential=True only the
    *shape* error is corrected - the common (mean) error across bands is
    removed each step, because overall level is the wideband AGC's job.
    couple_db then clamps every band's gain to within +/-couple of the
    pack mean, so no band can ever wander off and repaint the mix
    (violins fading while bass climbs on classical material).  The
    common gain rides together; only bounded, slow relative correction
    remains - transparent to the balance of the source.
    """

    def __init__(self, fs: float, nbands: int = 1, target_db: float = -17.0,
                 range_db: float = 12.0, attack_db_s: float = 2.5,
                 release_db_s: float = 1.2, gate_db: float = -45.0,
                 fast_recovery_db_s: float = 8.0, integrate_ms: float = 300.0,
                 window_db: float = 1.5, differential: bool = False,
                 couple_db: float | None = None, gate_hold: bool = False,
                 gate_fast: bool = False):
        self.fs = fs
        self.nbands = nbands
        self.target = np.full(nbands, target_db, dtype=np.float64)
        self.range_db = range_db
        self.att = attack_db_s * CONTROL_BLOCK / fs
        self.rel = release_db_s * CONTROL_BLOCK / fs
        self.fast = fast_recovery_db_s * CONTROL_BLOCK / fs
        self.gate_db = gate_db
        self.window = window_db
        self.differential = bool(differential) and nbands > 1
        self.couple = couple_db if couple_db is not None else None
        self.gain = np.zeros(nbands)
        self.drift = 0.05 * CONTROL_BLOCK / fs        # dB/ctrl toward 0 when gated
        self._tblk = CONTROL_BLOCK / fs
        self.int_alpha = float(np.exp(-self._tblk / (integrate_ms * 1e-3)))
        self.int_zi = np.zeros((1, nbands))
        # broadcast-AGC gating: decide the gate on a FAST detector so
        # silence freezes the gain before the slow loudness integrator
        # ever chases it, and hold rock-solid (no drift) while gated
        self.gate_hold = bool(gate_hold)
        self.gate_fast = bool(gate_fast)
        self.gate_alpha = float(np.exp(-self._tblk / 0.030))
        self.gate_zi = np.zeros((1, nbands))
        self.int_state = np.full(nbands, 1e-12)   # gate-frozen integrator
        # de-zipper
        self.sm_alpha = float(np.exp(-1.0 / (0.003 * fs)))
        self.sm_zi = np.zeros((1, nbands))
        self.meter = np.zeros(nbands)

    def set_params(self, target_db=None, range_db=None, attack_db_s=None,
                   release_db_s=None, gate_db=None, fast_db_s=None,
                   integrate_ms=None, window_db=None, couple_db=None):
        if target_db is not None:
            self.target = np.broadcast_to(np.asarray(target_db, dtype=np.float64),
                                          (self.nbands,)).copy()
        if range_db is not None: self.range_db = range_db
        if attack_db_s is not None: self.att = attack_db_s * CONTROL_BLOCK / self.fs
        if release_db_s is not None: self.rel = release_db_s * CONTROL_BLOCK / self.fs
        if gate_db is not None: self.gate_db = gate_db
        if fast_db_s is not None: self.fast = fast_db_s * CONTROL_BLOCK / self.fs
        if integrate_ms is not None:
            self.int_alpha = float(np.exp(-self._tblk
                                          / (max(integrate_ms, 10.0) * 1e-3)))
        if window_db is not None:
            self.window = window_db
        if couple_db is not None:
            self.couple = float(couple_db)

    def compute_gains(self, levels: np.ndarray) -> np.ndarray:
        """levels: (nblk, nbands) RMS dB per control block -> gains (nblk, nbands) dB."""
        # ---- integrated loudness (power-domain one-pole, vectorised)
        pwr = 10.0 ** (levels / 10.0)
        ia = self.int_alpha
        if self.gate_fast:
            ga = self.gate_alpha
            gpwr, self.gate_zi = lfilter([1.0 - ga], [1.0, -ga], pwr,
                                         axis=0, zi=self.gate_zi)
            gate_lv = 10.0 * np.log10(np.maximum(gpwr, 1e-12))
            # loudness integration FREEZES while gated, so the gain
            # returns to a correct reading after a gap - no re-settle
            st = self.int_state
            lv_out = np.empty_like(pwr)
            for k in range(pwr.shape[0]):
                if gate_lv[k, 0] >= self.gate_db:
                    st = ia * st + (1.0 - ia) * pwr[k]
                lv_out[k] = st
            self.int_state = st
            levels = 10.0 * np.log10(np.maximum(lv_out, 1e-12))
        else:
            pwr, self.int_zi = lfilter([1.0 - ia], [1.0, -ia], pwr,
                                       axis=0, zi=self.int_zi)
            levels = 10.0 * np.log10(np.maximum(pwr, 1e-12))
            gate_lv = levels

        nblk = levels.shape[0]
        out = np.empty_like(levels)
        if _HAVE_NUMBA:
            nb = levels.shape[1]

            def _vec(v):
                return np.ascontiguousarray(
                    np.broadcast_to(np.asarray(v, dtype=np.float64),
                                    (nb,)).copy())
            g = np.ascontiguousarray(self.gain, dtype=np.float64)
            _rider_loop(np.ascontiguousarray(levels, dtype=np.float64),
                        np.ascontiguousarray(gate_lv, dtype=np.float64),
                        g, _vec(self.target), _vec(self.window),
                        float(self.rel), float(self.att), float(self.fast),
                        float(self.drift), _vec(self.range_db),
                        _vec(self.gate_db), bool(self.differential),
                        float(self.couple) if self.couple is not None
                        else -1.0, bool(self.gate_hold), out)
            self.gain = g
            self.meter = g.copy()
            return out
        g = self.gain
        for i in range(nblk):
            lv = levels[i]
            gated = gate_lv[i] < self.gate_db
            err = self.target - lv - g          # positive => need more gain
            if self.differential:
                # shape-only correction: remove the common error (the
                # AGC owns overall level); gated bands don't vote
                live = ~gated
                if live.any():
                    err = err - err[live].mean()
            # correction window: hold still within +/-window of target -
            # only the *excess* beyond the window is slewed out, so the
            # gain parks at the window edge instead of hunting.
            mag = np.maximum(np.abs(err) - self.window, 0.0)
            eff = np.sign(err) * mag
            step = np.where(eff > 0,
                            np.minimum(eff, np.where(eff > 6.0, self.fast, self.rel)),
                            np.maximum(eff, -self.att))
            drift_step = 0.0 if self.gate_hold \
                else np.clip(-g, -self.drift, self.drift)
            g = np.where(gated, g + drift_step, g + step)
            g = np.clip(g, -self.range_db, self.range_db)
            if self.differential:
                g = g - g.mean()                # zero net gain: shape only
            if self.couple is not None and self.nbands > 1:
                m = 0.0 if self.differential else g.mean()
                g = np.clip(g, m - self.couple, m + self.couple)
            out[i] = g
        self.gain = g
        self.meter = g.copy()
        return out

    def apply(self, bands, levels_db):
        """bands: list of (N,C) arrays (or single). levels_db: (nblk,B)."""
        gains_db = self.compute_gains(levels_db)                    # (nblk,B)
        gains = np.repeat(gains_db, CONTROL_BLOCK, axis=0)          # (N,B)
        gains, self.sm_zi = lfilter([1 - self.sm_alpha], [1, -self.sm_alpha],
                                    gains, axis=0, zi=self.sm_zi)
        lin = 10.0 ** (gains / 20.0)
        if isinstance(bands, list):
            return [b * lin[:, i][:, None] for i, b in enumerate(bands)]
        return bands * lin[:, 0][:, None]

    def reset(self):
        self.gain[:] = 0; self.sm_zi[:] = 0; self.int_zi[:] = 0
        self.meter[:] = 0


class BandCompressor:
    """4-band program compressor - the 'sonic signature' stage.

    Soft-knee, per-band ratio / threshold / attack / release, computed
    at control rate with peak detection, vectorised across bands.

    The release is *program-dependent* (same riding philosophy as the
    WB limiter): each band's GR releases at its normal speed only down
    to a slowly-gliding platform - an averaged GR that leaks toward
    0 dB at ~1.2 dB/s.  Between drum hits or words the gain therefore
    holds near its working depth instead of collapsing to unity and
    slamming back, which is what chews decay tails and pumps.  The
    attack side and the static curve are untouched, so the density
    character ("signature") is preserved.
    """

    def __init__(self, fs: float, nbands: int = 4):
        self.fs = fs
        self.nbands = nbands
        self.threshold = np.full(nbands, -24.0)
        self.ratio = np.full(nbands, 3.0)
        self.knee = 6.0
        self.att_alpha = np.zeros(nbands)
        self.rel_alpha = np.zeros(nbands)
        self.set_times([18, 10, 7, 4], [280, 200, 150, 110])
        self.gr = np.zeros(nbands)
        self.platform = np.zeros(nbands)
        tblk = CONTROL_BLOCK / fs
        self.plat_alpha = float(np.exp(-tblk / 0.7))       # ~700 ms glide
        self.plat_leak = 1.2 * tblk                        # dB/s toward 0
        self.sm_alpha = float(np.exp(-1.0 / (0.0015 * fs)))
        self.sm_zi = np.zeros((1, nbands))
        self.meter = np.zeros(nbands)
        self.makeup = np.zeros(nbands)

    def set_times(self, attack_ms, release_ms):
        tblk = CONTROL_BLOCK / self.fs
        self.att_alpha = np.exp(-tblk / (np.asarray(attack_ms) * 1e-3))
        self.rel_alpha = np.exp(-tblk / (np.asarray(release_ms) * 1e-3))

    def set_params(self, threshold_db=None, ratio=None, attack_ms=None,
                   release_ms=None, makeup_db=None, knee_db=None,
                   platform_ms=None):
        if threshold_db is not None:
            self.threshold = np.asarray(threshold_db, dtype=np.float64).copy()
        if ratio is not None:
            self.ratio = np.asarray(ratio, dtype=np.float64).copy()
        if attack_ms is not None and release_ms is not None:
            self.set_times(attack_ms, release_ms)
        if makeup_db is not None:
            self.makeup = np.asarray(makeup_db, dtype=np.float64).copy()
        if knee_db is not None:
            self.knee = knee_db
        if platform_ms is not None:
            tblk = CONTROL_BLOCK / self.fs
            self.plat_alpha = float(np.exp(-tblk
                                           / (max(platform_ms, 50.0) * 1e-3)))

    def _static_gr(self, lv):
        """Soft-knee static curve, vectorised.  lv, out: dB."""
        over = lv - self.threshold
        k = self.knee
        inv = 1.0 - 1.0 / self.ratio
        gr = np.where(over <= -k / 2, 0.0,
             np.where(over >= k / 2, inv * over,
                      inv * (over + k / 2) ** 2 / (2 * k)))
        return -gr

    def apply(self, bands, levels_db):
        nblk = levels_db.shape[0]
        target = self._static_gr(levels_db)                # (nblk, B) <= 0
        out = np.empty_like(target)
        g = self.gr
        p = self.platform
        pa, leak = self.plat_alpha, self.plat_leak
        for i in range(nblk):
            t = target[i]
            attacking = t < g
            # release only down to the platform; attack straight to demand
            rt = np.where(attacking, t, np.minimum(t, p))
            alpha = np.where(attacking, self.att_alpha, self.rel_alpha)
            g = alpha * g + (1 - alpha) * rt
            # platform: slow EMA of applied GR, leaking gently toward 0
            p = pa * p + (1 - pa) * g
            p = np.minimum(p + leak, 0.0)
            out[i] = g
        self.gr = g
        self.platform = p
        self.meter = -g
        gains = np.repeat(out + self.makeup, CONTROL_BLOCK, axis=0)
        gains, self.sm_zi = lfilter([1 - self.sm_alpha], [1, -self.sm_alpha],
                                    gains, axis=0, zi=self.sm_zi)
        lin = 10.0 ** (gains / 20.0)
        return [b * lin[:, i][:, None] for i, b in enumerate(bands)]

    def reset(self):
        self.gr[:] = 0; self.platform[:] = 0
        self.sm_zi[:] = 0; self.meter[:] = 0


# --------------------------------------------------------------------------- #
#  Level measurement helpers (control rate)
# --------------------------------------------------------------------------- #

def block_rms_db(x: np.ndarray) -> np.ndarray:
    """(N,C) -> (N/CB,) RMS dB across channels per control block."""
    n = len(x)
    nblk = n // CONTROL_BLOCK
    m = np.mean(x[:nblk * CONTROL_BLOCK] ** 2, axis=1)
    m = m.reshape(nblk, CONTROL_BLOCK).mean(axis=1)
    return 10.0 * np.log10(np.maximum(m, _DB_FLOOR))


def block_peak_db(x: np.ndarray) -> np.ndarray:
    n = len(x)
    nblk = n // CONTROL_BLOCK
    p = np.max(np.abs(x[:nblk * CONTROL_BLOCK]), axis=1)
    p = p.reshape(nblk, CONTROL_BLOCK).max(axis=1)
    return _to_db(p)


# --------------------------------------------------------------------------- #
#  Clippers
# --------------------------------------------------------------------------- #

def cubic_clip(x: np.ndarray, ceiling: float = 1.0) -> np.ndarray:
    """8100-style cubic soft saturator: unity below, smooth cubic bend,
    hard ceiling reached at 1.5x drive.  Odd (3rd) harmonic only -
    the 'analog' clip character."""
    y = np.clip(x / ceiling, -1.5, 1.5)
    y = y - (4.0 / 27.0) * y * y * y
    return y * ceiling


def knee_clip(x: np.ndarray, threshold: float = 1.0, knee: float = 0.2) -> np.ndarray:
    """Soft-knee clipper: linear below (thr-knee), smooth quadratic into
    hard limit at thr.  knee=0 -> hard clip."""
    if knee <= 0.0:
        return np.clip(x, -threshold, threshold)
    t = threshold
    k = knee * t
    a = np.abs(x)
    s = np.sign(x)
    lin = a <= (t - k)
    hard = a >= (t + k)
    y = np.where(lin, a, np.where(hard, t, a - (a - (t - k)) ** 2 / (4 * k)))
    return s * y


class BassEnhancer:
    """Psycho-acoustic bass: mono sub extraction (<90 Hz LR4), tanh
    harmonic generation band-passed 60-300 Hz mixed back, plus a clean
    low shelf.  Small speakers 'hear' the fundamental through its
    harmonic series; big ones get the shelf."""

    def __init__(self, fs: float, channels: int = 2):
        from scipy.signal import butter as _butter
        from .filters import SOSFilter
        self.sub_lp = SOSFilter(_butter(4, 90.0, btype="low", fs=fs, output="sos"), 1)
        self.harm_bp = SOSFilter(_butter(2, [60.0, 300.0], btype="band", fs=fs, output="sos"), 1)
        self.shelf = None
        self.fs = fs
        self.channels = channels
        self.shelf_db = None
        self.harm_amount = 0.3
        self.drive = 2.0
        self.set_shelf(3.0)

    def set_shelf(self, gain_db: float):
        if gain_db == self.shelf_db:
            return
        from .filters import SOSFilter, rbj_lowshelf
        self.shelf_db = gain_db
        old = self.shelf
        self.shelf = SOSFilter(rbj_lowshelf(self.fs, 90.0, gain_db), self.channels)
        if old is not None:
            self.shelf.zi = old.zi * 0.0

    def set_params(self, shelf_db=None, harmonics=None):
        if shelf_db is not None:
            self.set_shelf(shelf_db)
        if harmonics is not None:
            self.harm_amount = harmonics

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        mono = np.mean(x, axis=1, keepdims=True)
        sub = self.sub_lp.process(mono)
        harm = np.tanh(sub * self.drive) / self.drive
        harm = self.harm_bp.process(harm - sub)          # harmonics only
        y = self.shelf.process(x)
        return y + self.harm_amount * harm

    def reset(self):
        self.sub_lp.reset(); self.harm_bp.reset()
        if self.shelf: self.shelf.reset()
