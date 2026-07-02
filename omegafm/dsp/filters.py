"""
omegafm.dsp.filters
===================

Broadcast-grade filter building blocks.

Everything here is *stateful* and block-based so it can run in a realtime
audio callback without clicks at block boundaries.  All filters process
float64 arrays shaped (N,) mono or (N, C) multichannel.

Design notes (why it is built this way):

*  The 4-band crossover is a serial Linkwitz-Riley 4th-order tree
   (splits at 1 kHz, then 160 Hz / 5 kHz) with all-pass compensation on
   the opposite branches.  The band sum is magnitude-flat (verified in
   tools/validate_chain.py to < 0.2 dB) - exactly how analog broadcast
   processors keep the recombined spectrum honest.
*  Pre-emphasis is a true bilinear-transformed analog 75/50 us network
   run at 192 kHz, where the compensation pole (~80 kHz) does not
   disturb the 0-15 kHz audio band.  Running it at 48 kHz would warp
   the curve near 15 kHz - the classic mistake software processors make.
*  The 15 kHz brick-wall and the 53 kHz MPX mask are linear-phase FIRs:
   in the composite domain every component (pilot / 38 kHz DSB / 57 kHz
   RDS) must see identical group delay or stereo separation collapses.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, lfilter, butter, firwin, firwin2, kaiserord


# --------------------------------------------------------------------------- #
#  Generic stateful SOS / FIR wrappers
# --------------------------------------------------------------------------- #

class SOSFilter:
    """Cascaded second-order sections with state, N channels."""

    def __init__(self, sos: np.ndarray, channels: int = 2):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.channels = channels
        nsect = self.sos.shape[0]
        self.zi = np.zeros((nsect, 2, channels))         # axis=0 layout

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y

    def reset(self):
        self.zi[:] = 0.0


class FIRFilter:
    """Streaming linear-phase FIR via lfilter with state."""

    def __init__(self, taps: np.ndarray, channels: int = 2):
        self.taps = np.asarray(taps, dtype=np.float64)
        self.channels = channels
        self.zi = np.zeros((len(self.taps) - 1, channels))

    @property
    def delay(self) -> int:
        return (len(self.taps) - 1) // 2

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = lfilter(self.taps, [1.0], x, axis=0, zi=self.zi)
        return y

    def reset(self):
        self.zi[:] = 0.0


class Delay:
    """Exact integer sample delay (matched-delay lines for FIR paths)."""

    def __init__(self, samples: int, channels: int = 2):
        self.n = int(samples)
        self.channels = channels
        self.buf = np.zeros((self.n, channels)) if self.n > 0 else None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return x
        if x.ndim == 1:
            x = x[:, None]
        joined = np.concatenate([self.buf, x], axis=0)
        self.buf = joined[-self.n:, :].copy()
        return joined[:len(x), :]

    def reset(self):
        if self.buf is not None:
            self.buf[:] = 0.0


# --------------------------------------------------------------------------- #
#  RBJ biquad designs (Audio EQ Cookbook)
# --------------------------------------------------------------------------- #

def rbj_peaking(fs: float, f0: float, gain_db: float, q: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cw
    a2 = 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def rbj_lowshelf(fs: float, f0: float, gain_db: float, slope: float = 0.9) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    two_sqA_alpha = 2 * np.sqrt(A) * alpha
    b0 = A * ((A + 1) - (A - 1) * cw + two_sqA_alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - two_sqA_alpha)
    a0 = (A + 1) + (A - 1) * cw + two_sqA_alpha
    a1 = -2 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - two_sqA_alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def allpass1(fs: float, f0: float) -> np.ndarray:
    """First-order all-pass as an SOS row (b2=a2=0)."""
    t = np.tan(np.pi * f0 / fs)
    c = (t - 1.0) / (t + 1.0)
    return np.array([[c, 1.0, 0.0, 1.0, c, 0.0]])


def allpass2_lr(fs: float, f0: float) -> np.ndarray:
    """2nd-order all-pass matching the phase of an LR4 crossover at f0.

    An LR4 split introduces the phase of a 2nd-order Butterworth all-pass
    at the crossover frequency; branches that skip a split get this to
    stay time-aligned with branches that took it.
    """
    sos = butter(2, f0, btype="low", fs=fs, output="sos")[0]
    a = sos[3:]                      # 1, a1, a2
    b = a[::-1]                      # a2, a1, 1  -> all-pass numerator
    return np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])


# --------------------------------------------------------------------------- #
#  4-band Linkwitz-Riley crossover tree (160 / 1k / 5k) with AP compensation
# --------------------------------------------------------------------------- #

class Crossover4:
    """
    Serial LR4 tree:

        in ── LR4@1k ──┬─ low  ── AP2(5k) ── LR4@160 ──┬─ band0 (LO   <160)
                       │                               └─ band1 (LM 160-1k)
                       └─ high ── AP2(160) ── LR4@5k ──┬─ band2 (HM 1k-5k)
                                                       └─ band3 (HI   >5k)

    The AP2 stages equalise the phase each branch missed so the four
    bands sum flat.  Fixed frequencies match the original OmegaFM
    (160 Hz / 1 kHz / 5 kHz).
    """

    FREQS = (160.0, 1000.0, 5000.0)

    def __init__(self, fs: float, channels: int = 2):
        f_lo, f_mid, f_hi = self.FREQS

        def lr4(f, btype):
            b2 = butter(2, f, btype=btype, fs=fs, output="sos")
            return np.vstack([b2, b2])          # LR4 = Butterworth-2 squared

        self.lp_mid = SOSFilter(lr4(f_mid, "low"), channels)
        self.hp_mid = SOSFilter(lr4(f_mid, "high"), channels)
        self.ap_low_branch = SOSFilter(allpass2_lr(fs, f_hi), channels)
        self.ap_high_branch = SOSFilter(allpass2_lr(fs, f_lo), channels)
        self.lp_lo = SOSFilter(lr4(f_lo, "low"), channels)
        self.hp_lo = SOSFilter(lr4(f_lo, "high"), channels)
        self.lp_hi = SOSFilter(lr4(f_hi, "low"), channels)
        self.hp_hi = SOSFilter(lr4(f_hi, "high"), channels)

    def process(self, x: np.ndarray):
        low = self.ap_low_branch.process(self.lp_mid.process(x))
        high = self.ap_high_branch.process(self.hp_mid.process(x))
        b0 = self.lp_lo.process(low)
        b1 = self.hp_lo.process(low)
        b2 = self.lp_hi.process(high)
        b3 = self.hp_hi.process(high)
        return [b0, b1, b2, b3]

    def reset(self):
        for f in (self.lp_mid, self.hp_mid, self.ap_low_branch, self.ap_high_branch,
                  self.lp_lo, self.hp_lo, self.lp_hi, self.hp_hi):
            f.reset()


# --------------------------------------------------------------------------- #
#  Phase rotator (Optimod style)
# --------------------------------------------------------------------------- #

class DelayLine:
    """Plain integer sample delay (per-channel), for phase-aligning
    parallel FIR correction paths."""

    def __init__(self, n: int, channels: int = 2):
        self.n = int(n)
        self.buf = np.zeros((self.n, channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return x
        w = np.concatenate([self.buf, x], axis=0)
        self.buf = w[-self.n:].copy()
        return w[:len(x)]

    def reset(self):
        self.buf[:] = 0.0


class PhaseRotator:
    """4 cascaded 1st-order all-passes @ 200 Hz.

    Symmetrises asymmetric program waveforms (voice!) before clipping so
    positive and negative peaks limit equally - a hallmark of Orban-style
    FM processing.
    """

    def __init__(self, fs: float, channels: int = 2, f0: float = 200.0, stages: int = 4):
        # staggered first-order sections (8100 style): spreading the
        # phase knee across 120-210 Hz symmetrises voice with far less
        # audible LF smear than stacking all stages on one frequency
        freqs = [120.0, 150.0, 180.0, 210.0][:stages] or [f0]
        sos = np.vstack([allpass1(fs, fq) for fq in freqs])
        self.filt = SOSFilter(sos, channels)

    def process(self, x): return self.filt.process(x)
    def reset(self): self.filt.reset()


# --------------------------------------------------------------------------- #
#  Pre-emphasis / de-emphasis @ 192 kHz  (bilinear transform of analog network)
# --------------------------------------------------------------------------- #

def _preemph_ba(fs: float, tau_us: float, f_pole: float = 20000.0):
    """H(s) = (1 + s*tau) / (1 + s/(2*pi*f_pole)), bilinear @ fs.

    tau places the boost corner (2122 Hz for 75 us / 3183 Hz for 50 us);
    the 20 kHz pole is the analog network's stop - without it the boost
    rises without limit and the top octave drives the HF stages several
    dB hotter than any real emphasis network ever would (the classic
    'harsh digital pre-emphasis' mistake; broadcasting 101).
    """
    tau = tau_us * 1e-6
    tau2 = 1.0 / (2.0 * np.pi * f_pole)
    k = 2.0 * fs
    b0 = 1.0 + k * tau
    b1 = 1.0 - k * tau
    a0 = 1.0 + k * tau2
    a1 = 1.0 - k * tau2
    return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])


class PreEmphasis:
    """Switchable 75 us / 50 us / flat first-order emphasis."""

    def __init__(self, fs: float, channels: int = 2, tau_us: float = 75.0):
        self.fs = fs
        self.channels = channels
        self.tau_us = None
        self.b = np.array([1.0, 0.0])
        self.a = np.array([1.0, 0.0])
        self.zi = np.zeros((1, channels))
        self.set_tau(tau_us)

    def set_tau(self, tau_us: float):
        if tau_us == self.tau_us:
            return
        self.tau_us = tau_us
        if tau_us and tau_us > 0:
            self.b, self.a = _preemph_ba(self.fs, tau_us)
        else:
            self.b = np.array([1.0, 0.0]); self.a = np.array([1.0, 0.0])
        self.zi = np.zeros((1, self.channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.tau_us in (None, 0):
            return x
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y


class DeEmphasis(PreEmphasis):
    """Inverse network for flat stereo monitoring after the clipper."""

    def set_tau(self, tau_us: float):
        if tau_us == self.tau_us:
            return
        self.tau_us = tau_us
        if tau_us and tau_us > 0:
            b, a = _preemph_ba(self.fs, tau_us)
            self.b, self.a = a, b          # swap => inverse
        else:
            self.b = np.array([1.0, 0.0]); self.a = np.array([1.0, 0.0])
        self.zi = np.zeros((1, self.channels))


# --------------------------------------------------------------------------- #
#  FIR designs used by the 192 kHz back end
# --------------------------------------------------------------------------- #

def design_lpf15k(fs: float) -> np.ndarray:
    """15 kHz brick-wall protecting the 19 kHz pilot region.

    Kaiser design: pass 15.0k, stop 18.6k, >=80 dB.  Linear phase so
    clipped material stays symmetric into the stereo matrix.
    """
    width = (18600.0 - 15000.0) / (fs / 2.0)
    ntaps, beta = kaiserord(80.0, width)
    if ntaps % 2 == 0:
        ntaps += 1
    return firwin(ntaps, 16600.0, window=("kaiser", beta), fs=fs)


def design_mpx_mask(fs: float, ntaps: int = 511) -> np.ndarray:
    """Composite-clipper product mask: LP @ 53 kHz + notch @ 19 kHz.

    Applied ONLY to clip products in "filtered" composite mode so
    splatter never lands on the pilot or above the 53 kHz top edge.
    firwin2 multiband, linear phase.
    """
    nyq = fs / 2.0
    freqs = [0.0, 17800.0, 18600.0, 19400.0, 20200.0, 53000.0, 56000.0, nyq]
    gains = [1.0, 1.0,     0.0,     0.0,     1.0,     1.0,     0.0,     0.0]
    return firwin2(ntaps, [f / nyq for f in freqs], gains)


def design_lpf53k(fs: float, ntaps: int = 191) -> np.ndarray:
    """Plain 53 kHz composite low-pass (raw composite-clip mode)."""
    return firwin(ntaps, 53500.0, window=("kaiser", 9.0), fs=fs)
