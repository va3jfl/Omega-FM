"""
omegafm.rds
===========

Real CENELEC EN 50067 RDS encoder.

*  Group scheduler interleaves 0A (PS, 8 chars, DI stereo) and 2A
   (RadioText 64 chars, A/B flip on change) - PS refreshes ~0.7 s,
   full RT ~2.8 s.
*  Each 26-bit block = 16 data bits + CRC-10 (g(x)=0x5B9) XOR offset
   word (A/B/C/D).
*  Differential encoding, then biphase (Manchester) symbols shaped with
   a raised-cosine (beta = 1) pulse per spec, placed by overlap-add at
   *fractional* sample positions - the 1187.5 baud clock is derived
   from the same phase reference as the 57 kHz carrier (57000/48), so
   carrier and data stay locked exactly as BS.450 demands.

The encoder produces *baseband* only; the stereo generator multiplies
it by sin(3*theta) so pilot, 38 kHz subcarrier and 57 kHz RDS carrier
all come from one accumulator.
"""

from __future__ import annotations

import threading
import numpy as np

CRC_POLY = 0x5B9
OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}
BIT_RATE = 1187.5


def crc10(data16: int) -> int:
    reg = data16 << 10
    for i in range(25, 9, -1):
        if reg & (1 << i):
            reg ^= CRC_POLY << (i - 10)
    return reg & 0x3FF


def make_block(data16: int, offset: str) -> list[int]:
    check = crc10(data16) ^ OFFSETS[offset]
    word = (data16 << 10) | check
    return [(word >> (25 - i)) & 1 for i in range(26)]


class RDSEncoder:
    def __init__(self, fs: float = 192000.0):
        self.fs = fs
        self._lock = threading.Lock()
        self.pi = 0x1234
        self.pty = 10
        self.tp = False
        self.ps = "OMEGAFM "
        self.rt = "OmegaFM"
        self.ab_flag = 0
        self._dirty = True

        self._groups: list[list[int]] = []
        self._gindex = 0
        self._bitbuf: list[int] = []
        self._last_diff = 0

        # ---- biphase pulse table (RC beta=1 over +/-2 Ts, +pulse at 0, -pulse at Ts/2)
        Ts = fs / BIT_RATE                       # samples per symbol (~161.68)
        half = int(np.ceil(2.0 * Ts))
        t = (np.arange(-half, half + 1)) / Ts    # in symbol periods
        self._pulse = self._rc(t) - self._rc(t - 0.5)
        # normalise so the overlap-added waveform stays within +/-1
        rng = np.random.default_rng(3)
        probe = np.zeros(int(Ts * 220) + len(self._pulse))
        pos = 0.0
        for _ in range(200):
            i0 = int(pos)
            amp = 1.0 if rng.integers(2) else -1.0
            probe[i0:i0 + len(self._pulse)] += amp * self._pulse
            pos += Ts
        self._pulse /= max(1.0, np.max(np.abs(probe)))
        self._pulse_center = half
        self._sym_period = Ts
        self._next_bit_pos = 0.0                 # fractional sample of next symbol
        self._tail = np.zeros(len(self._pulse) + 4)

    @staticmethod
    def _rc(t: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Raised-cosine impulse response h(t), t in symbol periods."""
        eps = 1e-9
        sinc = np.sinc(t)
        denom = 1.0 - (2.0 * beta * t) ** 2
        cosf = np.cos(np.pi * beta * t)
        out = np.where(np.abs(denom) < 1e-6,
                       (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta) + eps),
                       sinc * cosf / np.where(np.abs(denom) < 1e-6, 1.0, denom))
        return out

    # ------------------------------------------------------------------ text
    def set_text(self, ps: str | None = None, rt: str | None = None,
                 pi: int | None = None, pty: int | None = None,
                 tp: bool | None = None):
        with self._lock:
            if ps is not None:
                self.ps = (ps[:8]).ljust(8)
                self._dirty = True
            if rt is not None:
                rt = rt[:64]
                if rt != self.rt:
                    self.ab_flag ^= 1
                self.rt = rt
                self._dirty = True
            if pi is not None:
                self.pi = pi & 0xFFFF
                self._dirty = True
            if pty is not None:
                self.pty = pty & 0x1F
                self._dirty = True
            if tp is not None:
                self.tp = tp
                self._dirty = True

    # ------------------------------------------------------------ group build
    def _build_groups(self):
        ps = self.ps
        rt = self.rt
        if len(rt) < 64:
            rt = rt + "\r"
        rt = rt.ljust(64)
        pty = self.pty
        tp = 1 if self.tp else 0
        groups = []
        rt_segs = 16
        for seg in range(rt_segs):
            ps_addr = seg % 4
            # ---- 0A: PI | flags+addr | AF | PS chars
            b2 = (0 << 12) | (0 << 11) | (tp << 10) | (pty << 5)
            b2 |= (0 << 4) | (0 << 3)             # TA=0, MS=music
            di_bit = 1 if ps_addr == 3 else 0     # DI d0=stereo at addr 3
            b2 |= (di_bit << 2) | ps_addr
            b3 = 0xE0CD                            # no AF list
            c1, c2 = ps[2 * ps_addr], ps[2 * ps_addr + 1]
            b4 = (ord(c1) << 8) | ord(c2)
            groups.append(make_block(self.pi, "A") + make_block(b2, "B")
                          + make_block(b3, "C") + make_block(b4, "D"))
            # ---- 2A: RT
            b2 = (2 << 12) | (0 << 11) | (tp << 10) | (pty << 5)
            b2 |= (self.ab_flag << 4) | seg
            r = rt[4 * seg: 4 * seg + 4]
            b3 = (ord(r[0]) << 8) | ord(r[1])
            b4 = (ord(r[2]) << 8) | ord(r[3])
            groups.append(make_block(self.pi, "A") + make_block(b2, "B")
                          + make_block(b3, "C") + make_block(b4, "D"))
        return groups

    def _next_bit(self) -> int:
        if not self._bitbuf:
            with self._lock:
                if self._dirty or not self._groups:
                    self._groups = self._build_groups()
                    self._dirty = False
            self._bitbuf = list(self._groups[self._gindex])
            self._gindex = (self._gindex + 1) % len(self._groups)
        bit = self._bitbuf.pop(0)
        diff = bit ^ self._last_diff               # differential encoding
        self._last_diff = diff
        return diff

    # ------------------------------------------------------------- baseband
    def render_baseband(self, n: int) -> np.ndarray:
        """n samples of shaped biphase baseband, |x| ~<= 1."""
        out = np.zeros(n + len(self._pulse) + 4)
        out[:len(self._tail)] += self._tail
        pos = self._next_bit_pos
        L = len(self._pulse)
        while pos < n:
            bit = self._next_bit()
            amp = 1.0 if bit else -1.0
            i0 = int(np.floor(pos))
            frac = pos - i0
            start = i0 - self._pulse_center
            # 2-tap linear split for fractional placement
            w1, w0 = frac, 1.0 - frac
            s = start
            if s < 0:
                # left edge clip (only possible at very first call)
                cut = -s
                out[0:L - cut] += amp * w0 * self._pulse[cut:]
                out[1:L - cut + 1] += amp * w1 * self._pulse[cut:]
            else:
                out[s:s + L] += amp * w0 * self._pulse
                out[s + 1:s + 1 + L] += amp * w1 * self._pulse
            pos += self._sym_period
        self._next_bit_pos = pos - n
        self._tail = out[n:].copy()
        return out[:n]
