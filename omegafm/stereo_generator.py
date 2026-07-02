"""
omegafm.stereo_generator
========================

All-digital FM stereo / MPX generator @ 192 kHz.

One phase accumulator theta drives everything:
    pilot   = sin(theta)          (19 kHz)
    38 kHz  = sin(2*theta)        (DSB-SC subcarrier, phase-locked)
    57 kHz  = sin(3*theta)        (RDS carrier)
so the +/-1 Hz pilot tolerance and the pilot/subcarrier phase relation
of BS.450 hold *by construction*.

Composite clipping
------------------
"filtered" (the OmegaFM default):
    a  = M + S*sin(2*theta)            (audio MPX, no pilot yet)
    c  = kneeclip(a * drive)
    products p = c - a
    p' = MASK_FIR(p)                   (linear-phase LP 53k + notch 19k)
    out = delay(a, mask_delay) + p'    + pilot + RDS  <- generated with a
                                                          *delayed* phase
The pilot & RDS are injected after the mask using theta delayed by the
mask group delay, so pilot and subcarrier stay phase-coherent and clip
splatter can never land on 19 kHz or above 53 kHz.

"raw": clip the entire composite (pilot included) then a plain
linear-phase 53 kHz LP.  Louder, dirtier - it is on the panel because
the original had it.
"""

from __future__ import annotations

import numpy as np

from .dsp import filters as F
from .dsp import dynamics as D
from .rds import RDSEncoder

PILOT_HZ = 19000.0


class StereoGenerator:
    def __init__(self, fs: float = 192000.0):
        self.fs = fs
        self.dtheta = 2.0 * np.pi * PILOT_HZ / fs
        self.theta = 0.0

        mask = F.design_mpx_mask(fs)
        self.mask_fir = F.FIRFilter(mask, 1)
        self.mask_delay = self.mask_fir.delay
        self.audio_delay = F.Delay(self.mask_delay, 1)

        lp53 = F.design_lpf53k(fs)
        self.raw_lpf = F.FIRFilter(lp53, 1)

        self.rds = RDSEncoder(fs)

        # params
        self.pilot_inj = 0.09
        self.rds_inj = 0.04
        self.rds_enable = True
        self.mode = "filtered"
        self.drive = 10.0 ** (0.6 / 20.0)
        self.output_gain = 1.0
        self.mpx_peak = 0.0
        from collections import deque
        self._bs_g = 0.0
        self._bs_sm = 1.0
        self._bs_pow = self.BS_REF
        self._bs_hist = deque()
        self._bs_hn = 0
        self._bs_hs = 0.0
        self.bs412_dbr = -99.0
        self.bs412_gr = 0.0
        self.bs412_enable = False
        self.bs412_target = 0.0          # modulation fraction (1.0 = 100 %)

    # ------------------------------------------------------------------ params
    def apply_params(self, p: dict):
        self.pilot_inj = np.clip(p["pilot_pct"], 0.0, 20.0) / 100.0
        self.rds_inj = np.clip(p["rds_pct"], 0.0, 8.0) / 100.0
        self.rds_enable = bool(p["rds_enable"])
        self.mode = p["composite_clip_mode"]
        self.drive = 10.0 ** (p["composite_drive_db"] / 20.0)
        self.bs412_enable = bool(p.get("bs412_enable", False))
        self.bs412_target = float(p.get("bs412_target_dbr", 0.0))
        self.output_gain = float(np.clip(p["output_level"], 0.0, 1.0))
        self.rds.set_text(ps=p["rds_ps"], rt=p["rds_rt"], pi=p["rds_pi"],
                          pty=p["rds_pty"], tp=p["rds_tp"])

    # ------------------------------------------------------------------ audio
    # ---- ITU-R BS.412 MPX power limiter --------------------------------
    # The legal reference (0 dBr) is the power of a sine producing
    # +/-19 kHz deviation: (19/75)^2 / 2 in our units where 1.0 = 75 kHz.
    # The controller rides ONLY the audio part of the composite - pilot
    # and RDS injection are constitutionally untouched - on a ~6 s
    # power integrator with a 0.2 dB safety margin, and the meter
    # reports the honest sliding 60 s figure the regulator measures.
    BS_REF = (19.0 / 75.0) ** 2 / 2.0

    def process(self, lr: np.ndarray) -> np.ndarray:
        """lr: (N,2) @192k, |x|<=1 from the final section -> (N,) MPX."""
        n = len(lr)
        idx = np.arange(1, n + 1)
        theta = self.theta + self.dtheta * idx
        self.theta = float(theta[-1] % (2.0 * np.pi))

        m = 0.5 * (lr[:, 0] + lr[:, 1])
        s = 0.5 * (lr[:, 0] - lr[:, 1])

        rds_inj = self.rds_inj if self.rds_enable else 0.0
        audio_scale = max(0.1, 1.0 - self.pilot_inj - rds_inj)

        sub = np.sin(2.0 * theta)
        a = (m + s * sub) * audio_scale
        g_tgt = 10.0 ** (-self._bs_g / 20.0) if self.bs412_enable else 1.0
        self._bs_sm = 0.7 * self._bs_sm + 0.3 * g_tgt
        if abs(self._bs_sm - 1.0) > 1e-6:
            a = a * self._bs_sm

        if self.mode == "filtered":
            c = D.knee_clip(a * self.drive, audio_scale * 0.97, 0.1)
            p = c - a
            p = self.mask_fir.process(p[:, None])[:, 0]
            a_d = self.audio_delay.process(a[:, None])[:, 0]
            comp = a_d + p
            theta_c = theta - self.mask_delay * self.dtheta
        elif self.mode == "raw":
            theta_c = theta
            comp = a
        else:  # off
            theta_c = theta
            comp = a

        pilot = self.pilot_inj * np.sin(theta_c)
        comp = comp + pilot
        if rds_inj > 0.0:
            bb = self.rds.render_baseband(n)
            comp = comp + rds_inj * bb * np.sin(3.0 * theta_c)

        if self.mode == "raw":
            comp = D.knee_clip(comp * self.drive, 1.0, 0.05)
            comp = self.raw_lpf.process(comp[:, None])[:, 0]

        self.mpx_peak = float(np.max(np.abs(comp)))

        # ---- BS.412 measurement + controller (post composite clip) ----
        ss = float(np.dot(comp, comp))
        alpha = float(np.exp(-n / (6.0 * self.fs)))
        self._bs_pow = alpha * self._bs_pow + (1 - alpha) * (ss / n)
        self._bs_hist.append((n, ss))
        self._bs_hn += n
        self._bs_hs += ss
        max_n = int(60.0 * self.fs)
        while self._bs_hn - self._bs_hist[0][0] >= max_n:
            n0, s0 = self._bs_hist.popleft()
            self._bs_hn -= n0
            self._bs_hs -= s0
        dbr_fast = 10.0 * np.log10(self._bs_pow / self.BS_REF + 1e-15)
        self.bs412_dbr = 10.0 * np.log10(
            self._bs_hs / max(self._bs_hn, 1) / self.BS_REF + 1e-15)
        if self.bs412_enable:
            dt = n / self.fs
            err = dbr_fast - (self.bs412_target - 0.2)
            if err > 0:
                self._bs_g = min(12.0, self._bs_g
                                 + min(1.5, 0.8 * err) * dt)
            elif err < -0.4:                 # hysteresis: no hunting
                self._bs_g = max(0.0, self._bs_g - 0.15 * dt)
        else:
            self._bs_g = 0.0
        self.bs412_gr = self._bs_g
        return comp * self.output_gain

    def reset(self):
        self.theta = 0.0
        self._bs_g = 0.0
        self._bs_sm = 1.0
        self._bs_pow = self.BS_REF
        self._bs_hist.clear()
        self._bs_hn = 0
        self._bs_hs = 0.0
        self.bs412_dbr = -99.0
        self.bs412_gr = 0.0
        self.mask_fir.reset(); self.audio_delay.reset(); self.raw_lpf.reset()
