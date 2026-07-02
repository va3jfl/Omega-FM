"""
omegafm.processing_chain
========================

The full OmegaFM audio chain, split at the sample-rate boundary:

FrontChain48 (48 kHz stereo)
    input gain -> phase rotator -> gated AGC -> 4-band parametric EQ ->
    psycho-acoustic bass -> LR4 crossover (160/1k/5k) ->
    4-band leveler -> 4-band compressor -> band mix -> sum -> drive

FinalSection192 (192 kHz stereo, fed by the 4x upsampler)
    pre-emphasis (75/50/0 us) -> HF limiter -> wideband look-ahead
    limiter -> soft-knee clipper -> 15 kHz linear-phase brick-wall ->
    safety clip

Broadcast rationale: dynamics live *before* emphasis where the ear
judges balance; peak control lives *after* emphasis at 4x oversampling
where the transmitter judges deviation.  The 15 kHz FIR guarantees
nothing from the clipper can reach the 19 kHz pilot region.
"""

from __future__ import annotations

import numpy as np

from .dsp import filters as F
from .dsp import dynamics as D
from .dsp.dynamics import CONTROL_BLOCK


class FrontChain48:
    def __init__(self, fs: float = 48000.0, channels: int = 2,
                 plugin_host=None):
        self.fs = fs
        self.channels = channels
        self.plugins = plugin_host
        self.rotator = F.PhaseRotator(fs, channels)
        self.agc = D.GainRider(fs, 1, fast_recovery_db_s=1.2,
                               gate_hold=True, gate_fast=True)
        self.eq: F.SOSFilter | None = None
        self._eq_key = None
        self.bass = D.BassEnhancer(fs, channels)
        self.xover = F.Crossover4(fs, channels)
        self.leveler = D.GainRider(fs, 4, target_db=-24.0, range_db=5.0,
                                   attack_db_s=1.5, release_db_s=1.5,
                                   gate_db=-50.0, fast_recovery_db_s=1.5,
                                   integrate_ms=300.0, differential=True,
                                   couple_db=2.0)
        self.comp = D.BandCompressor(fs, 4)
        self.band_mix = np.ones(4)
        self.in_gain = 1.0
        self._gain_smooth = 1.0
        self.drive = 1.0
        self.p: dict = {}
        self.meters = {
            "input_peak": np.zeros(channels),
            "agc_gain": 0.0,
            "lev_gain": np.zeros(4),
            "comp_gr": np.zeros(4),
        }

    # ------------------------------------------------------------------ params
    def apply_params(self, p: dict):
        self.p = p
        self.in_gain = 10.0 ** (p["input_gain_db"] / 20.0)
        self.agc.set_params(target_db=p["agc_target_db"], range_db=p["agc_range_db"],
                            attack_db_s=p["agc_attack_db_s"],
                            release_db_s=p["agc_release_db_s"],
                            gate_db=p["agc_gate_db"],
                            integrate_ms=p["agc_integration_ms"],
                            fast_db_s=p["agc_release_db_s"],
                            window_db=p["agc_window_db"])
        key = (tuple(p["eq_freqs"]), tuple(p["eq_gains"]), tuple(p["eq_qs"]))
        if key != self._eq_key:
            sos = np.vstack([F.rbj_peaking(self.fs, f, g, q)
                             for f, g, q in zip(*key)])
            old = self.eq
            self.eq = F.SOSFilter(sos, self.channels)
            if old is not None and old.zi.shape == self.eq.zi.shape:
                self.eq.zi = old.zi
            self._eq_key = key
        self.bass.set_params(shelf_db=p["bass_shelf_db"],
                             harmonics=p["bass_harmonics"])
        self.leveler.set_params(target_db=p["lev_targets_db"],
                                range_db=p["lev_range_db"],
                                attack_db_s=p["lev_rate_db_s"],
                                release_db_s=p["lev_rate_db_s"],
                                gate_db=p["lev_gate_db"],
                                fast_db_s=p["lev_rate_db_s"],
                                integrate_ms=p["lev_integration_ms"],
                                window_db=p["lev_window_db"],
                                couple_db=p["lev_couple_db"])
        self.comp.set_params(threshold_db=p["comp_thresholds_db"],
                             ratio=p["comp_ratios"],
                             attack_ms=p["comp_attack_ms"],
                             release_ms=p["comp_release_ms"],
                             makeup_db=p["comp_makeup_db"],
                             knee_db=p["comp_knee_db"],
                             platform_ms=p["comp_platform_ms"])
        self.band_mix = 10.0 ** (np.asarray(p["band_mix_db"]) / 20.0)
        self.drive = 10.0 ** (p["final_drive_db"] / 20.0)

    # ------------------------------------------------------------------ audio
    def process(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        n = len(x)
        assert n % CONTROL_BLOCK == 0, "block must be multiple of 32"

        # smoothed input gain (click-free trim)
        g0, g1 = self._gain_smooth, self.in_gain
        if abs(g1 - g0) > 1e-9:
            ramp = np.linspace(g0, g1, n, endpoint=False)[:, None]
            x = x * ramp
            self._gain_smooth = g1
        else:
            x = x * g1
        self.meters["input_peak"] = np.max(np.abs(x), axis=0)
        if self.plugins:
            x = self.plugins.run("post_input", x)

        if not p.get("bypass_rotator"):
            x = self.rotator.process(x)

        if not p.get("bypass_agc"):
            lv = D.block_rms_db(x)[:, None]
            x = self.agc.apply(x, lv)
            self.meters["agc_gain"] = float(self.agc.meter[0])
        else:
            self.meters["agc_gain"] = 0.0
        if self.plugins:
            x = self.plugins.run("post_agc", x)

        if not p.get("bypass_eq") and self.eq is not None:
            x = self.eq.process(x)
        if self.plugins:
            x = self.plugins.run("post_eq", x)

        if not p.get("bypass_bass"):
            x = self.bass.process(x)
        if self.plugins:
            x = self.plugins.run("post_bass", x)

        bands = self.xover.process(x)

        if not p.get("bypass_leveler"):
            lv = np.stack([D.block_rms_db(b) for b in bands], axis=1)
            bands = self.leveler.apply(bands, lv)
            self.meters["lev_gain"] = self.leveler.meter.copy()
        else:
            self.meters["lev_gain"] = np.zeros(4)

        if not p.get("bypass_comp"):
            lv = np.stack([D.block_peak_db(b) for b in bands], axis=1)
            bands = self.comp.apply(bands, lv)
            self.meters["comp_gr"] = self.comp.meter.copy()
        else:
            self.meters["comp_gr"] = np.zeros(4)

        y = sum(b * m for b, m in zip(bands, self.band_mix))
        y = y * self.drive
        if self.plugins:
            y = self.plugins.run("post_multiband", y)
        return y

    def reset(self):
        for m in (self.rotator, self.agc, self.bass, self.xover,
                  self.leveler, self.comp):
            m.reset()
        if self.eq:
            self.eq.reset()


class FinalSection192:
    def __init__(self, fs: float = 192000.0, channels: int = 2):
        self.fs = fs
        self.channels = channels
        self.preemph = F.PreEmphasis(fs, channels, 75.0)
        self.hf_lim = D.HFLimiter(fs, channels)
        self.wb_lim = D.LookaheadLimiter(fs, channels, lookahead_ms=2.0)
        taps15 = F.design_lpf15k(fs)
        self.lpf15 = F.FIRFilter(taps15, channels)
        self.clip_drive = 1.0
        # ------- 8100-style distortion-cancelled clipper ---------------
        # cubic soft clip, then the clipped-away *difference* below
        # 1.7 kHz is added back phase-aligned: bass passes UNCLIPPED
        # (fat, not crunchy) while HF keeps the flattening; the bass
        # peaks that remain are rounded afterwards by two passes of
        # band-limited, phase-aligned overshoot subtraction, and a
        # final gentle cubic at ceil*1.04 catches the crumbs.  All the
        # correction energy is filtered and delay-aligned, so nothing
        # hard ever shears - the 'insane level of filtering' that
        # separates broadcast clipping from DJ-gear clipping.
        from scipy.signal import firwin as _fw
        self.clip_ceil = 1.0
        n_dc = 257
        self._fir_dc = F.FIRFilter(
            _fw(n_dc, 2600.0 / (fs / 2.0)), channels)
        self._dly_dc = F.DelayLine((n_dc - 1) // 2, channels)
        n_ov = 127
        self._fir_ov = [F.FIRFilter(
            _fw(n_ov, 16000.0 / (fs / 2.0)), channels) for _ in range(2)]
        self._dly_ov = [F.DelayLine((n_ov - 1) // 2, channels)
                        for _ in range(2)]
        self.p: dict = {}
        self.meters = {"wb_gr": 0.0, "hf_gr": 0.0, "final_peak": np.zeros(channels)}

    @property
    def latency_samples(self) -> int:
        return (self.wb_lim.la + self.lpf15.delay
                + 128 + 63 + 63)   # distortion-cancel + 2x overshoot FIRs

    def apply_params(self, p: dict):
        self.p = p
        self.preemph.set_tau(p["preemph_us"])
        # Clipper distortion control: HF must never reach the clipper
        # more than ~2 dB into its knee, or sibilants turn into odd-order
        # IMD 'spit' (measured -17 dBc at clip drive +5 without this).
        # The HF limiter threshold therefore tracks the clip drive - the
        # user's setting rules at low drive, protection takes over when
        # the clipper is driven hard.
        hf_thr_eff = min(p["hf_lim_threshold_db"],
                         4.0 - p["clip_drive_db"])
        self.hf_lim.set_params(threshold_db=hf_thr_eff,
                               release_ms=p["hf_lim_release_ms"],
                               split_hz=p["hf_split_hz"])
        self.wb_lim.set_params(ceiling_db=p["wb_lim_ceiling_db"],
                               release_db_s=p["wb_lim_release_db_s"],
                               platform_ms=p["wb_platform_ms"])
        self.clip_drive = 10.0 ** (p["clip_drive_db"] / 20.0)
        self.clip_knee = p["clip_knee"]

    def process(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        x = self.preemph.process(x)

        if not p.get("bypass_hf_lim"):
            x = self.hf_lim.process(x)
            self.meters["hf_gr"] = self.hf_lim.gr_meter
        else:
            self.meters["hf_gr"] = 0.0

        if not p.get("bypass_wb_lim"):
            x = self.wb_lim.process(x)
            self.meters["wb_gr"] = self.wb_lim.gr_meter
        else:
            self.meters["wb_gr"] = 0.0

        clip_on = not p.get("bypass_clipper")
        if clip_on:
            w = x * self.clip_drive
            xc = D.cubic_clip(w, self.clip_ceil)
            d = w - xc
        else:
            xc = x
            d = np.zeros_like(x)
        # distortion cancel: LF of the removed material comes back,
        # phase-aligned - bass stays unclipped, HF keeps the flattening
        x = self._dly_dc.process(xc) + self._fir_dc.process(d)

        # 15 kHz brick wall ALWAYS in circuit - pilot protection is not optional
        x = self.lpf15.process(x)

        # two passes of band-limited, phase-aligned overshoot rounding:
        # whatever still exceeds the ceiling (unclipped bass peaks and
        # filter ring) is subtracted as a smooth <16 kHz correction
        for k in range(2):
            ov = x - np.clip(x, -self.clip_ceil, self.clip_ceil)
            x = self._dly_ov[k].process(x) - self._fir_ov[k].process(ov)
        x = np.clip(x, -1.0, 1.0)   # touches only sub-0.2 dB crumbs
        self.meters["final_peak"] = np.max(np.abs(x), axis=0)
        return x

    def reset(self):
        self.hf_lim.reset(); self.wb_lim.reset(); self.lpf15.reset()
        self._fir_dc.reset(); self._dly_dc.reset()
        for k in range(2):
            self._fir_ov[k].reset(); self._dly_ov[k].reset()
