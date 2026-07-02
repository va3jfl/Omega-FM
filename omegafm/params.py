"""
omegafm.params
==============

Flat parameter dictionary + thread-safe store.

resolve(signature, density, trims) merges:
    DEFAULTS  <-  SIGNATURES[sig]  <-  DENSITIES[den]  <-  user trims
into the single dict the processing chain snapshots every block.
"""

from __future__ import annotations

import threading
import copy

BANDS = ("LO", "LM", "HM", "HI")

DEFAULTS = {
    # I/O -----------------------------------------------------------------
    "input_gain_db": 0.0,
    "output_level": 0.9,            # 0..1 scaler on the final output
    "mpx_mode": False,              # False = stereo L/R out, True = MPX
    "monitor_deemph": True,         # de-emphasize stereo monitor output
    "preemph_us": 75.0,             # 75 / 50 / 0

    # Stage bypasses --------------------------------------------------------
    "bypass_rotator": False,
    "bypass_agc": False,
    "bypass_eq": False,
    "bypass_bass": False,
    "bypass_leveler": False,
    "bypass_comp": False,
    "bypass_hf_lim": False,
    "bypass_wb_lim": False,
    "bypass_clipper": False,
    "bypass_comp_clip": False,

    # AGC -------------------------------------------------------------------
    "agc_target_db": -17.0,
    "agc_range_db": 12.0,
    "agc_attack_db_s": 2.5,
    "agc_release_db_s": 1.2,
    "agc_gate_db": -45.0,
    "agc_integration_ms": 400.0,   # loudness ballistic (true RMS integration)
    "agc_window_db": 1.0,          # deadband: hold within +/-window of target

    # 4-band parametric EQ ----------------------------------------------------
    "eq_freqs": [100.0, 400.0, 2000.0, 8000.0],
    "eq_gains": [0.0, 0.0, 0.0, 0.0],
    "eq_qs":    [1.0, 1.0, 1.0, 1.0],

    # Bass enhancement ----------------------------------------------------------
    "bass_shelf_db": 1.0,
    "bass_harmonics": 0.3,

    # 4-band leveler ---------------------------------------------------------
    "lev_targets_db": [-24.0, -22.5, -28.0, -27.0],
    "lev_range_db": 7.0,
    "lev_rate_db_s": 1.5,
    "lev_gate_db": -50.0,
    "lev_integration_ms": 300.0,   # per-band loudness ballistic
    "lev_window_db": 2.0,          # deadband around each band target
    "lev_couple_db": 2.0,          # max band deviation from the pack mean

    # 4-band compressor -----------------------------------------------------
    "comp_thresholds_db": [-18.5, -19.0, -19.0, -16.0],
    "comp_ratios": [2.5, 3.0, 3.0, 3.5],
    "comp_attack_ms": [18.0, 10.0, 7.0, 4.0],
    "comp_release_ms": [280.0, 200.0, 150.0, 110.0],
    "comp_makeup_db": [1.5, 1.5, 1.5, 1.5],
    "comp_knee_db": 6.0,
    "comp_platform_ms": 700.0,     # program-dependent release glide
    "band_mix_db": [0.0, 0.0, 0.0, 0.0],

    # Final section drive / limiting ------------------------------------------
    # Staging philosophy (Omega): the leveler+compressor make the level,
    # the WB/HF limiters only tickle 1-3 dB, the clipper makes loudness.
    "final_drive_db": 0.0,
    "hf_lim_threshold_db": -1.4,
    "hf_lim_release_ms": 80.0,
    "hf_split_hz": 4500.0,
    "wb_lim_ceiling_db": -0.5,
    "wb_lim_release_db_s": 90.0,    # fast path (returns to the platform)
    "wb_platform_ms": 250.0,        # platform charge time (down); leak up 0.8 dB/s
    "clip_drive_db": 2.0,           # into the L/R soft clipper
    "clip_knee": 0.25,

    # Stereo generator / composite ------------------------------------------
    "pilot_pct": 9.0,               # 0..20 % injection
    "bs412_enable": False,          # ITU-R BS.412 MPX power limiter
    "bs412_target_dbr": 0.0,        # 60 s MPX power target, dBr
    "composite_clip_mode": "filtered",  # "filtered" | "raw" | "off"
    "composite_drive_db": 0.6,

    # RDS ---------------------------------------------------------------------
    "rds_enable": True,
    "rds_pct": 4.0,                 # injection %
    "rds_pi": 0x1234,
    "rds_pty": 10,                  # Pop Music (RDS)
    "rds_ps": "OMEGAFM",
    "rds_rt": "OmegaFM - broadcast processing in Python",
    "rds_tp": False,
    "rds_dynamic_ps": False,
    "rds_file": "",
    "rds_watch_file": False,
}

# Signature = tonal character  -------------------------------------------------
SIGNATURES = {
    "NATURAL": {},
    "SMOOTH": {
        "eq_gains": [0.0, 0.5, -0.5, 1.0],
        "bass_shelf_db": 1.0,
        "comp_ratios": [2.2, 2.4, 2.4, 2.6],
        "comp_release_ms": [340.0, 260.0, 200.0, 150.0],
        "comp_attack_ms": [22.0, 14.0, 9.0, 6.0],
        "clip_knee": 0.35,
    },
    "AGGRESSIVE": {
        "eq_gains": [0.5, -0.5, 1.0, 3.0],
        "bass_shelf_db": 0.5,
        "bass_harmonics": 0.45,
        "comp_ratios": [3.0, 3.4, 3.4, 3.8],
        "comp_attack_ms": [12.0, 7.0, 5.0, 3.0],
        "comp_release_ms": [220.0, 160.0, 120.0, 90.0],
        "clip_knee": 0.15,
    },
    "CUSTOM": {},   # populated from saved user state
}

# Density = how hard everything is driven ------------------------------------
DENSITIES = {
    "LIGHT": {
        "_thr_shift": 3.5,       # raise thresholds => less GR
        "_rel_scale": 1.3,
        "clip_drive_db": 5.0,
        "composite_drive_db": 0.4,
        "final_drive_db": -2.9,
        "comp_makeup_db": [1.0, 1.0, 1.0, 1.0],
    },
    "MEDIUM": {
        "_thr_shift": 0.0,
        "_rel_scale": 1.0,
        "clip_drive_db": 5.5,
        "composite_drive_db": 0.8,
    },
    "HEAVY": {
        "_thr_shift": -3.0,
        "_rel_scale": 0.75,
        "clip_drive_db": 6.0,
        "composite_drive_db": 1.2,
        "comp_makeup_db": [2.0, 2.0, 2.0, 2.0],
        "final_drive_db": -1.9,
    },
}

SIGNATURE_ORDER = ("AGGRESSIVE", "SMOOTH", "NATURAL", "CUSTOM")
DENSITY_ORDER = ("HEAVY", "MEDIUM", "LIGHT")


def neutralize_base(params: dict, density: str) -> dict:
    """Un-bake the given density's arithmetic modifiers from a resolved
    snapshot so it can serve as a density-neutral CUSTOM base.

    resolve() re-applies _thr_shift / _rel_scale on top of the base, so a
    base captured under HEAVY must have them removed or they would apply
    twice.  Round-trip: resolve(CUSTOM, D) of neutralize(resolve(x, D), D)
    reproduces the exact same parameters.
    """
    den = DENSITIES.get(density, DENSITIES["MEDIUM"])
    shift = den.get("_thr_shift", 0.0)
    rels = den.get("_rel_scale", 1.0)
    base = copy.deepcopy(params)
    base["comp_thresholds_db"] = [t - shift for t in base["comp_thresholds_db"]]
    base["comp_release_ms"] = [r / rels for r in base["comp_release_ms"]]
    return base


def resolve(signature: str, density: str, trims: dict | None = None,
            custom_base: dict | None = None) -> dict:
    p = copy.deepcopy(DEFAULTS)
    den = DENSITIES.get(density, DENSITIES["MEDIUM"])
    shift = den.get("_thr_shift", 0.0)
    rels = den.get("_rel_scale", 1.0)
    if signature == "CUSTOM" and custom_base:
        # density absolutes first, then the *neutral* custom base wins -
        # a saved/imported preset keeps its drives and makeup exactly,
        # while density stays live via the threshold/release modifiers.
        for k, v in den.items():
            if not k.startswith("_"):
                p[k] = copy.deepcopy(v)
        for k, v in custom_base.items():
            if k in p:
                p[k] = copy.deepcopy(v)
    else:
        sig = SIGNATURES.get(signature, {})
        for k, v in sig.items():
            p[k] = copy.deepcopy(v)
        for k, v in den.items():
            if k.startswith("_"):
                continue
            p[k] = copy.deepcopy(v)
    p["comp_thresholds_db"] = [t + shift for t in p["comp_thresholds_db"]]
    p["comp_release_ms"] = [r * rels for r in p["comp_release_ms"]]
    if trims:
        for k, v in trims.items():
            if k in p:
                p[k] = copy.deepcopy(v)
    return p


class ParamStore:
    """Thread-safe parameter snapshot exchanged UI -> audio thread."""

    def __init__(self, initial: dict | None = None):
        self._lock = threading.Lock()
        self._params = dict(initial or resolve("AGGRESSIVE", "LIGHT"))
        self._version = 0

    def update(self, params: dict):
        with self._lock:
            self._params = dict(params)
            self._version += 1

    def patch(self, **kv):
        with self._lock:
            self._params.update(kv)
            self._version += 1

    def snapshot(self):
        with self._lock:
            return self._version, dict(self._params)
