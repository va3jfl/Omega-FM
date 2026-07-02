"""
omegafm.presets
===============

JSON preset files: the *sound* of the processor, portable between
installs and shareable between stations.

Format
------
{
  "format":    "omegafm-preset",
  "version":   1,
  "name":      "Club Night",
  "signature": "CUSTOM",
  "density":   "HEAVY",
  "created":   "2026-07-02T00:41:00",
  "params":    { ...processing parameters... }
}

What is (not) in a preset
-------------------------
A preset is the processing chain only.  Station identity and local
setup are deliberately excluded so importing someone else's sound never
clobbers your RDS text, pre-emphasis region, gain staging to your gear,
or pilot/RDS injection: rds_*, preemph_us, monitor_deemph,
input_gain_db, output_level, pilot_pct, rds_pct, mpx_mode.

Validation on load
------------------
Unknown keys are ignored; missing keys fall back to defaults; every
value is type-coerced against DEFAULTS (bool/number/string/number-list
with length fixed); numeric values are clamped to safe DSP ranges so a
hand-edited file can never blow up a filter design.
"""

from __future__ import annotations

import json
import numbers
from datetime import datetime
from pathlib import Path

from .params import DEFAULTS, SIGNATURE_ORDER, DENSITY_ORDER

FORMAT_TAG = "omegafm-preset"
FORMAT_VERSION = 1

# station / local-setup keys - never part of a sound preset
EXCLUDE = frozenset({
    "rds_enable", "rds_pi", "rds_pty", "rds_ps", "rds_rt",
    "rds_dynamic_ps", "rds_file", "rds_watch_file", "rds_tp",
    "preemph_us", "monitor_deemph",
    "input_gain_db", "output_level",
    "pilot_pct", "rds_pct",
    "bs412_enable", "bs412_target_dbr",
    "mpx_mode",
})

PRESET_KEYS = tuple(k for k in DEFAULTS if k not in EXCLUDE)

# safe DSP ranges (min, max) for numeric keys - applied per element
CLAMPS = {
    "agc_target_db": (-40, 0), "agc_range_db": (0, 20),
    "agc_attack_db_s": (0.1, 20), "agc_release_db_s": (0.05, 10),
    "agc_gate_db": (-90, -20), "agc_integration_ms": (20, 3000),
    "agc_window_db": (0, 8),
    "eq_gains": (-12, 12), "eq_freqs": (20, 16000), "eq_qs": (0.2, 8),
    "bass_shelf_db": (0, 12), "bass_harmonics": (0, 1),
    "lev_targets_db": (-60, -6), "lev_range_db": (0, 12),
    "lev_rate_db_s": (0.1, 10), "lev_gate_db": (-90, -25),
    "lev_integration_ms": (20, 3000), "lev_window_db": (0, 8),
    "lev_couple_db": (0, 7),
    "comp_thresholds_db": (-60, 0), "comp_ratios": (1.05, 20),
    "comp_attack_ms": (0.5, 200), "comp_release_ms": (20, 3000),
    "comp_makeup_db": (0, 15), "comp_knee_db": (0, 24),
    "comp_platform_ms": (50, 5000),
    "final_drive_db": (-6, 15),
    "hf_lim_threshold_db": (-12, 0), "hf_lim_release_ms": (5, 1000),
    "hf_split_hz": (1500, 10000),
    "wb_lim_ceiling_db": (-6, 0), "wb_lim_release_db_s": (10, 600),
    "wb_platform_ms": (50, 5000),
    "clip_drive_db": (0, 8), "clip_knee": (0, 0.6),
    "composite_drive_db": (0, 6),
}

CHOICES = {
    "composite_clip_mode": ("filtered", "raw", "off"),
}


def _clamp(key, v):
    lo, hi = CLAMPS.get(key, (None, None))
    if lo is None:
        return float(v)
    return float(min(max(float(v), lo), hi))


def _coerce(key: str, value):
    """Coerce `value` to the shape/type of DEFAULTS[key]; None if hopeless."""
    ref = DEFAULTS[key]
    if isinstance(ref, bool):
        return bool(value)
    if isinstance(ref, str):
        v = str(value)
        allowed = CHOICES.get(key)
        return v if (allowed is None or v in allowed) else None
    if isinstance(ref, list):
        if not isinstance(value, (list, tuple)):
            return None
        vals = [ _clamp(key, x) for x in value
                 if isinstance(x, numbers.Number) ]
        if not vals:
            return None
        # fix the length against the reference
        while len(vals) < len(ref):
            vals.append(vals[-1])
        return vals[:len(ref)]
    if isinstance(ref, numbers.Number):
        if not isinstance(value, numbers.Number):
            return None
        return _clamp(key, value)
    return None


def export_file(path, params: dict, signature: str, density: str,
                name: str | None = None, plugins: dict | None = None):
    """Write the current *sound* to a preset JSON file - including the
    plugin rack (enabled plugins + their knob values)."""
    path = Path(path)
    doc = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "name": name or path.stem,
        "signature": signature,
        "density": density,
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": {k: params[k] for k in PRESET_KEYS if k in params},
    }
    if plugins is not None:
        doc["plugins"] = {
            str(pid): {"enabled": bool(st.get("enabled")),
                       "params": {str(k): float(v)
                                  for k, v in (st.get("params") or {}).items()
                                  if isinstance(v, (int, float))}}
            for pid, st in plugins.items() if isinstance(st, dict)
        }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def load_file(path):
    """Read + validate a preset file.

    Returns (name, density, params) where params contains only known,
    coerced, range-clamped processing keys.  Raises ValueError with a
    human-readable reason on anything unusable.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"not readable as JSON: {e}") from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT_TAG:
        raise ValueError("not an OmegaFM preset (missing format tag)")
    if int(doc.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(f"preset version {doc.get('version')} is newer "
                         f"than this build understands ({FORMAT_VERSION})")
    raw = doc.get("params")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("preset contains no parameters")

    params = {}
    for k, v in raw.items():
        if k not in DEFAULTS or k in EXCLUDE:
            continue                       # unknown/foreign or station key
        cv = _coerce(k, v)
        if cv is not None:
            params[k] = cv
    if not params:
        raise ValueError("no usable parameters in preset")

    name = str(doc.get("name") or path.stem)[:24]
    density = doc.get("density")
    if density not in DENSITY_ORDER:
        density = None
    _ = SIGNATURE_ORDER  # (label only, kept for file readability)

    # plugin rack: None = file predates the feature (leave rack alone);
    # a dict (possibly empty) = apply exclusively on import
    plugins = None
    if "plugins" in doc:
        plugins = {}
        raw_pl = doc.get("plugins")
        if isinstance(raw_pl, dict):
            for pid, st in raw_pl.items():
                if not isinstance(st, dict):
                    continue
                plugins[str(pid)] = {
                    "enabled": bool(st.get("enabled")),
                    "params": {str(k): float(v)
                               for k, v in (st.get("params") or {}).items()
                               if isinstance(v, (int, float))},
                }
    return name, density, params, plugins
