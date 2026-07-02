"""Thread-safe meter exchange between the audio callback and the UI."""

from __future__ import annotations

import threading
import numpy as np


class MeterStore:
    KEYS = ("input_l", "input_r", "agc_gain", "lev_gain", "comp_gr",
            "hf_gr", "wb_gr", "out_l", "out_r", "mpx_pct", "underruns",
            "running", "latency_ms")

    def __init__(self):
        self._lock = threading.Lock()
        self._d = {
            "input_l": -80.0, "input_r": -80.0,
            "agc_gain": 0.0,
            "lev_gain": [0.0, 0.0, 0.0, 0.0],
            "comp_gr": [0.0, 0.0, 0.0, 0.0],
            "hf_gr": 0.0, "wb_gr": 0.0,
            "out_l": -80.0, "out_r": -80.0,
            "mpx_pct": 0.0,
            "underruns": 0, "running": False, "latency_ms": 0.0,
            "devices": "",
        }

    @staticmethod
    def _db(x: float) -> float:
        return 20.0 * np.log10(max(float(x), 1e-9))

    def push(self, **kv):
        with self._lock:
            self._d.update(kv)

    def read(self) -> dict:
        with self._lock:
            return dict(self._d)
