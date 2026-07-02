"""
omegafm.plugin_host
===================

Modular DSP plugins for the OmegaFM chain.

A plugin is ONE .py file dropped into the app's `plugins/` folder (or
`~/.omegafm/plugins/`).  It declares itself with a manifest dict and a
Plugin class:

    PLUGIN = {
        "id":      "stereo_widener",          # unique, stable
        "name":    "Stereo Widener",
        "version": "1.0",
        "insert":  "post_input",              # where in the chain
        "params": [                            # auto-built knobs
            {"key": "width", "label": "WIDTH", "min": 0.0, "max": 2.0,
             "default": 1.0, "fmt": "{:.2f}", "suffix": ""},
        ],
        "meters": [                            # auto-built LED bars
            {"key": "corr", "label": "CORR", "mode": "bipolar",
             "lo": -1.0, "hi": 1.0},
        ],
    }

    class Plugin:
        def __init__(self, fs, channels=2): ...
        def set_params(self, **kv): ...        # keys from manifest
        def process(self, x): return x         # float64 (N, 2) at fs
        def reset(self): ...
        # self.meters = {"corr": 0.97, ...}    # read by the UI timer

Insertion points (all 48 kHz stereo, full-band):

    post_input      after input gain, before the phase rotator
    post_agc        after the gated AGC
    post_eq         after the 4-band parametric EQ
    post_bass       after the psycho-acoustic bass, before the crossover
    post_multiband  after band mix + drive (end of the 48 kHz front)

Safety contract - the validated core chain can never be taken down by a
plugin: every process() call is wrapped; on an exception or a wrong
output shape the plugin is auto-bypassed, flagged with the traceback,
and audio continues through the untouched chain.
"""

from __future__ import annotations

import importlib.util
import threading
import traceback
from pathlib import Path

import numpy as np

INSERT_POINTS = ("post_input", "post_agc", "post_eq",
                 "post_bass", "post_multiband")


class LoadedPlugin:
    def __init__(self, path: Path, manifest: dict, module):
        self.path = path
        self.manifest = manifest
        self.module = module
        self.pid = manifest["id"]
        self.name = manifest.get("name", self.pid)
        self.version = str(manifest.get("version", "?"))
        self.insert = manifest["insert"]
        self.params_spec = list(manifest.get("params", []))
        self.meters_spec = list(manifest.get("meters", []))
        self.instance = None                  # set when enabled
        self.enabled = False
        self.params: dict = {p["key"]: p.get("default", 0.0)
                             for p in self.params_spec}
        self.error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "ACTIVE" if self.enabled else "loaded"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"omegafm_plugin_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate_manifest(m) -> str | None:
    if not isinstance(m, dict):
        return "PLUGIN manifest is not a dict"
    for key in ("id", "insert"):
        if key not in m:
            return f"manifest missing '{key}'"
    if m["insert"] not in INSERT_POINTS:
        return (f"insert '{m['insert']}' unknown "
                f"(use one of {', '.join(INSERT_POINTS)})")
    reserved = {"process", "reset", "set_params", "meters"}
    for prm in m.get("params", []):
        if prm.get("key") in reserved:
            return (f"param key '{prm.get('key')}' is reserved (it would "
                    f"shadow a Plugin method/attribute)")
    return None


class PluginHost:
    """Discovers plugin files, owns instances, and runs them inside the
    chain.  The audio thread only ever touches an immutable per-point
    tuple that is atomically swapped on any change."""

    def __init__(self, dirs=None, fs: float = 48000.0):
        self.dirs = [Path(d) for d in (dirs or [])]
        self.fs = fs
        self.plugins: dict[str, LoadedPlugin] = {}
        self._lock = threading.Lock()
        self._chain: dict[str, tuple] = {pt: () for pt in INSERT_POINTS}
        self.scan_errors: list[str] = []

    # ------------------------------------------------------------------ scan
    def scan(self):
        """(Re)discover plugin files; keeps enable state and params of
        plugins whose id survives the rescan."""
        old = self.plugins
        found: dict[str, LoadedPlugin] = {}
        self.scan_errors = []
        for d in self.dirs:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                try:
                    mod = _load_module(path)
                    manifest = getattr(mod, "PLUGIN", None)
                    err = _validate_manifest(manifest)
                    if err is None and not hasattr(mod, "Plugin"):
                        err = "no Plugin class"
                    if err:
                        self.scan_errors.append(f"{path.name}: {err}")
                        continue
                    lp = LoadedPlugin(path, manifest, mod)
                    # later search dirs (the user's ~/.omegafm/plugins)
                    # OVERRIDE shipped copies of the same id - dropping a
                    # newer standalone in simply replaces the built-in
                    found[lp.pid] = lp
                except Exception:
                    self.scan_errors.append(
                        f"{path.name}: {traceback.format_exc(limit=1)}")
        with self._lock:
            self.plugins = found
            for pid, lp in found.items():
                prev = old.get(pid)
                if prev is not None:
                    lp.params.update({k: v for k, v in prev.params.items()
                                      if k in lp.params})
                    if prev.enabled and prev.error is None:
                        self._enable_locked(lp)
            self._rebuild_locked()
        return list(found.values())

    # ------------------------------------------------------------------ control
    def _enable_locked(self, lp: LoadedPlugin):
        try:
            lp.instance = lp.module.Plugin(self.fs, 2)
            if lp.params:
                lp.instance.set_params(**lp.params)
            lp.enabled = True
            lp.error = None
        except Exception:
            lp.instance = None
            lp.enabled = False
            lp.error = traceback.format_exc()

    def set_enabled(self, pid: str, on: bool):
        with self._lock:
            lp = self.plugins.get(pid)
            if lp is None:
                return
            if on:
                self._enable_locked(lp)
            else:
                lp.enabled = False
                lp.instance = None
                lp.error = None
            self._rebuild_locked()

    def set_param(self, pid: str, key: str, value):
        with self._lock:
            lp = self.plugins.get(pid)
            if lp is None:
                return
            lp.params[key] = value
            if lp.instance is not None:
                try:
                    lp.instance.set_params(**{key: value})
                except Exception:
                    lp.error = traceback.format_exc()

    def reinit(self, fs: float | None = None):
        """Fresh instances for all enabled plugins (engine start)."""
        with self._lock:
            if fs:
                self.fs = fs
            for lp in self.plugins.values():
                if lp.enabled:
                    self._enable_locked(lp)
            self._rebuild_locked()

    def _rebuild_locked(self):
        chain = {pt: [] for pt in INSERT_POINTS}
        for lp in self.plugins.values():
            if lp.enabled and lp.instance is not None:
                chain[lp.insert].append(lp)
        self._chain = {pt: tuple(v) for pt, v in chain.items()}

    # ------------------------------------------------------------------ audio
    def run(self, point: str, x: np.ndarray) -> np.ndarray:
        """Called from the audio thread.  Crash-contained."""
        active = self._chain.get(point, ())
        for lp in active:
            try:
                y = lp.instance.process(x)
                if (not isinstance(y, np.ndarray) or y.shape != x.shape):
                    raise ValueError(
                        f"plugin returned shape {getattr(y, 'shape', None)}"
                        f", expected {x.shape}")
                x = y
            except Exception:
                lp.error = traceback.format_exc()
                lp.enabled = False
                lp.instance = None
                with self._lock:
                    self._rebuild_locked()
        return x

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        return {pid: {"enabled": lp.enabled, "params": dict(lp.params)}
                for pid, lp in self.plugins.items()}

    def load_state(self, st: dict, exclusive: bool = False):
        """Apply a saved rack state. With exclusive=True (preset import)
        every discovered plugin not enabled by `st` is disabled, so the
        rack ends up exactly as the preset describes."""
        if not isinstance(st, dict):
            return
        with self._lock:
            for pid, s in st.items():
                lp = self.plugins.get(pid)
                if lp is None or not isinstance(s, dict):
                    continue
                for k, v in (s.get("params") or {}).items():
                    if k in lp.params and isinstance(v, (int, float)):
                        lp.params[k] = float(v)
                if s.get("enabled"):
                    self._enable_locked(lp)
                elif exclusive:
                    lp.enabled = False
                    lp.instance = None
                    lp.error = None
            if exclusive:
                for pid, lp in self.plugins.items():
                    if pid not in st and lp.enabled:
                        lp.enabled = False
                        lp.instance = None
                        lp.error = None
            self._rebuild_locked()
