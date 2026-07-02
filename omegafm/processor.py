"""
omegafm.processor
=================

AudioEngine - real-time host around the DSP chain.

Modes
-----
stereo :  duplex device @ 48 kHz.  Chain runs 48k -> 4x up -> final
          192k section -> optional de-emphasis -> 4x down -> device.
          (De-emphasized monitoring gives a flat response after the
          clipper - what you'd hear off a tuner.)
mpx    :  duplex device @ 192 kHz.  Input decimated 4:1 into the 48k
          front end, then the same 192k back end feeds the stereo
          generator; the mono MPX composite is duplicated to both
          output channels.  Requires a true 192 kHz interface (flat to
          ~60 kHz) so the 57 kHz RDS subcarrier survives the DAC.

sounddevice is imported lazily so the DSP stack runs headless (tests,
CI, containers without PortAudio).
"""

from __future__ import annotations

import collections
import threading
import traceback
import numpy as np

from .processing_chain import FrontChain48, FinalSection192
from .stereo_generator import StereoGenerator
from .dsp.resample import Upsampler4, Decimator4
from .dsp.filters import DeEmphasis
from .dsp.dynamics import CONTROL_BLOCK
from .params import ParamStore, resolve, neutralize_base, DEFAULTS
from .meters import MeterStore

FS_FRONT = 48000.0
FS_BACK = 192000.0
BUFFER_SIZES = (256, 512, 1024, 2048)


def _sd():
    import sounddevice as sd
    return sd


def list_devices():
    """[{index, name, hostapi, max_in, max_out, default_sr, is_asio}] or []."""
    try:
        sd = _sd()
        apis = sd.query_hostapis()
        out = []
        for i, d in enumerate(sd.query_devices()):
            api = apis[d["hostapi"]]["name"]
            out.append({
                "index": i, "name": d["name"], "hostapi": api,
                "max_in": d["max_input_channels"],
                "max_out": d["max_output_channels"],
                "default_sr": d["default_samplerate"],
                "is_asio": "asio" in api.lower(),
            })
        return out
    except Exception:
        return []


class DSPGraph:
    """Rate-agnostic wiring of front chain + back end + stereo gen.

    Used by both the realtime engine and the offline validator.
    """

    def __init__(self, plugin_host=None):
        self.front = FrontChain48(FS_FRONT, plugin_host=plugin_host)
        self.up = Upsampler4(FS_FRONT)
        self.final = FinalSection192(FS_BACK)
        self.stereo_gen = StereoGenerator(FS_BACK)
        self.deemph = DeEmphasis(FS_BACK, 2, 75.0)
        self.down = Decimator4(FS_BACK)
        self.params: dict = resolve("AGGRESSIVE", "LIGHT")
        self.apply_params(self.params)

    def apply_params(self, p: dict):
        self.params = p
        self.front.apply_params(p)
        self.final.apply_params(p)
        self.stereo_gen.apply_params(p)
        self.deemph.set_tau(p["preemph_us"] if p.get("monitor_deemph", True) else 0.0)

    def latency_ms(self, blocksize48: int) -> float:
        dsp = (self.up.delay_out + self.final.latency_samples
               + self.stereo_gen.mask_delay) / FS_BACK
        return (dsp + blocksize48 / FS_FRONT) * 1000.0

    # both return 192k-domain data ------------------------------------------------
    def run_48(self, x48: np.ndarray) -> np.ndarray:
        y = self.front.process(x48)
        y = self.up.process(y)
        return self.final.process(y)

    def stereo_out(self, y192: np.ndarray, out_gain: float) -> np.ndarray:
        y = self.deemph.process(y192)
        y = self.down.process(y)
        return np.clip(y * out_gain, -1.0, 1.0)

    def mpx_out(self, y192: np.ndarray) -> np.ndarray:
        return self.stereo_gen.process(y192)


class AudioEngine:
    """Realtime engine on *independent* input and output streams.

    A single duplex stream forces both devices through one PortAudio
    host API at one sample rate - exactly why "pick input A + output B"
    silently fails or falls back on Windows.  Splitting the streams
    means: any input device with any output device (different host
    APIs fine), each side honours its own selection or its own system
    default, and MPX mode runs the *input* at native 48 kHz while only
    the output runs at 192 kHz.

    The DSP runs in the input callback; processed blocks cross to the
    output callback through a small lock-protected FIFO (2-block
    prefill absorbs scheduler jitter; drop-oldest on overflow absorbs
    slow clock drift between the two devices).
    """

    PREFILL = 2
    FIFO_MAX = 12

    def __init__(self, store: ParamStore, meters: MeterStore):
        self.store = store
        self.meters = meters
        self.plugin_host = None
        self.graph = DSPGraph()
        self.in_stream = None
        self.out_stream = None
        self._pver = -1
        self.underruns = 0
        self.blocksize = 1024
        self.mpx_mode = False
        self.in_dev = None
        self.out_dev = None
        self.error: str | None = None
        self._fifo = collections.deque(maxlen=self.FIFO_MAX)
        self._fifo_lock = threading.Lock()
        self._primed = False
        self._in_ch = 2

    # ------------------------------------------------------------------ control
    def configure(self, input_device=None, output_device=None,
                  blocksize: int = 1024, mpx_mode: bool = False):
        blocksize = int(blocksize)
        if blocksize % CONTROL_BLOCK:
            raise ValueError("blocksize must be a multiple of 32")
        self.in_dev = input_device
        self.out_dev = output_device
        self.blocksize = blocksize
        self.mpx_mode = mpx_mode

    def _input_channels(self, sd) -> int:
        """1 for mono mics, else 2 - so channel count never blocks a device."""
        try:
            dev = self.in_dev
            if dev is None:
                dev = sd.default.device[0]
            if dev is None or (isinstance(dev, int) and dev < 0):
                return 2
            return int(min(2, max(1, sd.query_devices(dev)["max_input_channels"])))
        except Exception:
            return 2

    def start(self):
        # realtime hygiene: the cycle collector's 10-20 ms pauses are
        # audible as stutter; steady-state DSP frees everything by
        # refcount, so park the GC while streaming (resumed on stop)
        import gc
        gc.collect()
        gc.freeze()
        gc.disable()
        sd = _sd()
        self.stop()
        self.error = None
        self.underruns = 0
        if self.plugin_host is not None:
            self.plugin_host.reinit(FS_FRONT)
        self.graph = DSPGraph(self.plugin_host)
        self._pver = -1
        self._sync_params()
        self._fifo = collections.deque(maxlen=self.FIFO_MAX)
        self._primed = False
        self._in_ch = self._input_channels(sd)

        blk = self.blocksize
        if self.mpx_mode:
            out_fs, out_frames = FS_BACK, blk * 4
        else:
            out_fs, out_frames = FS_FRONT, blk

        self.in_stream = sd.InputStream(
            samplerate=FS_FRONT, blocksize=blk, dtype="float32",
            channels=self._in_ch, device=self.in_dev, latency="low",
            callback=self._cb_in)
        self.out_stream = sd.OutputStream(
            samplerate=out_fs, blocksize=out_frames, dtype="float32",
            channels=2, device=self.out_dev, latency="low",
            callback=self._cb_out)
        self.in_stream.start()
        self.out_stream.start()

        # report the *resolved* devices so the selection is visible
        def dev_name(stream):
            try:
                return str(sd.query_devices(stream.device)["name"])[:24]
            except Exception:
                return "?"
        self.meters.push(
            running=True,
            devices=f"{dev_name(self.in_stream)} > {dev_name(self.out_stream)}",
            latency_ms=self.graph.latency_ms(blk)
            + self.PREFILL * blk / FS_FRONT * 1000.0)

    def stop(self):
        import gc
        gc.enable()
        gc.collect()
        for s in (self.in_stream, self.out_stream):
            if s is not None:
                try:
                    s.stop(); s.close()
                except Exception:
                    pass
        self.in_stream = None
        self.out_stream = None
        self.meters.push(running=False, devices="")

    @property
    def running(self) -> bool:
        return (self.in_stream is not None and self.in_stream.active
                and self.out_stream is not None and self.out_stream.active)

    # ------------------------------------------------------------------ internals
    def _sync_params(self):
        ver, p = self.store.snapshot()
        if ver != self._pver:
            self.graph.apply_params(p)
            self._pver = ver
        return self.graph.params

    def _meter_common(self, x48, y192):
        f = self.graph.front.meters
        b = self.graph.final.meters
        self.meters.push(
            input_l=MeterStore._db(f["input_peak"][0]),
            input_r=MeterStore._db(f["input_peak"][-1]),
            agc_gain=f["agc_gain"],
            lev_gain=list(f["lev_gain"]),
            comp_gr=list(f["comp_gr"]),
            hf_gr=b["hf_gr"], wb_gr=b["wb_gr"],
            underruns=self.underruns,
        )

    def _cb_in(self, indata, frames, time_info, status):
        """Input side: run the whole DSP chain, queue the output block."""
        try:
            if status:
                self.underruns += 1
            p = self._sync_params()
            x = indata.astype(np.float64)
            if x.shape[1] == 1:                      # mono mic -> stereo
                x = np.repeat(x, 2, axis=1)
            y192 = self.graph.run_48(x)
            if self.mpx_mode:
                mpx = self.graph.mpx_out(y192)
                out = np.repeat(np.clip(mpx, -1, 1)[:, None], 2,
                                axis=1).astype(np.float32)
                pk = self.graph.stereo_gen.mpx_peak
                self.meters.push(out_l=MeterStore._db(pk),
                                 out_r=MeterStore._db(pk),
                                 mpx_pct=pk * 100.0,
                                 bs412_dbr=self.graph.stereo_gen.bs412_dbr,
                                 bs412_gr=self.graph.stereo_gen.bs412_gr)
            else:
                o = self.graph.stereo_out(y192, p["output_level"])
                out = o.astype(np.float32)
                self.meters.push(out_l=MeterStore._db(np.max(np.abs(o[:, 0]))),
                                 out_r=MeterStore._db(np.max(np.abs(o[:, 1]))),
                                 mpx_pct=0.0)
            self._meter_common(x, y192)
            with self._fifo_lock:
                if len(self._fifo) == self._fifo.maxlen:
                    self._fifo.popleft()             # clock drift: drop oldest
                    self.underruns += 1
                self._fifo.append(out)
        except Exception:
            self.error = traceback.format_exc()
            raise _sd().CallbackAbort

    def _cb_out(self, outdata, frames, time_info, status):
        """Output side: pull processed blocks from the FIFO."""
        try:
            if status:
                self.underruns += 1
            buf = None
            with self._fifo_lock:
                if not self._primed:
                    if len(self._fifo) >= self.PREFILL:
                        self._primed = True
                if self._primed and self._fifo:
                    buf = self._fifo.popleft()
            if buf is None:
                outdata[:] = 0.0
                if self._primed:                     # starved: re-prime
                    self.underruns += 1
                    self._primed = False
            else:
                outdata[:] = buf
        except Exception:
            self.error = traceback.format_exc()
            outdata[:] = 0.0
            raise _sd().CallbackAbort


# Power-on plugin rack: the station-tuned factory default. Applied on a
# fresh install (no saved config / no saved rack); a user's own saved
# rack always wins afterwards.
FACTORY_RACK = {
    # station-tuned power-on rack: all nine modules, operator
    # calibrated (2026-07-02 reference export)
    "azimuth_repair": {"enabled": True, "params": {
        "correct": 10.0,
        "maxlag_us": 400.0,
        "polfix": 1.0,
    }},
    "de_clipper": {"enabled": True, "params": {
        "declip": 5.0,
        "lossy": 3.0,
        "test": 0.0,
    }},
    "de_esser": {"enabled": True, "params": {
        "freq": 6000.0,
        "thresh_db": -3.0,
        "range_db": 4.4,
        "release_ms": 51.0,
    }},
    "dehum_gate": {"enabled": True, "params": {
        "hum": 6.0,
        "harm": 5.0,
        "gate_db": -55.0,
        "range_db": 12.0,
        "test": 0.0,
    }},
    "natural_dynamics": {"enabled": True, "params": {
        "amount": 3.0,
        "sens": 4.0,
        "speed_ms": 60.0,
    }},
    "power_bass": {"enabled": True, "params": {
        "drive_db": 3.5,
        "freq": 120.0,
        "punch": 0.25,
        "trim_db": 0.0,
    }},
    "sonic_maximizer": {"enabled": True, "params": {
        "proc": 2.0,
        "contour": 1.9,
        "out_db": -0.5,
    }},
    "stereo_governor": {"enabled": True, "params": {
        "ceiling_db": -3.0,
        "release_ms": 250.0,
        "range_db": 12.0,
    }},
    "stereo_widener": {"enabled": True, "params": {
        "width": 1.25,
        "mono_hz": 107.0,
    }},
}


class Controller:
    """Facade the UI talks to: presets, params, engine lifecycle, RDS."""

    def __init__(self):
        self.meters = MeterStore()
        self.signature = "AGGRESSIVE"
        self.density = "LIGHT"
        self.trims: dict = {}
        self.custom_base: dict = {}
        self.preset_name: str = ""
        self.store = ParamStore(resolve(self.signature, self.density))
        self.engine = AudioEngine(self.store, self.meters)
        from .plugin_host import PluginHost
        from pathlib import Path
        app_dir = Path(__file__).resolve().parents[1] / "plugins"
        user_dir = Path.home() / ".omegafm" / "plugins"
        self.plugin_host = PluginHost([app_dir, user_dir], fs=48000.0)
        self.plugin_host.scan()
        self.engine.plugin_host = self.plugin_host
        # live graph (used when idle / for RDS) shares the host too
        self.engine.graph = DSPGraph(self.plugin_host)

    # ------------------------------------------------------------------ presets
    def _keep_station_trims(self):
        """Preset changes reset the sound, never station/local setup."""
        from .presets import EXCLUDE
        self.trims = {k: v for k, v in self.trims.items() if k in EXCLUDE}

    def set_preset(self, signature=None, density=None):
        if signature:
            self.signature = signature
        if density:
            self.density = density
        self._keep_station_trims()
        self._push()

    def set_trim(self, **kv):
        self.trims.update(kv)
        self._push()

    def save_custom(self):
        _, p = self.store.snapshot()
        self.custom_base = neutralize_base(p, self.density)
        self.signature = "CUSTOM"
        self.preset_name = "CUSTOM"
        self._push()

    # ------------------------------------------------------------------ presets
    def export_preset(self, path):
        from . import presets
        _, p = self.store.snapshot()
        name = self.preset_name if self.signature == "CUSTOM" else self.signature
        return presets.export_file(path, p, self.signature, self.density,
                                   name=None if not name else name,
                                   plugins=self.plugin_host.state())

    def import_preset(self, path):
        import copy
        from . import presets
        name, density, params, rack = presets.load_file(path)
        den = density or self.density
        # density-neutral base: defaults for anything the file omits,
        # file values un-baked from the density they were exported under
        base = {k: copy.deepcopy(DEFAULTS[k]) for k in presets.PRESET_KEYS}
        if "comp_thresholds_db" in params or "comp_release_ms" in params:
            tmp = dict(base); tmp.update(params)
            tmp = neutralize_base(tmp, den)
            for k in ("comp_thresholds_db", "comp_release_ms"):
                if k in params:
                    params[k] = tmp[k]
        base.update(params)
        self.custom_base = base
        if density:
            self.density = density
        self.signature = "CUSTOM"
        self.preset_name = name.upper()
        self._keep_station_trims()
        if rack is not None:
            self.plugin_host.load_state(rack, exclusive=True)
        self._push()
        return name

    def _push(self):
        p = resolve(self.signature, self.density, self.trims, self.custom_base)
        self.store.update(p)

    def current_params(self) -> dict:
        return self.store.snapshot()[1]

    # ------------------------------------------------------------------ rds live
    def rds_update(self, ps=None, rt=None):
        self.engine.graph.stereo_gen.rds.set_text(ps=ps, rt=rt)

    # ------------------------------------------------------------------ engine
    def start(self, input_device, output_device, blocksize, mpx_mode):
        self.engine.configure(input_device, output_device, blocksize, mpx_mode)
        self.set_trim(mpx_mode=mpx_mode)
        self.engine.start()

    def stop(self):
        self.engine.stop()
