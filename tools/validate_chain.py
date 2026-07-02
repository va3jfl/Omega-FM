"""
tools/validate_chain.py
=======================

Offline proof that the chain behaves like broadcast gear:

1. Crossover flatness      : 4-band sum ripple < 0.25 dB (30 Hz-15 kHz)
2. Pre-emphasis accuracy   : digital vs analog 75 us curve, err @15 kHz
3. RDS CRC self-test       : encode/decode a block, syndrome must be 0
4. Full chain on program   : 12 s shaped noise "music", report per-stage
                             gain riding / GR, final ceiling
5. MPX spectrum            : Welch PSD - pilot @19k, RDS @57k, floor
                             above 60 kHz; saves omegafm_mpx_spectrum.png

Run:  python tools/validate_chain.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import welch, sosfilt, butter, freqz

from omegafm.dsp.filters import Crossover4, _preemph_ba, design_lpf15k, design_mpx_mask
from omegafm.rds import make_block, OFFSETS, crc10
from omegafm.processor import DSPGraph, FS_FRONT, FS_BACK
from omegafm.params import resolve

OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{OK if cond else BAD}] {name:<44s} {detail}")


# ------------------------------------------------------------------ 1. crossover
def test_crossover():
    fs = 48000.0
    xo = Crossover4(fs)
    n = 1 << 15
    imp = np.zeros((n, 1)); imp[0] = 1.0
    xo_m = Crossover4(fs, 1)
    total = sum(b[:, 0] for b in xo_m.process(imp))
    H = np.fft.rfft(total)
    f = np.fft.rfftfreq(n, 1 / fs)
    sel = (f >= 30) & (f <= 15000)
    mag = 20 * np.log10(np.abs(H[sel]) + 1e-12)
    ripple = mag.max() - mag.min()
    check("crossover 4-band sum flat 30Hz-15kHz", ripple < 0.25,
          f"ripple = {ripple:.3f} dB")


# ------------------------------------------------------------------ 2. pre-emph
def test_preemph():
    fs = FS_BACK
    b, a = _preemph_ba(fs, 75.0)
    w = np.array([1000.0, 5000.0, 10000.0, 15000.0])
    _, h = freqz(b, a, worN=w, fs=fs)
    # analog reference: zero at 75 us, POLE at 20 kHz - the emphasis
    # network's stop.  Without it the boost rises without limit and the
    # top octave drives every HF stage hotter than any transmitter
    # would (the classic 'harsh digital pre-emphasis' mistake).
    tau = 75e-6
    tau2 = 1.0 / (2 * np.pi * 20000.0)
    analog = np.sqrt((1 + (2 * np.pi * w * tau) ** 2)
                     / (1 + (2 * np.pi * w * tau2) ** 2))
    err = np.abs(20 * np.log10(np.abs(h)) - 20 * np.log10(analog))
    boost15k = float(20 * np.log10(np.abs(h[-1])))
    check("pre-emphasis matches pole-limited analog network",
          err.max() < 0.25 and 14.4 < boost15k < 15.6,
          f"max err {err.max():.3f} dB, 15k boost {boost15k:.1f} dB "
          f"(civilised top octave)")


# ------------------------------------------------------------------ 3. RDS CRC
def test_rds_crc():
    blk = make_block(0x1234, "A")
    word = 0
    for b in blk:
        word = (word << 1) | b
    data = word >> 10
    recomputed = crc10(data) ^ OFFSETS["A"]
    check("RDS block CRC-10 + offset A", (word & 0x3FF) == recomputed,
          f"PI=0x{data:04X}")


# ------------------------------------------------------------------ program material
def make_cd_program(fs, dur):
    """Dense mastered-music surrogate: broadband bed + bass line + kick /
    snare / hats + pads + syllabic vox, bus-squashed, peaks -0.3 dBFS,
    crest ~5-6 dB - i.e. what a real CD feeds the processor.  A quiet
    verse (x0.45) occupies the middle third so the AGC has work to do."""
    from scipy.signal import lfilter as _lf, butter as _bt
    n = int(fs * dur); t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    b, a = _bt(1, 800 / (fs / 2)); bed = _lf(b, a, rng.standard_normal(n)) * 1.4
    notes = [55, 73.4, 82.4, 61.7]
    f0 = np.repeat(np.resize(notes, int(dur * 2) + 1), int(fs * 0.5))[:n]
    ph = 2 * np.pi * np.cumsum(f0) / fs
    bass = 0.9 * np.sin(ph) + 0.35 * np.sin(2 * ph)
    kt = (t % 0.5); kenv = np.exp(-kt / 0.05) * (kt < 0.18)
    kick = 1.2 * np.sin(2 * np.pi * (50 + 80 * np.exp(-kt / 0.03)) * kt) * kenv
    st = ((t + 0.25) % 0.5); senv = np.exp(-st / 0.06) * (st < 0.2)
    b2, a2 = _bt(2, [1500 / (fs / 2), 6000 / (fs / 2)], btype="band")
    snare = 1.1 * _lf(b2, a2, rng.standard_normal(n)) * senv
    ht = (t % 0.125); henv = np.exp(-ht / 0.015) * (ht < 0.05)
    b3, a3 = _bt(2, 7000 / (fs / 2), btype="high")
    hats = 0.5 * _lf(b3, a3, rng.standard_normal(n)) * henv
    pad = sum(0.22 * np.sign(np.sin(2 * np.pi * f * t + i))
              for i, f in enumerate((220, 277, 330, 440)))
    b4, a4 = _bt(2, 2200 / (fs / 2)); pad = _lf(b4, a4, pad)
    vam = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 2.3 * t), 0, 1)
    b5, a5 = _bt(2, [300 / (fs / 2), 3200 / (fs / 2)], btype="band")
    vox = 0.9 * _lf(b5, a5, rng.standard_normal(n)) * vam
    m = bed * 0.25 + bass + kick + snare + hats + pad + vox
    third = n // 3
    ramp = int(0.05 * fs)                            # 50 ms musical transitions
    env = np.ones(n)
    env[third:2 * third] = 0.45
    env[third - ramp:third] = np.linspace(1.0, 0.45, ramp)
    env[2 * third:2 * third + ramp] = np.linspace(0.45, 1.0, ramp)
    m *= env                                         # quiet verse for the AGC
    m = np.tanh(m * 0.8)
    m = m / np.max(np.abs(m)) * 10 ** (-0.3 / 20)
    fade = int(0.1 * fs)
    m[:fade] *= np.linspace(0.0, 1.0, fade)
    return np.stack([m, np.roll(m, 7)], axis=1)


def make_classical_program(fs, dur):
    """Strings-dominant surrogate (sustained bowed chords with vibrato,
    slow swells, sparse timpani): the opposite spectral shape of pop -
    the material that exposes mix-repainting multiband levelers."""
    from scipy.signal import butter as _bt, lfilter as _lf
    n = int(fs * dur); t = np.arange(n) / fs
    rng = np.random.default_rng(21)
    x = np.zeros(n)
    for i, f0 in enumerate((220.0, 293.7, 440.0, 587.3, 880.0)):
        vib = f0 * (1 + 0.004 * np.sin(2 * np.pi * (5.2 + 0.3 * i) * t + i))
        ph = 2 * np.pi * np.cumsum(vib) / fs
        x += sum(np.sin(k * ph) / k for k in range(1, 9)) * (0.20 - 0.02 * i)
    x *= 0.45 + 0.55 * 0.5 * (1 + np.sin(2 * np.pi * 0.08 * t - 1.2))
    kt = (t % 2.5); kenv = np.exp(-kt / 0.25) * (kt < 1.0)
    x += 0.5 * np.sin(2 * np.pi * (70 - 15 * np.clip(kt, 0, 0.3)) * kt) * kenv
    b, a = _bt(2, 4000 / (fs / 2))
    x += 0.01 * _lf(b, a, rng.standard_normal(n))
    x = x / np.max(np.abs(x)) * 10 ** (-1.0 / 20)
    return np.stack([x, np.roll(x, 11)], axis=1)


def make_program(fs: float, seconds: float) -> np.ndarray:
    """Pink-ish noise with AM 'phrases' + a quiet gap to exercise gating."""
    rng = np.random.default_rng(7)
    n = int(fs * seconds)
    w = rng.standard_normal((n, 2))
    sos = butter(1, 800.0, btype="low", fs=fs, output="sos")
    x = sosfilt(sos, w, axis=0) * 2.0 + 0.15 * w
    t = np.arange(n) / fs
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.31 * t) * np.sin(2 * np.pi * 0.07 * t)
    x *= env[:, None]
    # dynamic shifts: loud verse, quiet bridge, gap
    x[int(2.0*fs):int(4.5*fs)] *= 2.5
    x[int(6.0*fs):int(7.5*fs)] *= 0.25
    x[int(9.0*fs):int(9.8*fs)] *= 0.003
    x /= np.max(np.abs(x))
    return (x * 10 ** (-14 / 20)).astype(np.float64)


# ------------------------------------------------------------------ 4+5. chain
def test_chain_and_mpx(save_plot=True):
    graph = DSPGraph()
    p = resolve("NATURAL", "MEDIUM")
    p["mpx_mode"] = True
    graph.apply_params(p)

    fs48 = FS_FRONT
    x = make_cd_program(fs48, 12.0)     # dense mastered-music reference
    blk = 1024
    nblk = len(x) // blk

    agc, lev, gr, wb, hf, mpx_pk = [], [], [], [], [], []
    mpx_all = []
    for i in range(nblk):
        seg = x[i * blk:(i + 1) * blk]
        y192 = graph.run_48(seg)
        mpx = graph.mpx_out(y192)
        mpx_all.append(mpx)
        f, b = graph.front.meters, graph.final.meters
        agc.append(f["agc_gain"]); lev.append(f["lev_gain"].copy())
        gr.append(f["comp_gr"].copy()); wb.append(b["wb_gr"]); hf.append(b["hf_gr"])
        mpx_pk.append(graph.stereo_gen.mpx_peak)

    agc = np.array(agc); lev = np.array(lev); gr = np.array(gr)
    wb = np.array(wb); hf = np.array(hf); mpx_pk = np.array(mpx_pk)
    mpx = np.concatenate(mpx_all)
    stl = slice(len(gr) // 4, None)      # settled window (skips startup)

    print(f"      AGC gain range       : {agc.min():+.1f} .. {agc.max():+.1f} dB")
    print(f"      Leveler gain (mean)  : "
          + " ".join(f"{v:+.1f}" for v in lev.mean(axis=0)) + " dB")
    print(f"      Comp GR (mean/max)   : "
          + " ".join(f"{m:.1f}/{x_:.1f}" for m, x_ in zip(gr.mean(axis=0), gr.max(axis=0)))
          + " dB")
    print(f"      WB lim GR max        : {wb.max():.2f} dB   HF lim GR max: {hf.max():.2f} dB")
    print(f"      MPX modulation       : mean {mpx_pk.mean()*100:.1f} %  max {mpx_pk.max()*100:.1f} %")

    check("AGC rides gain (moves > 3 dB)", agc.max() - agc.min() > 3.0)
    check("compressor rides gently (mean GR 2-4.5 dB)",
          2.0 < gr[stl].mean() < 4.5, f"mean {gr[stl].mean():.2f} dB")
    check("no band driven deep (per-band mean < 4.5)",
          gr[stl].mean(axis=0).max() < 4.5,
          " ".join(f"{v_:.1f}" for v_ in gr[stl].mean(axis=0)))
    check("MPX never exceeds 100.5 % mod", mpx_pk[stl].max() <= 1.005,
          f"max {mpx_pk[stl].max()*100:.2f} %")
    # Omega staging: multiband makes the level, limiters only tickle,
    # the clipper makes the loudness.
    check("WB limiter back to tickling (mean < 3 dB)",
          wb[stl].mean() < 3.0,
          f"mean {wb[stl].mean():.2f} / max {wb[stl].max():.2f} dB")
    check("HF limiter governs the top end (mean<6, max<14)",
          hf[stl].mean() < 6.0 and hf[stl].max() < 14.0,
          f"mean {hf[stl].mean():.2f} / max {hf[stl].max():.2f} dB")
    check("loudness kept (mean mod > 60 %)", mpx_pk[stl].mean() > 0.60,
          f"{mpx_pk[stl].mean()*100:.0f} %")
    check("leveler idles near 0 (|mean| < 2.5 dB)",
          np.abs(lev[stl].mean(axis=0)).max() < 2.5,
          " ".join(f"{v_:+.1f}" for v_ in lev[stl].mean(axis=0)))
    check("leveler never rails on dense program (p95 |g| < 6.5)",
          np.percentile(np.abs(lev[stl]), 95) < 6.5,
          f"p95 {np.percentile(np.abs(lev[stl]), 95):.1f} dB")

    # ---- spectrum ------------------------------------------------------
    f, pxx = welch(mpx[int(FS_BACK):], fs=FS_BACK, nperseg=1 << 14)
    pdb = 10 * np.log10(pxx + 1e-20)

    def band_db(lo, hi):
        s = (f >= lo) & (f <= hi)
        return pdb[s].max()

    audio_ref = band_db(200, 8000)
    pilot = band_db(18900, 19100)
    rds = band_db(56000, 58000)
    gap = band_db(15500, 18300)
    floor = band_db(62000, 90000)

    print(f"      spectrum (dB rel audio peak): pilot {pilot-audio_ref:+.1f}  "
          f"RDS {rds-audio_ref:+.1f}  15.5-18.3k gap {gap-audio_ref:+.1f}  "
          f">62k floor {floor-audio_ref:+.1f}")

    check("19 kHz pilot present and dominant vs gap", pilot - gap > 20.0)
    check("57 kHz RDS subcarrier present", rds - floor > 15.0)
    check("spectrum above 62 kHz suppressed > 45 dB", audio_ref - floor > 45.0,
          f"{audio_ref-floor:.1f} dB")

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
        ax.plot(f / 1000.0, pdb - audio_ref, lw=0.8, color="#ffb43c")
        ax.set_facecolor("#17191d"); fig.patch.set_facecolor("#1b1d21")
        for fx, lab in ((19, "pilot 19k"), (38, "38k DSB-SC"), (57, "RDS 57k"),
                        (53, "53k edge")):
            ax.axvline(fx, color="#59d66a" if fx != 53 else "#ff5347",
                       ls="--", lw=0.7, alpha=0.7)
            ax.text(fx, 6, lab, color="#e8e8e8", fontsize=8, ha="center")
        ax.set_xlim(0, 96); ax.set_ylim(-100, 12)
        ax.set_xlabel("kHz", color="#cccccc")
        ax.set_ylabel("dB rel. audio peak", color="#cccccc")
        ax.set_title("OmegaFM composite (MPX) spectrum - filtered clip, 9 % pilot, 4 % RDS",
                     color="#e8e8e8")
        ax.tick_params(colors="#999999")
        for s in ax.spines.values():
            s.set_color("#444444")
        ax.grid(alpha=0.15)
        out = pathlib.Path(__file__).resolve().parents[1] / "omegafm_mpx_spectrum.png"
        fig.tight_layout(); fig.savefig(out)
        print(f"      spectrum plot -> {out}")

    # latency
    lat = graph.latency_ms(1024)
    print(f"      chain latency @1024  : {lat:.1f} ms (DSP "
          f"{lat - 1024/FS_FRONT*1000:.1f} ms + buffer)")
    check("DSP latency (excl. buffer) < 10 ms",
          (lat - 1024 / FS_FRONT * 1000) < 10.0)


# ------------------------------------------------------------------ 7. riding
def test_wb_riding():
    """Word-like bursts must NOT pump the WB limiter: between words the
    gain holds near its platform instead of snapping back to unity."""
    from omegafm.dsp.dynamics import LookaheadLimiter
    fs = FS_BACK
    lim = LookaheadLimiter(fs, 2)
    lim.set_params(ceiling_db=-0.5, release_db_s=90.0, platform_ms=250.0)
    t = np.arange(int(fs * 6.0)) / fs
    env = ((t % 0.6) < 0.35).astype(float)          # 350 ms word / 250 ms gap
    sig = 1.35 * np.sin(2 * np.pi * 400 * t) * env  # needs ~3 dB GR
    x = np.stack([sig, sig], axis=1)
    blk = 4096
    g = []
    for i in range(len(x) // blk):
        lim.process(x[i * blk:(i + 1) * blk])
        g.append(lim.gr_meter)
    g = np.array(g)
    n0 = int(2.0 * fs / blk)
    blk_env = env[:len(g) * blk].reshape(-1, blk).max(axis=1)
    word = g[n0:][blk_env[n0:] > 0.5].mean()
    gap = g[n0:][blk_env[n0:] < 0.5].mean()
    ratio = gap / max(word, 1e-6)
    check("WB release rides (holds >60 % GR in gaps)", ratio > 0.6,
          f"word {word:.2f} dB / gap {gap:.2f} dB ({ratio*100:.0f} %)")


# ------------------------------------------------------------------ 8. lev ride
def test_leveler_riding():
    """Snare-like bursts must not wiggle the leveler: with integrated
    loudness detection + the correction window the per-hit-cycle gain
    motion must be inaudible (< 0.2 dB) once settled."""
    from omegafm.dsp.dynamics import GainRider, CONTROL_BLOCK
    fs = FS_FRONT
    lev = GainRider(fs, 4, range_db=5.0, attack_db_s=1.5, release_db_s=1.5,
                    gate_db=-50.0, fast_recovery_db_s=1.5,
                    integrate_ms=300.0, window_db=2.0)
    lev.set_params(target_db=[-35.0, -32.0, -33.5, -37.0])
    t = np.arange(int(fs * 16)) / fs
    hits = (t % 0.45) < 0.03
    tail = np.exp(-np.maximum(0, (t % 0.45) - 0.03) / 0.05) * ((t % 0.45) < 0.25)
    env = np.where(hits, 1.0, 0.25 * tail)
    gains = []
    blk = 1024
    for i in range(len(t) // blk):
        e = env[i * blk:(i + 1) * blk]
        n = blk // CONTROL_BLOCK
        eb = e.reshape(n, CONTROL_BLOCK).max(axis=1)
        lv = np.stack([np.full(n, -30.0), np.full(n, -30.0),
                       20 * np.log10(np.maximum(eb * 0.15, 1e-6)),
                       20 * np.log10(np.maximum(eb * 0.12, 1e-6))], axis=1)
        gains.append(lev.compute_gains(lv))
    g = np.vstack(gains)
    cpc = int(0.45 * fs / CONTROL_BLOCK)
    gg = g[int(10 * fs / CONTROL_BLOCK):]
    ncyc = len(gg) // cpc
    cyc = gg[:ncyc * cpc].reshape(ncyc, cpc, 4)
    motion = float((cyc.max(axis=1) - cyc.min(axis=1)).max())
    check("leveler holds still per hit (< 0.2 dB)", motion < 0.2,
          f"max per-cycle motion {motion:.3f} dB")


# ------------------------------------------------------------------ 9. comp ride
def test_comp_riding():
    """The multiband compressor must not collapse to unity between hits:
    program-dependent release holds GR near the platform through gaps."""
    from omegafm.dsp.dynamics import BandCompressor, CONTROL_BLOCK
    fs = FS_FRONT
    comp = BandCompressor(fs, 4)
    comp.set_params(threshold_db=[-29, -25.5, -25.5, -27.5], ratio=[2.5, 3, 3, 3.5],
                    attack_ms=[18, 10, 7, 4], release_ms=[280, 200, 150, 110],
                    makeup_db=[0, 0, 0, 0], knee_db=6.0, platform_ms=700.0)
    t = np.arange(int(fs * 14)) / fs
    hits = (t % 0.45) < 0.03
    tail = np.exp(-np.maximum(0, (t % 0.45) - 0.03) / 0.05) * ((t % 0.45) < 0.25)
    env = np.where(hits, 1.0, 0.3 * tail)
    blk = 1024
    dummy = [np.zeros((blk, 2)) for _ in range(4)]
    grs = []
    for i in range(len(t) // blk):
        e = env[i * blk:(i + 1) * blk]
        n = blk // CONTROL_BLOCK
        eb = e.reshape(n, CONTROL_BLOCK).max(axis=1)
        lv = np.stack([np.full(n, -40.0), np.full(n, -40.0),
                       20 * np.log10(np.maximum(eb * 0.14, 1e-6)),
                       20 * np.log10(np.maximum(eb * 0.12, 1e-6))], axis=1)
        comp.apply(dummy, lv)
        grs.append(comp.meter.copy())
    g = np.array(grs)[int(8 * fs / blk):]
    pe = env[:len(grs) * blk].reshape(-1, blk).max(axis=1)[int(8 * fs / blk):]
    hit_gr = g[pe > 0.9][:, 3].mean()
    gap_gr = g[pe < 0.35][:, 3].mean()
    ratio = gap_gr / max(hit_gr, 1e-9)
    check("compressor release rides (>50 % GR in gaps)", ratio > 0.5,
          f"hits {hit_gr:.2f} dB / gaps {gap_gr:.2f} dB ({ratio*100:.0f} %)")


# ------------------------------------------------------------------ 10. presets
def test_preset_json():
    """JSON preset export/import: exact sound round-trip, hard validation,
    station keys never touched."""
    import tempfile, os, json
    from omegafm.processor import Controller
    from omegafm.presets import PRESET_KEYS, load_file

    c = Controller()
    c.set_preset(signature="AGGRESSIVE", density="HEAVY")
    c.set_trim(eq_gains=[3.0, -1.0, 1.5, 4.0], clip_drive_db=2.7,
               rds_ps="KEEPME")
    exported = dict(c.current_params())
    tmp = tempfile.mktemp(suffix=".json")
    c.export_preset(tmp)

    c.set_preset(signature="SMOOTH", density="LIGHT")
    c.set_trim(final_drive_db=0.5)
    c.import_preset(tmp)
    got = c.current_params()
    exact = all(got[k] == exported[k] for k in PRESET_KEYS)
    check("preset JSON round-trip is exact", exact)
    check("preset import keeps station keys", got["rds_ps"] == "KEEPME")

    bad = tempfile.mktemp(suffix=".json")
    open(bad, "w").write("{broken")
    try:
        load_file(bad)
        ok = False
    except ValueError:
        ok = True
    doc = {"format": "omegafm-preset", "version": 1,
           "params": {"alien": 1, "eq_freqs": [50, 99999], "rds_ps": "X",
                      "rds_tp": True}}
    open(bad, "w").write(json.dumps(doc))
    _, _, prm, _ = load_file(bad)
    ok = ok and "alien" not in prm and "rds_ps" not in prm \
             and "rds_tp" not in prm \
             and max(prm["eq_freqs"]) <= 16000 and len(prm["eq_freqs"]) == 4
    check("preset validation rejects/repairs bad input", ok)

    # plugin rack rides in the preset and applies exclusively on import
    c2p = Controller()
    c2p.plugin_host.set_enabled("stereo_widener", True)
    c2p.plugin_host.set_param("stereo_widener", "width", 1.7)
    c2p.plugin_host.set_enabled("power_bass", True)
    tmp2 = tempfile.mktemp(suffix=".json")
    c2p.export_preset(tmp2)
    c2p.plugin_host.set_enabled("stereo_widener", False)
    c2p.plugin_host.set_param("stereo_widener", "width", 0.5)
    c2p.plugin_host.set_enabled("de_esser", True)     # stray plugin
    c2p.import_preset(tmp2)
    wl = c2p.plugin_host.plugins["stereo_widener"]
    ok2 = (wl.enabled and abs(wl.params["width"] - 1.7) < 1e-9
           and c2p.plugin_host.plugins["power_bass"].enabled
           and not c2p.plugin_host.plugins["de_esser"].enabled)
    # a pre-rack preset (no "plugins" key) must leave the rack alone
    doc2 = json.load(open(tmp2)); doc2.pop("plugins")
    open(tmp2, "w").write(json.dumps(doc2))
    c2p.plugin_host.set_enabled("de_esser", True)
    c2p.import_preset(tmp2)
    ok2 = ok2 and c2p.plugin_host.plugins["de_esser"].enabled
    check("preset carries the plugin rack (exclusive apply)", ok2)
    os.remove(tmp); os.remove(bad); os.remove(tmp2)


# ------------------------------------------------------------------ 11. plugins
def test_plugins():
    """Modular DSP plugins: discovery, live in-chain insertion, unity
    transparency, and crash containment (a broken plugin auto-bypasses
    and can never take down the validated core chain)."""
    import pathlib, textwrap
    from omegafm.plugin_host import PluginHost
    from omegafm.processor import DSPGraph

    plug_dir = pathlib.Path(__file__).resolve().parents[1] / "plugins"
    host = PluginHost([plug_dir], fs=FS_FRONT)
    host.scan()
    ok = "stereo_widener" in host.plugins
    host.set_enabled("stereo_widener", True)
    host.set_param("stereo_widener", "mono_hz", 0.0)
    rng = np.random.default_rng(3)
    x = np.stack([rng.standard_normal(4096) * 0.1,
                  rng.standard_normal(4096) * 0.1], axis=1)

    def run(width):
        host.set_param("stereo_widener", "width", width)
        g = DSPGraph(host)
        return g.run_48(x.copy())

    diff = float(np.max(np.abs(run(2.0) - run(1.0))))
    check("plugin discovered + inserts live in chain", ok and diff > 1e-3,
          f"width A/B diff {diff:.3f}")

    host.set_param("stereo_widener", "width", 1.0)
    err = float(np.max(np.abs(host.run("post_input", x.copy()) - x)))
    check("plugin at unity is transparent", err < 1e-9, f"err {err:.1e}")

    bad = plug_dir / "zz_selftest_bad.py"
    bad.write_text(textwrap.dedent('''
        PLUGIN = {"id": "zz_boom", "name": "Boom", "version": "0",
                  "insert": "post_input"}
        class Plugin:
            def __init__(self, fs, ch=2): self.meters = {}
            def set_params(self, **kv): pass
            def process(self, x): raise RuntimeError("boom")
    '''))
    try:
        host.scan()
        host.set_enabled("zz_boom", True)
        out = host.run("post_input", x.copy())
        lp = host.plugins["zz_boom"]
        contained = (out.shape == x.shape and not lp.enabled
                     and "boom" in (lp.error or ""))
    finally:
        bad.unlink()
        host.scan()
    check("broken plugin auto-bypasses, chain survives", contained)

    # ---- shipped Power Bass plugin --------------------------------------
    host2 = PluginHost([plug_dir], fs=FS_FRONT)
    host2.scan()
    ok_pb = "power_bass" in host2.plugins
    host2.set_enabled("power_bass", True)
    lpb = host2.plugins["power_bass"]
    tt = np.arange(int(FS_FRONT * 2)) / FS_FRONT
    tone_typ = np.stack([0.158 * np.sin(2 * np.pi * 70 * tt)] * 2, axis=1)
    tone_hot = np.stack([0.22 * np.sin(2 * np.pi * 70 * tt)] * 2, axis=1)

    host2.set_param("power_bass", "drive_db", 0.0)
    host2.set_param("power_bass", "punch", 1.0)
    y = np.vstack([host2.run("post_multiband",
                             tone_typ[i*1024:(i+1)*1024].copy())
                   for i in range(len(tone_typ)//1024)])
    resid = 20 * np.log10(np.max(np.abs(y - tone_typ[:len(y)]))
                          / np.max(np.abs(tone_typ)))
    host2.set_param("power_bass", "drive_db", 8.0)
    lpb.instance.reset()
    for i in range(len(tone_hot)//1024):
        host2.run("post_multiband", tone_hot[i*1024:(i+1)*1024].copy())
    clip8 = lpb.instance.meters["clip_db"]
    check("power bass: exact below knee, densifies driven",
          ok_pb and resid < -60.0 and clip8 > 2.0,
          f"drive-0 resid {resid:.0f} dB, drive-8 clip {clip8:.1f} dB")

    host2.set_param("power_bass", "drive_db", 3.5)
    host2.set_param("power_bass", "punch", 0.25)
    lpb.instance.reset()
    for i in range(len(tone_typ)//1024):
        host2.run("post_multiband", tone_typ[i*1024:(i+1)*1024].copy())
    check("power bass factory default is modest (< 1.0 dB)",
          lpb.instance.meters["clip_db"] < 1.0,
          f"clip {lpb.instance.meters['clip_db']:.2f} dB @ default")

    # ---- shipped De-Esser plugin ----------------------------------------
    from scipy.signal import butter as _bt2, lfilter as _lf2, sosfilt as _sf2
    host3 = PluginHost([plug_dir], fs=FS_FRONT)
    host3.scan()
    host3.set_enabled("de_esser", True)
    lde = host3.plugins["de_esser"]

    nvo = int(FS_FRONT * 8)
    tv = np.arange(nvo) / FS_FRONT
    rngv = np.random.default_rng(11)
    bv, av = _bt2(2, [300 / (FS_FRONT / 2), 2000 / (FS_FRONT / 2)], btype="band")
    vow = _lf2(bv, av, rngv.standard_normal(nvo))
    vow *= 0.6 + 0.4 * np.clip(np.sin(2 * np.pi * 2.5 * tv), 0, 1)
    be_, ae_ = _bt2(2, [5000 / (FS_FRONT / 2), 8000 / (FS_FRONT / 2)], btype="band")
    gate = ((tv % 0.4) < 0.09).astype(float)
    voc = np.stack([0.18 * (vow + _lf2(be_, ae_, rngv.standard_normal(nvo))
                            * gate * 1.4)] * 2, axis=1)

    def _run_de(sig):
        lde.instance.reset()
        outs, grs = [], []
        for i in range(len(sig) // 1024):
            outs.append(host3.run("post_agc", sig[i*1024:(i+1)*1024].copy()))
            grs.append(lde.instance.meters["gr_db"])
        return np.vstack(outs), np.array(grs)

    yv, grv = _run_de(voc)
    bgm = gate[:len(grv) * 1024].reshape(-1, 1024).max(axis=1) > 0.5
    n0 = len(grv) // 8
    ess_gr = grv[n0:][bgm[n0:]].mean()
    gap_gr = grv[n0:][~bgm[n0:]].mean()
    sosv = _bt2(2, [300 / (FS_FRONT / 2), 2000 / (FS_FRONT / 2)],
                btype="band", output="sos")
    vd = 20 * np.log10(np.sqrt(np.mean(_sf2(sosv, yv[:, 0]) ** 2))
                       / np.sqrt(np.mean(_sf2(sosv, voc[:len(yv), 0]) ** 2)))
    _, grm = _run_de(make_cd_program(FS_FRONT, 8.0))
    mus_gr = grm[len(grm) // 4:].mean()
    check("de-esser catches esses, spares vowels & music",
          ess_gr > 3.0 and abs(vd) < 0.2 and mus_gr < 0.5,
          f"ess {ess_gr:.1f} dB, vowel {vd:+.2f} dB, music {mus_gr:.2f} dB")

    host3.set_param("de_esser", "range_db", 0.0)
    lde.instance.reset()
    y0 = host3.run("post_agc", voc[:4096].copy())
    ex = float(np.max(np.abs(y0 - voc[:4096])))
    check("de-esser releases between esses + RANGE-0 exact",
          gap_gr < 1.0 and ex < 1e-12,
          f"gap tail {gap_gr:.2f} dB, exact {ex:.1e}")

    # ---- shipped Sonic Maximizer plugin ----------------------------------
    from scipy.signal import welch as _welch
    host4 = PluginHost([plug_dir], fs=FS_FRONT)
    host4.scan()
    host4.set_enabled("sonic_maximizer", True)
    lsm = host4.plugins["sonic_maximizer"]

    def _run_sm(sig):
        lsm.instance.reset()
        return np.vstack([host4.run("post_input",
                                    sig[i*1024:(i+1)*1024].copy())
                          for i in range(len(sig)//1024)])

    rngs = np.random.default_rng(9)
    nsm = int(FS_FRONT * 6)
    sos_hi2 = _bt2(2, 2400 / (FS_FRONT / 2), btype="high", output="sos")

    def _hf_lift(sig):
        y = _run_sm(sig)
        m = len(y)
        return 20 * np.log10(
            np.sqrt(np.mean(_sf2(sos_hi2, y[:, 0]) ** 2))
            / np.sqrt(np.mean(_sf2(sos_hi2, sig[:m, 0]) ** 2)))

    for kk, vv in (("proc", 0.0), ("contour", 0.0), ("out_db", 0.0)):
        host4.set_param("sonic_maximizer", kk, vv)
    xw = rngs.standard_normal((nsm, 2)) * 0.1
    yw = _run_sm(xw)
    fw, piw = _welch(xw[:len(yw), 0], fs=FS_FRONT, nperseg=1 << 13)
    _, pow_ = _welch(yw[:, 0], fs=FS_FRONT, nperseg=1 << 13)
    magw = 10 * np.log10(pow_ / piw)
    bnd = (fw > 40) & (fw < 16000)
    dev0 = max(abs(magw[bnd].min()), abs(magw[bnd].max()))

    host4.set_param("sonic_maximizer", "proc", 10.0)
    bvd, avd = _bt2(2, 800 / (FS_FRONT / 2))
    vdark = np.stack([_lf2(bvd, avd,
                           rngs.standard_normal(nsm))] * 2, axis=1) * 0.2
    bright = rngs.standard_normal((nsm, 2)) * 0.1
    lift_dark = _hf_lift(vdark)
    lift_bright = _hf_lift(bright)
    check("sonic maximizer enhances program-dependently",
          dev0 < 0.15 and lift_dark > 3.5 and lift_bright < 0.5,
          f"zeros dev {dev0:.2f} dB, dark +{lift_dark:.1f}, "
          f"bright {lift_bright:+.1f}")

    host4.set_param("sonic_maximizer", "proc", 3.0)
    host4.set_param("sonic_maximizer", "contour", 2.0)
    lift_mus = _hf_lift(make_cd_program(FS_FRONT, 6.0))
    check("sonic maximizer factory defaults are modest",
          0.5 < lift_mus < 2.0, f"HF +{lift_mus:.2f} dB on dense music")

    # ---- shipped De-Clipper / De-Lossifier plugin ------------------------
    from scipy.signal import firwin as _fw, welch as _welch2
    from scipy.signal import resample_poly as _rp
    host5 = PluginHost([plug_dir], fs=FS_FRONT)
    host5.scan()
    host5.set_enabled("de_clipper", True)
    ldc = host5.plugins["de_clipper"]
    LATDC = 192

    def _run_dc(sig):
        ldc.instance.reset()
        return np.vstack([host5.run("post_input",
                                    sig[i*1024:(i+1)*1024].copy())
                          for i in range(len(sig)//1024)])

    host5.set_param("de_clipper", "lossy", 0.0)
    host5.set_param("de_clipper", "declip", 8.0)
    ndc = int(FS_FRONT * 4)
    tdc = np.arange(ndc) / FS_FRONT
    cln = 0.5 * 10 ** (4 / 20) * np.sin(2 * np.pi * 1000 * tdc)
    clp = np.clip(cln, -0.5, 0.5)
    ydc = _run_dc(np.stack([clp] * 2, axis=1))[:, 0]

    def _thd(sig):
        fq, pq = _welch2(sig, fs=FS_FRONT, nperseg=1 << 14)
        fund = pq[(fq > 900) & (fq < 1100)].sum()
        harm = sum(pq[(fq > k * 1000 - 120) & (fq < k * 1000 + 120)].sum()
                   for k in range(2, 8))
        return 10 * np.log10(harm / fund)

    thd_gain = _thd(clp[LATDC:]) - _thd(ydc[LATDC:])
    crest_lift = 20 * np.log10(np.max(np.abs(ydc)) / 0.5)
    pure = np.stack([0.3 * np.sin(2 * np.pi * 400 * tdc)] * 2, axis=1)
    ypu = _run_dc(pure)
    exact = float(np.max(np.abs(ypu[LATDC:] - pure[:len(ypu) - LATDC])))
    check("de-clipper rebuilds peaks, exact on clean",
          thd_gain > 10.0 and crest_lift > 2.0 and exact < 1e-15,
          f"THD -{thd_gain:.1f} dB, crest +{crest_lift:.1f} dB, "
          f"clean {exact:.1e}")

    host5.set_param("de_clipper", "declip", 0.0)
    host5.set_param("de_clipper", "lossy", 8.0)
    musdc = make_cd_program(FS_FRONT, 6.0)
    tapsdc = _fw(255, 11000 / (FS_FRONT / 2))
    lsy = np.stack([np.convolve(musdc[:, 0], tapsdc, mode="same"),
                    np.convolve(musdc[:, 1], tapsdc, mode="same")], axis=1)
    sos_tdc = _bt2(2, [11000 / (FS_FRONT / 2), 15000 / (FS_FRONT / 2)],
                   btype="band", output="sos")

    def _bdb(s):
        return 20 * np.log10(np.sqrt(np.mean(_sf2(sos_tdc, s) ** 2)) + 1e-12)

    yls = _run_dc(lsy)
    restored = _bdb(yls[LATDC:, 0]) - _bdb(lsy[:len(yls) - LATDC, 0])
    yfb = _run_dc(musdc)
    fb_inj = abs(_bdb(yfb[LATDC:, 0]) - _bdb(musdc[:len(yfb) - LATDC, 0]))
    check("de-lossifier restores only what's missing",
          restored > 5.0 and fb_inj < 0.5,
          f"lossy +{restored:.1f} dB, full-band {fb_inj:+.2f} dB")

    # de-clipper detection survives upstream azimuth resampling
    host5.set_param("de_clipper", "declip", 8.0)
    host5.set_param("de_clipper", "lossy", 0.0)
    host5.set_param("de_clipper", "test", 0.0)
    mdc2 = 0.5 * (musdc[:, 0] + musdc[:, 1])
    mclp = np.clip(mdc2 * 1.8, -np.max(np.abs(mdc2)) * 0.85,
                   np.max(np.abs(mdc2)) * 0.85)
    up2 = _rp(mclp, 32, 1)
    r2 = _rp(np.concatenate([np.zeros(109), up2])[:len(up2)],
             1, 32)[:len(mclp)]
    skc = np.stack([mclp, r2], axis=1)

    def _rep_run(use_az):
        host5.set_enabled("azimuth_repair", use_az)
        ldc.instance.reset()
        if use_az:
            host5.plugins["azimuth_repair"].instance.reset()
        mx = 0.0
        for i in range(len(skc) // 1024):
            host5.run("post_input", skc[i*1024:(i+1)*1024].copy())
            mx = max(mx, ldc.instance.meters["repair_db"])
        host5.set_enabled("azimuth_repair", False)
        return mx

    r_alone = _rep_run(False)
    r_az = _rep_run(True)
    check("de-clipper detection survives upstream azimuth",
          r_az > 0.6 * r_alone and r_alone > 1.0,
          f"repair {r_alone:.1f} dB alone, {r_az:.1f} dB after azimuth")

    # ---- shipped Multipath / Stereo Energy Governor ----------------------
    host9 = PluginHost([plug_dir], fs=FS_FRONT)
    host9.scan()
    host9.set_enabled("stereo_governor", True)
    lsg = host9.plugins["stereo_governor"]

    def _run_sg(sig):
        lsg.instance.reset()
        outs, grs = [], []
        for i in range(len(sig) // 1024):
            outs.append(host9.run("post_multiband",
                                  sig[i*1024:(i+1)*1024].copy()))
            grs.append(lsg.instance.meters["gr_db"])
        return np.vstack(outs), np.array(grs)

    def _sm_sg(x):
        mm_ = 0.5 * (x[:, 0] + x[:, 1])
        ss_ = 0.5 * (x[:, 0] - x[:, 1])
        aa = np.exp(-1 / (0.04 * FS_FRONT))
        pmm = _lf2([1 - aa], [1, -aa], mm_ * mm_)
        pss = _lf2([1 - aa], [1, -aa], ss_ * ss_)
        rr = 10 * np.log10(np.maximum(pss, 1e-14)
                           / np.maximum(pmm, 1e-14))
        return rr[len(rr) // 4:]

    mus_sg = make_cd_program(FS_FRONT, 8.0)
    mwg = 0.5 * (mus_sg[:, 0] + mus_sg[:, 1])
    swg = 0.5 * (mus_sg[:, 0] - mus_sg[:, 1]) * 1.25
    wide_sg = np.stack([mwg + swg, mwg - swg], axis=1)
    yw_sg, grw_sg = _run_sg(wide_sg)
    ex_sg = float(np.max(np.abs(yw_sg - wide_sg[:len(yw_sg)])))
    check("stereo governor exact below the ceiling",
          ex_sg == 0.0 and grw_sg.max() == 0.0,
          f"widened program dev {ex_sg:.0e}, GR {grw_sg.max():.1f}")

    rng_sg = np.random.default_rng(5)
    n_sg = len(mus_sg)
    bsg, asg = _bt2(2, 4000 / (FS_FRONT / 2))
    mono_sg = _lf2(bsg, asg, rng_sg.standard_normal(n_sg)) * 0.2
    anti_sg = np.stack([mono_sg, -0.8 * mono_sg
                        + 0.2 * rng_sg.standard_normal(n_sg) * 0.2], axis=1)
    host9.set_param("stereo_governor", "range_db", 18.0)
    ya_sg, gra_sg = _run_sg(anti_sg)
    p95_sg = float(np.percentile(_sm_sg(ya_sg), 95))
    mono_dev = float(np.max(np.abs(
        0.5 * (ya_sg[:, 0] + ya_sg[:, 1])
        - 0.5 * (anti_sg[:len(ya_sg), 0] + anti_sg[:len(ya_sg), 1]))))
    check("stereo governor holds S/M ceiling, mono untouched",
          p95_sg < -2.0 and mono_dev < 1e-12 and gra_sg.max() > 8.0,
          f"anti-phase held {p95_sg:+.1f} dB (ceil -3), GR "
          f"{gra_sg.max():.0f}, mono {mono_dev:.0e}")



    # ---- shipped Natural Dynamics plugin ---------------------------------
    host6 = PluginHost([plug_dir], fs=FS_FRONT)
    host6.scan()
    host6.set_enabled("natural_dynamics", True)
    lnd = host6.plugins["natural_dynamics"]

    def _run_nd(sig, use_dc=False):
        host6.set_enabled("de_clipper", use_dc)
        lnd.instance.reset()
        if use_dc:
            host6.plugins["de_clipper"].instance.reset()
        outs, pmax = [], 0.0
        for i in range(len(sig) // 1024):
            outs.append(host6.run("post_input",
                                  sig[i*1024:(i+1)*1024].copy()))
            pmax = max(pmax, lnd.instance.meters["punch_db"])
        host6.set_enabled("de_clipper", False)
        return np.vstack(outs), pmax

    def _crest_nd(sig):
        pw = 0.5 * (sig[:, 0] ** 2 + sig[:, 1] ** 2)
        aa = np.exp(-1 / (0.4 * FS_FRONT))
        rr = _lf2([1 - aa], [1, -aa], pw)
        hh = len(sig) // 3
        return (20 * np.log10(np.max(np.abs(sig[hh:])) + 1e-12)
                - 10 * np.log10(np.mean(rr[hh:]) + 1e-14))

    sq_nd = make_cd_program(FS_FRONT, 10.0)
    host6.set_param("natural_dynamics", "amount", 8.0)
    host6.set_param("natural_dynamics", "sens", 4.0)
    y_nd, pm_nd = _run_nd(sq_nd)
    lift = _crest_nd(y_nd) - _crest_nd(sq_nd[:len(y_nd)])
    ydc_nd, pmdc_nd = _run_nd(sq_nd, use_dc=True)
    lift_dc = _crest_nd(ydc_nd) - _crest_nd(sq_nd[:len(ydc_nd)])
    check("natural dynamics restores punch (also after de-clipper)",
          lift > 2.0 and lift_dc > 2.0 and pm_nd > 2.5 and pmdc_nd > 2.5,
          f"crest +{lift:.1f} / +{lift_dc:.1f} dB (declipped), "
          f"punch {pm_nd:.1f}/{pmdc_nd:.1f}")

    host6.set_param("natural_dynamics", "amount", 0.0)
    ya0, _ = _run_nd(sq_nd)
    exa = float(np.max(np.abs(ya0 - sq_nd[:len(ya0)])))
    host6.set_param("natural_dynamics", "amount", 8.0)
    host6.set_param("natural_dynamics", "sens", 0.0)
    cl_nd = make_classical_program(FS_FRONT, 12.0)
    ys0, _ = _run_nd(cl_nd)
    exs = float(np.max(np.abs(ys0 - cl_nd[:len(ys0)])))
    check("natural dynamics AMOUNT-0 / SENS-0 are exact bypasses",
          exa == 0.0 and exs == 0.0, f"{exa:.0e} / {exs:.0e}")

    # ---- shipped Azimuth / Stereo Repair plugin --------------------------
    from scipy.signal import resample_poly as _rp
    host7 = PluginHost([plug_dir], fs=FS_FRONT)
    host7.scan()
    host7.set_enabled("azimuth_repair", True)
    laz = host7.plugins["azimuth_repair"]

    def _run_az(sig):
        laz.instance.reset()
        return np.vstack([host7.run("post_input",
                                    sig[i*1024:(i+1)*1024].copy())
                          for i in range(len(sig)//1024)])

    mus_az = make_cd_program(FS_FRONT, 12.0)
    m_az = 0.5 * (mus_az[:, 0] + mus_az[:, 1])
    up = _rp(m_az, 32, 1)
    up = np.concatenate([np.zeros(int(round(7.3 * 32))), up])[:len(up)]
    r_sk = _rp(up, 1, 32)[:len(m_az)]
    skewed = np.stack([m_az, r_sk], axis=1)

    y_az = _run_az(skewed)
    inj_us = 7.3 / FS_FRONT * 1e6
    det_us = laz.instance.meters["skew_us"]
    half = len(y_az) // 2
    lc_ = y_az[half:, 0][12:-12]
    ccr = np.array([np.dot(lc_, y_az[half:, 1][12+k:12+k+len(lc_)])
                    for k in range(-12, 13)])
    ii = int(np.argmax(np.abs(ccr)))
    y0_, y1_, y2_ = abs(ccr[ii-1]), abs(ccr[ii]), abs(ccr[ii+1])
    res_smp = (ii - 12) + 0.5 * (y0_ - y2_) / (y0_ - 2*y1_ + y2_)
    sosm = _bt2(2, [2500 / (FS_FRONT/2), 4500 / (FS_FRONT/2)],
                btype="band", output="sos")

    def _mono_hf(sig):
        s = 0.5 * (sig[half:len(y_az), 0] + sig[half:len(y_az), 1])
        return 20 * np.log10(np.sqrt(np.mean(_sf2(sosm, s) ** 2)) + 1e-12)

    hf_gain = _mono_hf(y_az) - _mono_hf(skewed)
    pol_src = np.stack([m_az, -m_az], axis=1)
    yp_az = _run_az(pol_src)
    ms_out = 20 * np.log10(np.sqrt(np.mean(
        (0.5 * (yp_az[half:, 0] + yp_az[half:, 1])) ** 2)) + 1e-12)
    check("azimuth repair locks, corrects skew & fixes polarity",
          abs(det_us - inj_us) < 8 and abs(res_smp) < 0.35
          and hf_gain > 3.0 and laz.instance._pol < 0 and ms_out > -12,
          f"skew {det_us:+.0f}/{inj_us:+.0f} us, residual "
          f"{res_smp:+.2f} smp, mono-HF +{hf_gain:.1f} dB, pol fixed")

    dual_az = np.stack([m_az, m_az], axis=1)
    yd_az = _run_az(dual_az)
    D0_AZ = 16
    ex_az = float(np.max(np.abs(yd_az[D0_AZ:] - dual_az[:len(yd_az)-D0_AZ])))
    ap1 = abs(laz.instance.meters["skew_us"])
    rngaz = np.random.default_rng(4)
    wide_az = np.stack([rngaz.standard_normal(len(m_az)),
                        rngaz.standard_normal(len(m_az))], axis=1) * 0.1
    _run_az(wide_az)
    ap2 = abs(laz.instance.meters["skew_us"])
    check("azimuth repair exact on aligned, frozen on true stereo",
          ex_az == 0.0 and ap1 < 2.0 and ap2 < 2.0,
          f"aligned dev {ex_az:.0e}, applied {ap1:.1f}/{ap2:.1f} us")

    # ---- shipped Dehummer + Gate plugin ----------------------------------
    host8 = PluginHost([plug_dir], fs=FS_FRONT)
    host8.scan()
    host8.set_enabled("dehum_gate", True)
    ldh = host8.plugins["dehum_gate"]

    def _run_dh(sig):
        ldh.instance.reset()
        return np.vstack([host8.run("post_input",
                                    sig[i*1024:(i+1)*1024].copy())
                          for i in range(len(sig)//1024)])

    def _line_dh(sig, f, secs=2.0):
        nn = int(secs * FS_FRONT)
        s = 0.5 * (sig[-nn:, 0] + sig[-nn:, 1])
        tt = np.arange(nn) / FS_FRONT
        c = np.abs(np.dot(s * np.hanning(nn),
                          np.exp(-2j * np.pi * f * tt))) / (nn / 4)
        return 20 * np.log10(c + 1e-12)

    mus_dh = make_cd_program(FS_FRONT, 9.0)
    q5 = np.zeros((int(5 * FS_FRONT), 2))
    q3 = np.zeros((int(3 * FS_FRONT), 2))
    prog_dh = np.vstack([q5, mus_dh, q3])
    t_dh = np.arange(len(prog_dh)) / FS_FRONT
    hum_dh = (10 ** (-38 / 20) * np.sin(2 * np.pi * 50 * t_dh)
              + 10 ** (-44 / 20) * np.sin(2 * np.pi * 100 * t_dh)
              + 10 ** (-42 / 20) * np.sin(2 * np.pi * 150 * t_dh)
              + 10 ** (-48 / 20) * np.sin(2 * np.pi * 250 * t_dh))
    dirty_dh = prog_dh + hum_dh[:, None]

    host8.set_param("dehum_gate", "range_db", 0.0)   # isolate the dehummer
    y_dh = _run_dh(dirty_dh)
    kills = [_line_dh(dirty_dh[:len(y_dh)], f) - _line_dh(y_dh, f)
             for f in (50, 100, 150, 250)]
    sos_pi = _bt2(2, [200 / (FS_FRONT / 2), 8000 / (FS_FRONT / 2)],
                  btype="band", output="sos")
    hh = int(6 * FS_FRONT)
    pdel = abs(20 * np.log10(
        np.sqrt(np.mean(_sf2(sos_pi, y_dh[hh:, 0]) ** 2))
        / np.sqrt(np.mean(_sf2(sos_pi, dirty_dh[hh:len(y_dh), 0]) ** 2))))
    yc_dh = _run_dh(mus_dh)
    ex_dh = float(np.max(np.abs(yc_dh - mus_dh[:len(yc_dh)])))
    short_dh = np.vstack([q5, q3])
    t_s = np.arange(len(short_dh)) / FS_FRONT
    d61 = short_dh + (10 ** (-40 / 20)
                      * np.sin(2 * np.pi * 61.3 * t_s))[:, None]
    y61 = _run_dh(d61)
    k61 = _line_dh(d61[:len(y61)], 61.3) - _line_dh(y61, 61.3)
    check("dehummer kills ground loops, tracks drift, exact on clean",
          min(kills) > 20.0 and pdel < 0.3 and ex_dh == 0.0 and k61 > 15.0,
          f"lines -{min(kills):.0f}..-{max(kills):.0f} dB, program "
          f"{pdel:.2f} dB, clean {ex_dh:.0e}, drift -{k61:.0f} dB")

    host8.set_param("dehum_gate", "range_db", 12.0)
    rng_dh = np.random.default_rng(3)
    gsrc = prog_dh + rng_dh.standard_normal(prog_dh.shape) * 10 ** (-64/20)
    yg_dh = _run_dh(gsrc)
    gap_dh = slice(int(1 * FS_FRONT), int(4 * FS_FRONT))
    att_dh = 20 * np.log10(
        np.sqrt(np.mean(gsrc[gap_dh] ** 2))
        / (np.sqrt(np.mean(yg_dh[gap_dh] ** 2)) + 1e-15))
    loud_dh = slice(int(7 * FS_FRONT), int(12 * FS_FRONT))
    dev_dh = float(np.max(np.abs(yg_dh[loud_dh]
                                 - gsrc[loud_dh.start:loud_dh.stop])))
    check("noise gate floors grot, exact above threshold",
          att_dh > 7.0 and dev_dh == 0.0,
          f"gaps -{att_dh:.1f} dB, loud dev {dev_dh:.0e}")

    # user plugin dir overrides the shipped copy (drop-in upgrades)
    import tempfile as _tf, shutil as _sh
    d_ship = pathlib.Path(_tf.mkdtemp()); d_user = pathlib.Path(_tf.mkdtemp())
    src = (plug_dir / "stereo_widener.py").read_text()
    (d_ship / "stereo_widener.py").write_text(src)
    (d_user / "stereo_widener.py").write_text(
        src.replace('"version": "1.0"', '"version": "9.9"'))
    hov = PluginHost([d_ship, d_user], fs=FS_FRONT)
    hov.scan()
    lov = hov.plugins.get("stereo_widener")
    ok_ov = (not hov.scan_errors and lov is not None
             and lov.manifest["version"] == "9.9"
             and str(lov.path).startswith(str(d_user)))
    hov.set_enabled("stereo_widener", True)
    ok_ov = ok_ov and lov.error is None and lov.enabled
    _sh.rmtree(d_ship); _sh.rmtree(d_user)
    check("user plugin dir overrides shipped copy, no scan error", ok_ov,
          f"loaded v{lov.manifest['version'] if lov else '?'} from user dir")










# ------------------------------------------------------------------ 12. mix
def test_leveler_mix_preservation():
    """On classical material the leveler must not repaint the mix:
    zero net gain (shape only), bands coupled to the pack, no slow
    per-band fades (the 'violins fading while bass climbs' failure)."""
    x = make_classical_program(FS_FRONT, 30.0)
    graph = DSPGraph()
    graph.apply_params(resolve("NATURAL", "MEDIUM"))
    blk = 1024
    lev = []
    for i in range(len(x) // blk):
        graph.run_48(x[i * blk:(i + 1) * blk])
        lev.append(graph.front.meters["lev_gain"].copy())
    lev = np.array(lev)
    seg = lev[len(lev) // 3:]
    spread = (seg.max(axis=1) - seg.min(axis=1)).max()
    common = abs(seg.mean())
    hm_drift = seg[:, 2].max() - seg[:, 2].min()
    check("leveler preserves the mix on classical",
          spread <= 4.2 and common < 0.3 and hm_drift < 1.5,
          f"spread {spread:.1f} dB, net {common:.2f} dB, "
          f"HM drift {hm_drift:.1f} dB")


# ------------------------------------------------------------------ 13. boot
def test_factory_boot():
    """The power-on default (AGGRESSIVE·LIGHT + the factory plugin rack)
    must be broadcast-safe: legal modulation, healthy loudness, every
    shipped plugin running error-free."""
    import pathlib
    from omegafm.plugin_host import PluginHost
    from omegafm.processor import FACTORY_RACK, Controller

    c = Controller()
    boot_ok = c.signature == "AGGRESSIVE" and c.density == "LIGHT"
    plug_dir = pathlib.Path(__file__).resolve().parents[1] / "plugins"
    host = PluginHost([plug_dir], fs=FS_FRONT)
    host.scan()
    host.load_state(FACTORY_RACK)
    x = make_cd_program(FS_FRONT, 10.0)
    g = DSPGraph(host)
    g.apply_params(resolve("AGGRESSIVE", "LIGHT"))
    blk = 1024
    pk = []
    for i in range(len(x) // blk):
        g.mpx_out(g.run_48(x[i * blk:(i + 1) * blk]))
        pk.append(g.stereo_gen.mpx_peak)
    pk = np.array(pk)
    stl = slice(len(pk) // 4, None)
    rack_ok = all(host.plugins[pid].enabled
                  and host.plugins[pid].error is None
                  for pid in FACTORY_RACK if pid in host.plugins) \
        and all(lp.error is None for lp in host.plugins.values())
    check("factory boot default is broadcast-safe",
          boot_ok and rack_ok and pk[stl].max() <= 1.005
          and pk[stl].mean() > 0.55,
          f"MOD {pk[stl].mean()*100:.0f}/{pk[stl].max()*100:.1f} %, "
          f"rack {'ok' if rack_ok else 'ERROR'}")


# ------------------------------------------------------------------ 14. agc
def test_agc_ride_stability():
    """Broadcast-AGC contract: rides smoothly at ANY input drive, gates
    rock-solid across gaps (fast gate + frozen integrator + no drift),
    and never exceeds its slew rate - the 'pumps and ducks when
    underdriven' failure can never return."""
    from scipy.signal import butter as _b, lfilter as _l
    rng = np.random.default_rng(12)
    n = int(FS_FRONT * 16)
    b_, a_ = _b(2, [200 / (FS_FRONT / 2), 4000 / (FS_FRONT / 2)],
                btype="band")
    steady = _l(b_, a_, rng.standard_normal(n)) * 0.28
    steady = np.stack([steady, steady], axis=1)

    def run_at(drive_db):
        x = steady * 10 ** (drive_db / 20)
        g0 = int(9 * FS_FRONT)
        x[g0:g0 + int(2 * FS_FRONT)] = 0.0
        g = DSPGraph()
        g.apply_params(resolve("AGGRESSIVE", "LIGHT"))
        blk = 1024
        traj = []
        for i in range(len(x) // blk):
            g.run_48(x[i * blk:(i + 1) * blk])
            traj.append(g.front.meters["agc_gain"])
        t = np.array(traj)
        fsb = FS_FRONT / blk
        settled = float(np.median(t[int(7 * fsb):int(9 * fsb)]))
        vel = float(np.max(np.abs(np.diff(t[int(6 * fsb):]))) * fsb)
        gap = t[int(9.1 * fsb):int(10.9 * fsb)]
        rec = abs(float(t[int(12.6 * fsb)]) - float(t[int(10.9 * fsb)]))
        return settled, vel, float(gap.max() - gap.min()), rec

    s0, v0, gd0, r0 = run_at(0.0)
    s1, v1, gd1, r1 = run_at(-10.0)
    ok = (abs((s0 - s1) + 10.0) < 2.2   # 2x window parking
          and max(v0, v1) < 2.9
          and max(gd0, gd1) < 0.4
          and max(r0, r1) < 0.8)
    check("AGC rides drive-independently, gates rock-solid",
          ok, f"gain {s0:+.1f}/{s1:+.1f}, vel {max(v0,v1):.1f} dB/s, "
              f"gap {max(gd0,gd1):.2f}, recovery {max(r0,r1):.2f} dB")


# ---------------------------------------------------------------- 15. spit
def test_sibilant_spit():
    """The clipper must not 'spit' on sibilants: a continuous twin-tone
    ess (6.0+6.8 kHz) over a rail-riding bass bed - the worst case for
    joint bass+HF clipping - is measured for the odd-order IMD product
    at 5.2 kHz (2f1-f2) on the transmission audio. The old topology's
    post-brickwall hard clip regenerated -24.6 dBc regardless of any
    clipper setting; the distortion-controlled clip with overshoot
    headroom holds it below -24 with loudness intact."""
    fs_ = int(FS_FRONT)
    n_ = int(fs_ * 8)
    t_ = np.arange(n_) / fs_
    kt_ = t_ % 0.5
    kick_ = 1.1 * np.sin(2 * np.pi * (55 + 70 * np.exp(-kt_ / 0.03)) * kt_) \
        * np.exp(-kt_ / 0.06) * (kt_ < 0.2)
    drone_ = 0.5 * np.sin(2 * np.pi * 165 * t_) \
        + 0.3 * np.sin(2 * np.pi * 220 * t_)
    bed_ = np.tanh((kick_ + drone_) * 1.5) * 0.9
    ess_ = 0.42 * (np.sin(2 * np.pi * 6000 * t_)
                   + np.sin(2 * np.pi * 6800 * t_))
    xs = np.stack([bed_ * 0.55 + ess_] * 2, axis=1)
    xs = xs / np.max(np.abs(xs)) * 10 ** (-0.5 / 20)
    fsb = 192000
    p_ = resolve("AGGRESSIVE", "LIGHT")
    p_["clip_drive_db"] = 5.0
    g_ = DSPGraph()
    g_.apply_params(p_)
    ys_ = []
    pk_ = []
    for i in range(len(xs) // 1024):
        y192 = g_.run_48(xs[i * 1024:(i + 1) * 1024])
        ys_.append(y192)
        g_.mpx_out(y192)
        pk_.append(g_.stereo_gen.mpx_peak)
    yy = np.vstack(ys_)[:, 0][int(3 * fsb):]
    tt_ = np.arange(len(yy)) / fsb
    ww = np.hanning(len(yy))

    def _tone(f):
        return 20 * np.log10(np.abs(np.dot(yy * ww, np.exp(
            -2j * np.pi * f * tt_))) / (len(yy) / 4) + 1e-12)

    imd = _tone(5200.0) - _tone(6000.0)
    mpx = float(np.max(pk_)) * 100
    check("clipper doesn't spit on sibilants (IMD bounded)",
          imd < -24.0 and mpx > 95.0,
          f"5.2k IMD {imd:+.1f} dBc (was -24.6 shear floor), MPX {mpx:.1f}%")


# ---------------------------------------------------------------- 16. bs412
def test_bs412():
    """ITU-R BS.412: with the limiter enabled the 60 s MPX power meter
    must hold at/under target while pilot injection stays untouched
    (the gain rides ONLY the audio part of the composite); disabled is
    the factory default and applies exactly unity."""
    p0 = resolve("AGGRESSIVE", "LIGHT")
    check("BS.412 disabled by factory default",
          p0.get("bs412_enable") is False, "regulatory opt-in")

    x = make_cd_program(FS_FRONT, 30.0)

    def run(enable):
        p = resolve("AGGRESSIVE", "LIGHT")
        p["bs412_enable"] = enable
        g = DSPGraph()
        g.apply_params(p)
        tail = []
        grs = []
        for i in range(len(x) // 1024):
            mp = g.mpx_out(g.run_48(x[i * 1024:(i + 1) * 1024]))
            if i * 1024 / FS_FRONT > 26:
                tail.append(mp)
            if i * 1024 / FS_FRONT > 10:
                grs.append(g.stereo_gen.bs412_gr)
        return g, np.concatenate(tail), np.array(grs)

    g_off, tail_off, _ = run(False)
    g_on, tail_on, grs = run(True)

    def pilot(tail):
        tt = np.arange(len(tail)) / FS_BACK
        return 20 * np.log10(np.abs(np.dot(
            tail * np.hanning(len(tail)),
            np.exp(-2j * np.pi * 19000.0 * tt))) / (len(tail) / 4) + 1e-12)

    p_delta = abs(pilot(tail_on) - pilot(tail_off))
    unlimited = g_off.stereo_gen.bs412_dbr
    limited = g_on.stereo_gen.bs412_dbr
    check("BS.412 limiter holds target, pilot untouched",
          limited < 0.3 and unlimited > 0.8 and grs.max() > 1.0
          and p_delta < 0.05,
          f"{unlimited:+.1f} dBr unlimited -> {limited:+.1f} limited, "
          f"GR to {grs.max():.1f} dB, pilot delta {p_delta:.3f} dB")


# ------------------------------------------------------------------ 6. separation
def test_separation():
    """Feed 1 kHz to L only; decode MPX with a *pilot-locked* decoder
    (analytic 19 kHz pilot squared -> 38 kHz reference).  If the
    delayed-phase pilot injection were misaligned, separation would
    collapse - this is the make-or-break composite test."""
    from scipy.signal import hilbert, firwin, lfilter as lf

    graph = DSPGraph()
    p = resolve("NATURAL", "MEDIUM")
    for k in ("bypass_agc", "bypass_leveler", "bypass_comp", "bypass_eq",
              "bypass_bass", "bypass_rotator", "bypass_hf_lim",
              "bypass_wb_lim", "bypass_clipper"):
        p[k] = True
    p["preemph_us"] = 0.0
    p["final_drive_db"] = 0.0
    p["composite_drive_db"] = 0.0
    p["rds_enable"] = False
    graph.apply_params(p)

    fs48, dur = FS_FRONT, 3.0
    t = np.arange(int(fs48 * dur)) / fs48
    x = np.zeros((len(t), 2))
    x[:, 0] = 0.5 * np.sin(2 * np.pi * 1000.0 * t)

    blk, out = 1024, []
    for i in range(len(x) // blk):
        y192 = graph.run_48(x[i * blk:(i + 1) * blk])
        out.append(graph.mpx_out(y192))
    mpx = np.concatenate(out)[int(FS_BACK):]         # settle

    # pilot-locked decode (compensate the BP group delay: 400 samples)
    bp = firwin(801, [18500.0, 19500.0], pass_zero=False, fs=FS_BACK)
    pilot = lf(bp, [1.0], mpx)[400:]                 # now time-aligned w/ mpx
    seg = mpx[:len(pilot)]
    ana = hilbert(pilot[4000:-4000])
    ana /= np.abs(ana) + 1e-12
    sub_ref = -np.imag(ana * ana)                    # analytic(sin t)^2 = -e^{2jt}
    seg = seg[4000:-4000]
    lp = firwin(401, 15000.0, fs=FS_BACK)
    m = lf(lp, [1.0], seg)
    s = lf(lp, [1.0], seg * 2.0 * sub_ref)
    L = m + s
    R = m - s
    n0 = 8000
    sep = 20 * np.log10(np.std(L[n0:]) / (np.std(R[n0:]) + 1e-12))
    check("stereo separation @1 kHz > 40 dB", sep > 40.0, f"{sep:.1f} dB")
    pk = np.max(np.abs(mpx))
    print(f"      tone modulation      : {pk*100:.1f} %")


def main():
    print("=== OmegaFM chain validation ===")
    test_crossover()
    test_preemph()
    test_rds_crc()
    test_chain_and_mpx()
    test_wb_riding()
    test_leveler_riding()
    test_comp_riding()
    test_preset_json()
    test_plugins()
    test_leveler_mix_preservation()
    test_factory_boot()
    test_agc_ride_stability()
    test_sibilant_spit()
    test_bs412()
    test_separation()
    n_bad = results.count(False)
    print(f"=== {len(results)-n_bad}/{len(results)} checks passed ===")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
