# OmegaFM · Native Edition (Windows 10/11 x64)

A native, compiled port of the OmegaFM web-edition FM broadcast processor with a
**bit-exact DSP core** — the math is identical, sample for sample, to the web
app running in Chrome — plus WASAPI live audio, an MPX/RDS composite path, and
a multi-threaded file/batch renderer. Your **AGGRESSIVE · MEDIUM** preset
(`aggressive_medium3.json`, including all 12 rack plugins) is baked in as the
power-on default.

One file, no install: `omegafm.exe` (fully static, ~3 MB).

---

## Bit-exactness — what was done and how it was proven

* Every DSP class of the web engine (AGC, 4-band leveler, EQ, bass enhancer,
  crossovers, 4-band compressor, HF + look-ahead limiters, distortion-cancelled
  clipper, 15 kHz brickwall, overshoot control, stereo generator, filtered
  composite clipper, BS.412, RDS encoder, and all 12 plugins) was
  transliterated to C++ preserving IEEE-754 double precision and the **exact
  operation order** (built with `-ffp-contract=off`, no fast-math).
* Transcendentals (`exp`, `pow`, `log10`, `sin`, `cos`, `tan`, `tanh`,
  `log1p`, `hypot`, JS `Math.round`) use **V8's own fdlibm sources** (the same
  code Node/Chrome run), verified bit-identical over millions of samples.
* The FIR inner loop is the **same WebAssembly SIMD kernel the browser
  executes**, machine-translated to C (`wasm2c`) and proven bit-identical
  against V8 executing the original `fir.wasm`.
* End-to-end certification: the actual Windows executable was run (under Wine)
  against the untouched web DSP JavaScript running in Node/V8 over multi-second
  program material — silence, tones, clipped flat-tops, noise, skewed stereo,
  sibilance, full-scale — in six configurations (44.1 k / 48 k / 192 k, stereo
  and true-MPX+RDS, reference and mastering quality, with the full default
  rack active). **Every output sample and every meter value matched
  bit-for-bit.**

*Fine print:* `Math.sin/cos` match the fdlibm build of V8 (what Node and
gcc-built engines use). Clang-built Chrome may differ in the last ULP of
sin/cos on some arguments (~ -300 dB); every other function and the whole
chain structure are exact.

## Running the GUI

Double-click `omegafm.exe`. The interface is the web UI, pixel for pixel —
same knobs, meters, menus, Advanced and Plugins panels, preset import/export
(fully interchangeable `.json` presets).

* **AUDIO SETUP** → devices are listed by name automatically (native WASAPI,
  no permission prompts) → pick input/output, sample rate (44.1/48 k, or
  176.4/192 k for true MPX), DSP quality, output buffer → **START**.
* **Output: MPX composite @192k** puts the real composite — pilot, 38 kHz DSB,
  57 kHz RDS — on both output channels for a 192 kHz interface feeding an
  exciter/MPX input. RDS (PS/RT/PI/PTY, dynamic PS, now-playing file watch)
  is in the RDS panel; the file watcher reads `RT=` / `PS=` lines.
* Audio is processed float64 end to end and delivered to WASAPI as float —
  Windows renders it at the device's configured format (set the device to
  24-bit / 192 kHz in Windows Sound settings for full 24/192 output).
* **File → Reset to Factory** returns to the baked-in AGGRESSIVE · MEDIUM
  preset. The five original factory presets remain in the File menu.

## File & batch processing (File → File / Batch Processor…)

Queue individual files or a whole folder (`wav` / `flac` / `mp3`), choose an
output folder, mode (processed stereo, or MPX composite @192 k), WAV depth
(24-bit default), quality, and how many CPU workers to use — each worker runs
its own engine instance, so a folder renders many times faster than real time.
Progress shows *file x of N* with per-file percentage. By default it renders
with the **current panel sound**; untick to use the baked-in default preset.

Inputs at other sample rates are conditioned to the engine rate by a
high-quality windowed-sinc resampler (the same role the browser's decoder
played); files already at the engine rate enter the chain untouched.

## Command line (automation)

```
omegafm.exe --in in.wav --out out.wav            single file
omegafm.exe --batch inDir outDir --threads 8     whole folder
options:
  --preset file.json    any OmegaFM preset (default: baked-in AGGRESSIVE·MEDIUM)
  --rate N              44100 | 48000 | 176400 | 192000   (default: auto)
  --quality Q           reference | mastering | balanced | eco
  --mpx                 render the MPX composite (192 k, pilot + RDS)
  --bits N              16 | 24 | 32 (float)              (default: 24)
```

## Requirements

* Windows 10 (1809+) or Windows 11, x64. No dependencies for the CLI/batch.
* The GUI uses the **WebView2 Runtime** (preinstalled on all current
  Windows 10/11; otherwise Microsoft's free "Evergreen WebView2 Runtime").
* If the GUI ever can't start, everything is still available via
  `omegafm.exe --help`.

---
DSP chain: phase rotator → gated AGC (HP-weighted detector) → 4-band parametric
EQ → psychoacoustic bass → LR4 crossover → 4-band leveler → 4-band "signature"
compressor with spectral coupling → 4× oversampled back end: 75/50 µs
pre-emphasis → dynamic HF limiter → look-ahead wideband limiter → 4-band masked
clipper (rack) / distortion-cancelled main clipper with adaptive guard →
15 kHz brickwall → dual band-limited overshoot control → stereo generator with
filtered composite clipping, 9 % pilot, CENELEC RDS (0A/2A, CRC-10, differential
biphase, RC β=1) and BS.412 MPX power control. Rack: azimuth repair,
de-clipper/de-lossifier, de-esser, dehummer+gate, natural dynamics, vintage
tube (ADAA), dynamic EQ, 4-band masked clipper, power bass, sonic maximizer,
multipath governor, stereo widener.
