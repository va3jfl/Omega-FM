"""
tools/render_ui.py
==================

Offscreen smoke test + screenshots proving the scalable canvas.

Renders the panel through the ScalableView at two window sizes
(1600x900 and 880x540).  Because the panel lives in a QGraphicsScene
and the view calls fitInView(KeepAspectRatio), the small render is a
perfectly scaled miniature - the "fits every screen" requirement.

Run:  QT_QPA_PLATFORM=offscreen python tools/render_ui.py
"""

from __future__ import annotations

import os
import sys
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QRectF

from omegafm.processor import Controller
from omegafm.ui import (FrontPanel, AdvancedPanel, ScalableView,
                        PluginDetailPanel)


def demo_meters(c: Controller):
    """Inject lively values so the screenshots show the meters working."""
    c.meters.push(
        input_l=-9.5, input_r=-11.0,
        agc_gain=4.2,
        lev_gain=[1.8, -0.9, 1.2, -2.1],
        comp_gr=[3.5, 5.8, 6.4, 4.9],
        hf_gr=2.8, wb_gr=3.6,
        out_l=-1.2, out_r=-1.6,
        mpx_pct=97.0,
        running=True, latency_ms=26.0, underruns=0,
    )


def render(view: ScalableView, w: int, h: int, path: pathlib.Path):
    view.resize(w, h)
    view.show()
    QApplication.processEvents()
    img = QImage(w, h, QImage.Format_ARGB32)
    p = QPainter(img)
    view.render(p, QRectF(0, 0, w, h))
    p.end()
    img.save(str(path))
    print(f"rendered {w}x{h} -> {path}")


def main():
    app = QApplication(sys.argv)
    c = Controller()
    c.set_preset(signature="AGGRESSIVE", density="MEDIUM")
    panel = FrontPanel(c)
    panel.meter_timer.stop()
    panel.rds_timer.stop()
    demo_meters(c)
    panel._poll_meters()
    panel.status.setText("RUNNING · MPX 192k · 26 ms · xruns 0")
    view = ScalableView(panel)

    out = pathlib.Path(__file__).resolve().parents[1]
    render(view, 1600, 900, out / "omegafm_panel_full.png")
    render(view, 880, 540, out / "omegafm_panel_small.png")

    # advanced window (opened exactly as the ADVANCED button does)
    panel._open_advanced()
    render(panel._adv_view, 1440, 840, out / "omegafm_panel_advanced.png")

    # plugins list + a live detail window for the shipped widener
    c.plugin_host.set_enabled("stereo_widener", True)
    c.plugin_host.set_enabled("power_bass", True)
    lp = c.plugin_host.plugins["power_bass"]
    c.plugin_host.set_param("power_bass", "drive_db", 6.0)
    import numpy as np
    tt = np.arange(8192) / 48000.0
    lp.instance.process(np.stack([0.2 * np.sin(2 * np.pi * 70 * tt)] * 2, axis=1))
    panel._open_plugins()
    plg = panel._plugins_view.proxy.widget()
    plg.refresh(scan=False)
    render(panel._plugins_view, 780, 520, out / "omegafm_panel_plugins.png")
    det = ScalableView(PluginDetailPanel(c.plugin_host, lp),
                       title="OmegaFM · Power Bass")
    det.proxy.widget().timer.stop()
    det.proxy.widget()._poll()
    render(det, 540, 340, out / "omegafm_panel_plugin_detail.png")

    c.plugin_host.set_enabled("de_esser", True)
    lde = c.plugin_host.plugins["de_esser"]
    from scipy.signal import butter, lfilter
    fsv = 48000.0
    tv = np.arange(8192) / fsv
    be, ae = butter(2, [5000 / (fsv / 2), 8000 / (fsv / 2)], btype="band")
    ess = lfilter(be, ae, np.random.default_rng(2).standard_normal(8192)) * 0.28
    lde.instance.process(np.stack([ess] * 2, axis=1))
    det2 = ScalableView(PluginDetailPanel(c.plugin_host, lde),
                        title="OmegaFM · De-Esser")
    det2.proxy.widget().timer.stop()
    det2.proxy.widget()._poll()
    render(det2, 540, 340, out / "omegafm_panel_deesser.png")

    c.plugin_host.set_enabled("sonic_maximizer", True)
    lsm = c.plugin_host.plugins["sonic_maximizer"]
    c.plugin_host.set_param("sonic_maximizer", "proc", 6.0)
    bmx, amx = butter(2, 1200 / 24000.0)
    dull = lfilter(bmx, amx, np.random.default_rng(3).standard_normal(16384)) * 0.15
    lsm.instance.process(np.stack([dull] * 2, axis=1))
    det3 = ScalableView(PluginDetailPanel(c.plugin_host, lsm),
                        title="OmegaFM · Sonic Maximizer")
    det3.proxy.widget().timer.stop()
    det3.proxy.widget()._poll()
    render(det3, 540, 340, out / "omegafm_panel_maximizer.png")

    c.plugin_host.set_enabled("de_clipper", True)
    ldc = c.plugin_host.plugins["de_clipper"]
    tdc = np.arange(16384) / 48000.0
    clipped = np.clip(0.85 * np.sin(2 * np.pi * 1000 * tdc), -0.5, 0.5)
    for _ in range(3):
        ldc.instance.process(np.stack([clipped[:8192]] * 2, axis=1))
    det4 = ScalableView(PluginDetailPanel(c.plugin_host, ldc),
                        title="OmegaFM · De-Clipper / De-Lossifier")
    det4.proxy.widget().timer.stop()
    det4.proxy.widget()._poll()
    render(det4, 540, 340, out / "omegafm_panel_declipper.png")

    c.plugin_host.set_enabled("natural_dynamics", True)
    lnd = c.plugin_host.plugins["natural_dynamics"]
    c.plugin_host.set_param("natural_dynamics", "amount", 6.0)
    tn = np.arange(48000) / 48000.0
    ke = np.exp(-(tn % 0.25) / 0.03) * ((tn % 0.25) < 0.1)
    beat = np.tanh((0.7 * np.sin(2 * np.pi * 70 * tn) * ke
                    + 0.35 * np.sin(2 * np.pi * 300 * tn)) * 2.2) * 0.9
    sqf = np.stack([beat] * 2, axis=1)
    for k in range(0, 48000 - 4096, 4096):
        lnd.instance.process(sqf[k:k + 4096])
    det5 = ScalableView(PluginDetailPanel(c.plugin_host, lnd),
                        title="OmegaFM · Natural Dynamics")
    det5.proxy.widget().timer.stop()
    det5.proxy.widget()._poll()
    render(det5, 540, 340, out / "omegafm_panel_naturaldyn.png")

    c.plugin_host.set_enabled("azimuth_repair", True)
    laz = c.plugin_host.plugins["azimuth_repair"]
    from scipy.signal import butter as _b2, lfilter as _l2
    fsz = 48000.0
    rngz = np.random.default_rng(6)
    bz, az = _b2(2, 3000 / (fsz / 2))
    mz = _l2(bz, az, rngz.standard_normal(int(fsz * 6))) * 0.2
    srcz = np.stack([mz, np.roll(mz, 7)], axis=1)
    for k in range(0, len(srcz) - 4096, 4096):
        laz.instance.process(srcz[k:k + 4096])
    det6 = ScalableView(PluginDetailPanel(c.plugin_host, laz),
                        title="OmegaFM · Azimuth / Stereo Repair")
    det6.proxy.widget().timer.stop()
    det6.proxy.widget()._poll()
    render(det6, 620, 340, out / "omegafm_panel_azimuth.png")

    c.plugin_host.set_enabled("dehum_gate", True)
    ldh = c.plugin_host.plugins["dehum_gate"]
    fsd = 48000.0
    td = np.arange(int(fsd * 7)) / fsd
    rngd = np.random.default_rng(8)
    humd = (10 ** (-38 / 20) * np.sin(2 * np.pi * 50 * td)
            + 10 ** (-43 / 20) * np.sin(2 * np.pi * 100 * td)
            + 10 ** (-41 / 20) * np.sin(2 * np.pi * 150 * td))
    feedd = np.stack([humd] * 2, axis=1) \
        + rngd.standard_normal((len(td), 2)) * 10 ** (-66 / 20)
    for k in range(0, len(feedd) - 4096, 4096):
        ldh.instance.process(feedd[k:k + 4096].copy())
    det7 = ScalableView(PluginDetailPanel(c.plugin_host, ldh),
                        title="OmegaFM · Dehummer + Gate")
    det7.proxy.widget().timer.stop()
    det7.proxy.widget()._poll()
    render(det7, 680, 340, out / "omegafm_panel_dehum.png")

    c.plugin_host.set_enabled("stereo_governor", True)
    lsg = c.plugin_host.plugins["stereo_governor"]
    rngg = np.random.default_rng(9)
    decg = np.stack([rngg.standard_normal(45000),
                     rngg.standard_normal(45000)], axis=1) * 0.15
    for k in range(0, 45000 - 4096, 4096):
        lsg.instance.process(decg[k:k + 4096].copy())
    det8 = ScalableView(PluginDetailPanel(c.plugin_host, lsg),
                        title="OmegaFM · Multipath Governor")
    det8.proxy.widget().timer.stop()
    det8.proxy.widget()._poll()
    render(det8, 620, 340, out / "omegafm_panel_governor.png")

    from omegafm.ui import PluginsPanel
    pl_panel = PluginsPanel(c)
    for pid, lpx in c.plugin_host.plugins.items():
        dpx = PluginDetailPanel(c.plugin_host, lpx)
        dpx.timer.stop()
        dpx._poll()
    print(f"GUI smoke: PluginsPanel + {len(c.plugin_host.plugins)} detail panels OK")
    print("UI smoke test OK")


if __name__ == "__main__":
    main()
