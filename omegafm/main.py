"""
omegafm.main
============

Application wiring: load config -> Controller -> FrontPanel inside the
ScalableView -> save config on exit.
"""

from __future__ import annotations

import sys


def main():
    from PySide6.QtWidgets import QApplication
    from .processor import Controller
    from .ui import FrontPanel, ScalableView
    from . import config, __version__
    print(f"OmegaFM v{__version__}")

    app = QApplication(sys.argv)
    app.setApplicationName("OmegaFM")

    c = Controller()
    cfg = config.load()
    from .processor import FACTORY_RACK
    # Boot policy: every launch starts from the factory default preset -
    # AGGRESSIVE-LIGHT plus the factory plugin rack - so the app always
    # comes up in the exact validated state. Only station identity and
    # local setup survive between sessions (RDS, injection levels, I/O
    # trims, devices). Sound tweaks live in preset files: IMPORT one.
    c.plugin_host.load_state(FACTORY_RACK)
    if cfg:
        from .presets import EXCLUDE
        c.trims = {k: v for k, v in dict(cfg.get("trims", {})).items()
                   if k in EXCLUDE}
        c._push()

    panel = FrontPanel(c)
    if cfg.get("buffer"):
        panel.buf.setCurrentText(str(cfg["buffer"]))
    if cfg.get("mpx_mode"):
        panel.outmode.setCurrentIndex(1)
    panel.sync_from_params()

    view = ScalableView(panel)
    # with the streaming GC parked, sweep gen-0 briefly from the GUI
    # thread so week-long sessions can't accumulate Qt cycles
    import gc
    from PySide6.QtCore import QTimer
    _gc_t = QTimer(view)
    _gc_t.timeout.connect(lambda: gc.collect(0))
    _gc_t.start(10000)
    view.show()

    rc = app.exec()

    c.stop()
    config.save({
        "signature": c.signature,
        "density": c.density,
        "trims": c.trims,
        "custom_base": c.custom_base,
        "preset_name": c.preset_name,
        "app_version": __version__,
        "plugins": c.plugin_host.state(),
        "buffer": int(panel.buf.currentText()),
        "mpx_mode": panel.outmode.currentIndex() == 1,
    })
    sys.exit(rc)


if __name__ == "__main__":
    main()
