"""
omegafm.ui
==========

Classic hardware front panel in PySide6.

Scalability
-----------
The entire panel is a fixed-reference-size QWidget (1600 x 1010) hosted
inside a QGraphicsScene via QGraphicsProxyWidget.  ScalableView calls
fitInView(..., KeepAspectRatio) on every resize, so the *whole* canvas -
meters, type, knobs, LEDs - scales up or down to any window or screen.
Nothing ever clips; on a small laptop it simply renders smaller.

Design language: dark charcoal chassis, one amber accent, green/red
status LEDs, generous section grooves - restrained, legible, hardware.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit,
    QDoubleSpinBox,
    QGroupBox, QFrame, QSizePolicy, QFileDialog, QMessageBox, QSpinBox)

from . import __version__
from .params import SIGNATURE_ORDER, DENSITY_ORDER
from .processor import list_devices, BUFFER_SIZES

REF_W, REF_H = 1600, 1010

# ------------------------------------------------------------------ palette
BG     = "#1b1d21"
PANEL  = "#212429"
INSET  = "#17191d"
LINE   = "#33373d"
TEXT   = "#d6d8dc"
DIM    = "#8a8f98"
AMBER  = "#ffb43c"
GREEN  = "#59d66a"
RED    = "#ff5347"
YELLOW = "#ffd24a"

QSS = f"""
* {{ font-family: 'DejaVu Sans', 'Segoe UI', sans-serif; }}
QWidget#panel {{ background: {BG}; }}
QLabel {{ color: {TEXT}; font-size: 12px; }}
QLabel[cls="section"] {{
    color: {DIM}; font-size: 10px; font-weight: 700; letter-spacing: 2px; }}
QLabel[cls="brand"] {{
    color: {TEXT}; font-size: 26px; font-weight: 800; letter-spacing: 3px; }}
QLabel[cls="brandsub"] {{
    color: {DIM}; font-size: 10px; letter-spacing: 4px; }}
QLabel[cls="lamp"] {{
    color: {AMBER}; font-size: 22px; font-weight: 800; letter-spacing: 2px;
    background: {INSET}; border: 1px solid {LINE}; border-radius: 4px;
    padding: 4px 14px; }}
QLabel[cls="status"] {{ color: {DIM}; font-size: 11px; }}
QGroupBox {{
    color: {DIM}; font-size: 10px; font-weight: 700; letter-spacing: 2px;
    border: 1px solid {LINE}; border-radius: 6px;
    margin-top: 8px; padding: 5px 6px 4px 6px; background: {PANEL}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QPushButton {{
    color: {TEXT}; background: #2a2e34; border: 1px solid {LINE};
    border-radius: 4px; padding: 5px 8px; font-size: 12px; font-weight: 600;
    min-height: 18px; }}
QPushButton:hover {{ background: #33383f; }}
QPushButton:checked {{
    background: #3a3018; border: 1px solid {AMBER}; color: {AMBER}; }}
QPushButton#start {{ font-size: 14px; font-weight: 800; padding: 8px; }}
QPushButton#start:checked {{
    background: #17301d; border-color: {GREEN}; color: {GREEN}; }}
QComboBox, QLineEdit, QSpinBox {{
    color: {TEXT}; background: {INSET}; border: 1px solid {LINE};
    border-radius: 4px; padding: 3px 6px; font-size: 12px; min-height: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL}; color: {TEXT}; selection-background-color: #3a3018;
    selection-color: {AMBER}; }}
QCheckBox {{ color: {TEXT}; font-size: 11px; spacing: 5px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px; border: 1px solid {LINE}; border-radius: 3px;
    background: {INSET}; }}
QCheckBox::indicator:checked {{ background: {AMBER}; border-color: {AMBER}; }}
QFrame[cls="rule"] {{ background: {LINE}; max-height: 1px; min-height: 1px; }}
"""


# =========================================================================== #
#  Custom widgets
# =========================================================================== #

class LEDBar(QWidget):
    """Vertical LED bargraph with a numeric LED readout underneath.

    mode:
      "level"   bottom-up, green->yellow->red, dB scale (lo..hi)
      "gr"      top-down red, 0..(hi-lo) dB of gain reduction
      "bipolar" center-out amber, +/- half-range (leveler / AGC gain)

    The readout replaces a printed scale: an inset box under the bar
    shows the actual number in the bar's colour.  Ballistics keep it
    readable - GR holds its most recent maximum for ~0.9 s before
    tracking down, level shows the peak-hold, bipolar values glide.
    """

    def __init__(self, label: str, mode: str = "level", lo: float = -42.0,
                 hi: float = 3.0, segments: int = 26, parent=None,
                 unit: str = "db"):
        super().__init__(parent)
        self.label = label
        self.mode = mode
        self.unit = unit                 # "db" | "pct" (MOD %) | "hz"
        self.lo, self.hi = lo, hi
        self.segments = segments
        self.value = lo if mode == "level" else 0.0
        self.peak = lo
        self._peak_t = 0.0
        self.disp = 0.0 if mode != "level" else lo
        self._disp_t = 0.0
        self.setMinimumSize(28, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, v: float):
        self.value = float(v)
        now = time.monotonic()
        if self.mode == "level":
            if v >= self.peak or now - self._peak_t > 1.4:
                self.peak = v
                self._peak_t = now
        elif self.mode == "gr":
            # hold the deepest recent reduction so the number is readable
            if v >= self.disp - 0.05:
                self.disp = v
                self._disp_t = now
            elif now - self._disp_t > 0.9:
                self.disp = v
        else:                                      # bipolar: glide
            self.disp = 0.75 * self.disp + 0.25 * v
        self.update()

    # ------------------------------------------------------------------ text
    def _readout(self):
        """(text, QColor) for the LED numeric display."""
        if self.mode == "gr":
            if self.disp >= 0.05:
                return f"-{self.disp:.1f}", QColor(RED)
            return "0.0", QColor("#7d8894")
        if self.mode == "bipolar":
            if abs(self.disp) < 0.05:
                return "0.0", QColor("#7d8894")
            return f"{self.disp:+.1f}", QColor(AMBER)
        # level
        if self.unit == "pct":
            v = max(self.peak, 0.0)
            col = GREEN if v < 90 else (YELLOW if v <= 100 else RED)
            return f"{v:.0f}", QColor(col)
        if self.peak <= self.lo + 0.5:
            return "--", QColor("#7d8894")
        if self.unit == "hz":
            return f"{self.peak:.0f}", QColor(AMBER)
        col = GREEN if self.peak < -12 else (YELLOW if self.peak < -3 else RED)
        return f"{self.peak:.0f}", QColor(col)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        label_h = 14
        read_h = 15
        bar = QRectF(w * 0.20, 4, w * 0.60, h - label_h - read_h - 12)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(QPen(QColor(LINE)))
        p.setBrush(QColor(INSET))
        p.drawRoundedRect(bar, 3, 3)

        n = self.segments
        gap = 2.0
        seg_h = (bar.height() - gap * (n + 1)) / n
        frac = float(np.clip((self.value - self.lo) / (self.hi - self.lo), 0, 1))

        for i in range(n):
            y = bar.bottom() - gap * (i + 1) - seg_h * (i + 1)
            r = QRectF(bar.left() + 3, y, bar.width() - 6, seg_h)
            fi = (i + 0.5) / n
            if self.mode == "level":
                on = fi <= frac
                v_here = self.lo + fi * (self.hi - self.lo)
                if self.unit == "pct":
                    base = GREEN if v_here < 90 else (YELLOW if v_here <= 100 else RED)
                    col = QColor(base) if on else QColor(base).darker(340)
                elif self.unit == "hz":
                    col = QColor(AMBER) if on else QColor("#25272c")
                else:
                    base = GREEN if v_here < -12 else (YELLOW if v_here < -3 else RED)
                    col = QColor(base) if on else QColor(base).darker(340)
            elif self.mode == "gr":
                gr_frac = float(np.clip(self.value / (self.hi - self.lo), 0, 1))
                on = (1.0 - fi) <= gr_frac             # lights from the top
                col = QColor(RED) if on else QColor("#25272c")
            else:                                      # bipolar
                half = (self.hi - self.lo) / 2.0
                vfrac = float(np.clip(self.value / half, -1, 1)) * 0.5
                c = fi - 0.5
                on = (0 <= c <= vfrac) or (vfrac <= c <= 0) or abs(c) < 0.5 / n
                col = QColor(AMBER) if on else QColor("#25272c")
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(r, 1.5, 1.5)

        if self.mode == "level":
            pf = float(np.clip((self.peak - self.lo) / (self.hi - self.lo), 0, 1))
            y = bar.bottom() - pf * bar.height()
            p.setPen(QPen(QColor("#ffffff"), 1.4))
            p.drawLine(QPointF(bar.left() + 2, y), QPointF(bar.right() - 2, y))

        # ---- LED numeric readout box ------------------------------------
        txt, tcol = self._readout()
        bw = min(w - 4.0, 46.0)
        rb = QRectF(w / 2.0 - bw / 2.0, bar.bottom() + 5, bw, read_h)
        p.setPen(QPen(QColor(LINE)))
        p.setBrush(QColor("#101216"))
        p.drawRoundedRect(rb, 3, 3)
        f = QFont("DejaVu Sans Mono")
        f.setPixelSize(10); f.setBold(True)
        p.setFont(f)
        p.setPen(tcol)
        p.drawText(rb, Qt.AlignCenter, txt)

        # ---- label --------------------------------------------------------
        p.setPen(QColor(DIM))
        f = QFont(); f.setPixelSize(10); f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(0, h - label_h, w, label_h), Qt.AlignCenter, self.label)
        p.end()


class Knob(QWidget):
    """Painted rotary control with value readout. 270 degree sweep.
    Drag vertically, scroll wheel, double-click = midpoint."""

    valueChanged = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float,
                 fmt: str = "{:+.1f}", suffix: str = "dB", parent=None):
        super().__init__(parent)
        self.label, self.lo, self.hi = label, lo, hi
        self.fmt, self.suffix = fmt, suffix
        self._v = float(value)
        self._drag_y = None
        self._drag_v = None
        self.setMinimumSize(72, 96)
        self.setMaximumSize(116, 132)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def value(self):
        return self._v

    def set_value(self, v, emit=False):
        v = float(np.clip(v, self.lo, self.hi))
        if v != self._v:
            self._v = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    def mousePressEvent(self, e):
        self._drag_y = e.position().y()
        self._drag_v = self._v

    def mouseMoveEvent(self, e):
        if self._drag_y is None:
            return
        dy = self._drag_y - e.position().y()
        self.set_value(self._drag_v + dy / 140.0 * (self.hi - self.lo), emit=True)

    def mouseReleaseEvent(self, e):
        self._drag_y = None

    def wheelEvent(self, e):
        step = (self.hi - self.lo) / 40.0
        self.set_value(self._v + (step if e.angleDelta().y() > 0 else -step),
                       emit=True)

    def mouseDoubleClickEvent(self, e):
        self.set_value((self.lo + self.hi) / 2.0, emit=True)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        d = min(w, h - 34)
        cx, cy = w / 2.0, 6 + d / 2.0
        r = d / 2.0 - 3

        start, sweep = 225.0, -270.0
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setPen(QPen(QColor(INSET), 5, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, int(start * 16), int(sweep * 16))
        frac = (self._v - self.lo) / (self.hi - self.lo)
        p.setPen(QPen(QColor(AMBER), 5, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, int(start * 16), int(sweep * frac * 16))

        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(QColor("#2b2f35"))
        p.drawEllipse(QPointF(cx, cy), r - 6, r - 6)
        ang = np.deg2rad(start + sweep * frac)
        p.setPen(QPen(QColor(TEXT), 2.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + 0.35 * r * np.cos(ang),
                           cy - 0.35 * r * np.sin(ang)),
                   QPointF(cx + 0.85 * (r - 6) * np.cos(ang),
                           cy - 0.85 * (r - 6) * np.sin(ang)))

        f = QFont(); f.setPixelSize(10); f.setBold(True); p.setFont(f)
        p.setPen(QColor(DIM))
        p.drawText(QRectF(0, h - 30, w, 14), Qt.AlignCenter, self.label)
        p.setPen(QColor(TEXT))
        val = self.fmt.format(self._v)
        if self.suffix:
            val += f" {self.suffix}"
        p.drawText(QRectF(0, h - 16, w, 14), Qt.AlignCenter, val)
        p.end()


def _rule():
    fr = QFrame()
    fr.setProperty("cls", "rule")
    return fr


def _section_label(txt):
    lbl = QLabel(txt)
    lbl.setProperty("cls", "section")
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


# =========================================================================== #
#  Front panel
# =========================================================================== #

class FrontPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        self.setObjectName("panel")
        self.setFixedSize(REF_W, REF_H)
        self.setStyleSheet(QSS)
        self._building = True
        self._rds_mtime = 0.0
        self._ps_chunks: list[str] = []
        self._ps_i = 0
        self._adv_panel = None
        self._adv_view = None
        self._plugins_view = None
        self._child_views = []
        self._build()
        self._building = False
        self.refresh_devices()
        self.sync_from_params()

        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self._poll_meters)
        self.meter_timer.start(33)
        self.rds_timer = QTimer(self)
        self.rds_timer.timeout.connect(self._poll_rds)
        self.rds_timer.start(1000)

    # ------------------------------------------------------------------ build
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(8)

        # header ------------------------------------------------------------
        head = QHBoxLayout()
        brand_box = QVBoxLayout(); brand_box.setSpacing(0)
        b = QLabel("OMEGA FM"); b.setProperty("cls", "brand")
        s = QLabel(f"DIGITAL FM BROADCAST AUDIO PROCESSOR  ·  STEREO GEN  ·  RDS  ·  v{__version__}")
        s.setProperty("cls", "brandsub")
        brand_box.addWidget(b); brand_box.addWidget(s)
        head.addLayout(brand_box)
        head.addStretch(1)
        self.lamp = QLabel("NATURAL · MEDIUM")
        self.lamp.setProperty("cls", "lamp")
        head.addWidget(self.lamp)
        head.addSpacing(14)
        self.status = QLabel("STOPPED")
        self.status.setProperty("cls", "status")
        self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status.setMinimumWidth(250)
        head.addWidget(self.status)
        root.addLayout(head)
        root.addWidget(_rule())

        # middle ------------------------------------------------------------
        mid = QHBoxLayout(); mid.setSpacing(12)
        root.addLayout(mid, stretch=1)
        mid.addWidget(self._build_meter_bridge(), stretch=5)
        mid.addWidget(self._build_controls(), stretch=3)

        # bottom ------------------------------------------------------------
        root.addWidget(self._build_tweaks())

    def _build_meter_bridge(self):
        box = QGroupBox("METERS")
        lay = QHBoxLayout(box); lay.setSpacing(6)

        def group(title, bars):
            v = QVBoxLayout(); v.setSpacing(2)
            hh = QHBoxLayout(); hh.setSpacing(3)
            for bar in bars:
                hh.addWidget(bar)
            v.addLayout(hh, stretch=1)
            v.addWidget(_section_label(title))
            w = QWidget(); w.setLayout(v)
            return w

        self.m_in_l = LEDBar("L"); self.m_in_r = LEDBar("R")
        self.m_agc = LEDBar("±12", "bipolar", -12, 12)
        self.m_lev = [LEDBar(n, "bipolar", -5, 5) for n in ("LO", "LM", "HM", "HI")]
        self.m_cmp = [LEDBar(n, "gr", 0, 14) for n in ("LO", "LM", "HM", "HI")]
        self.m_hf = LEDBar("HF", "gr", 0, 14)
        self.m_wb = LEDBar("WB", "gr", 0, 10)
        self.m_out_l = LEDBar("L"); self.m_out_r = LEDBar("R")
        self.m_mpx = LEDBar("MPX", "level", 0, 110, unit="pct")

        lay.addWidget(group("INPUT", [self.m_in_l, self.m_in_r]))
        lay.addWidget(group("AGC", [self.m_agc]))
        lay.addWidget(group("LEVELER dB", self.m_lev))
        lay.addWidget(group("COMPRESSOR GR", self.m_cmp))
        lay.addWidget(group("LIMIT", [self.m_hf, self.m_wb]))
        lay.addWidget(group("OUTPUT", [self.m_out_l, self.m_out_r]))
        lay.addWidget(group("MOD %", [self.m_mpx]))
        return box

    def _build_controls(self):
        col = QWidget()
        v = QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # SIGNATURE 2x2
        gb = QGroupBox("SIGNATURE")
        g = QGridLayout(gb); g.setSpacing(6)
        self.sig_btns = {}
        for i, name in enumerate(SIGNATURE_ORDER):
            btn = QPushButton(name); btn.setCheckable(True)
            btn.clicked.connect(lambda _, n=name: self._pick_signature(n))
            self.sig_btns[name] = btn
            g.addWidget(btn, i // 2, i % 2)
        v.addWidget(gb)

        # DENSITY 1x3
        gb = QGroupBox("PROCESSING DENSITY")
        g = QHBoxLayout(gb); g.setSpacing(6)
        self.den_btns = {}
        for name in DENSITY_ORDER:
            btn = QPushButton(name); btn.setCheckable(True)
            btn.clicked.connect(lambda _, n=name: self._pick_density(n))
            self.den_btns[name] = btn
            g.addWidget(btn)
        v.addWidget(gb)

        # TRANSMISSION
        gb = QGroupBox("TRANSMISSION")
        g = QGridLayout(gb); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        g.addWidget(QLabel("Pre-emphasis"), 0, 0)
        self.preemph = QComboBox()
        self.preemph.addItems(["75 µs", "50 µs", "Flat"])
        self.preemph.currentIndexChanged.connect(self._push_transmission)
        g.addWidget(self.preemph, 0, 1)
        g.addWidget(QLabel("Output"), 0, 2)
        self.outmode = QComboBox()
        self.outmode.addItems(["Stereo L/R @48k", "MPX composite @192k"])
        g.addWidget(self.outmode, 0, 3)
        g.addWidget(QLabel("Composite clip"), 1, 0)
        self.clipmode = QComboBox()
        self.clipmode.addItems(["Filtered", "Raw", "Off"])
        self.clipmode.currentIndexChanged.connect(self._push_transmission)
        g.addWidget(self.clipmode, 1, 1)
        self.bs412_on = QCheckBox("BS.412 limit")
        self.bs412_on.toggled.connect(self._push_bs412)
        g.addWidget(self.bs412_on, 2, 0)
        self.bs412_tgt = QDoubleSpinBox()
        self.bs412_tgt.setRange(-6.0, 3.0); self.bs412_tgt.setSingleStep(0.1)
        self.bs412_tgt.setSuffix(" dBr"); self.bs412_tgt.setValue(0.0)
        self.bs412_tgt.valueChanged.connect(self._push_bs412)
        g.addWidget(self.bs412_tgt, 2, 1)
        self.bs412_lbl = QLabel("-- dBr")
        self.bs412_lbl.setProperty("cls", "status")
        g.addWidget(self.bs412_lbl, 2, 2, 1, 2)
        self.mon_deemph = QCheckBox("De-emph monitor")
        self.mon_deemph.toggled.connect(self._push_transmission)
        g.addWidget(self.mon_deemph, 1, 2, 1, 2)
        v.addWidget(gb)

        # RDS
        gb = QGroupBox("RDS ENCODER")
        g = QGridLayout(gb); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        self.rds_on = QCheckBox("Enable")
        self.rds_on.toggled.connect(self._push_rds)
        g.addWidget(self.rds_on, 0, 0)
        g.addWidget(QLabel("PI"), 0, 1)
        self.rds_pi = QLineEdit("1234"); self.rds_pi.setMaximumWidth(64)
        self.rds_pi.editingFinished.connect(self._push_rds)
        g.addWidget(self.rds_pi, 0, 2)
        g.addWidget(QLabel("PTY"), 0, 3)
        self.rds_pty = QSpinBox(); self.rds_pty.setRange(0, 31)
        self.rds_pty.setValue(10)
        self.rds_pty.valueChanged.connect(self._push_rds)
        g.addWidget(self.rds_pty, 0, 4)
        g.addWidget(QLabel("PS"), 1, 0)
        self.rds_ps = QLineEdit("OMEGAFM"); self.rds_ps.setMaxLength(8)
        self.rds_ps.editingFinished.connect(self._push_rds)
        g.addWidget(self.rds_ps, 1, 1, 1, 2)
        self.rds_dyn = QCheckBox("Dynamic PS")
        self.rds_dyn.toggled.connect(self._push_rds)
        g.addWidget(self.rds_dyn, 1, 3, 1, 2)
        g.addWidget(QLabel("RT"), 2, 0)
        self.rds_rt = QLineEdit("OmegaFM - broadcast processing in Python")
        self.rds_rt.setMaxLength(64)
        self.rds_rt.editingFinished.connect(self._push_rds)
        g.addWidget(self.rds_rt, 2, 1, 1, 4)
        g.addWidget(QLabel("Live file"), 3, 0)
        self.rds_file = QLineEdit()
        self.rds_file.setPlaceholderText("path to now-playing .txt")
        g.addWidget(self.rds_file, 3, 1, 1, 2)
        btn = QPushButton("…"); btn.setMaximumWidth(30)
        btn.clicked.connect(self._browse_rds)
        g.addWidget(btn, 3, 3)
        self.rds_watch = QCheckBox("Watch")
        self.rds_watch.toggled.connect(self._push_rds)
        g.addWidget(self.rds_watch, 3, 4)
        v.addWidget(gb)

        # AUDIO I/O
        gb = QGroupBox("AUDIO I/O")
        g = QGridLayout(gb); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        g.addWidget(QLabel("Input"), 0, 0)
        self.dev_in = QComboBox()
        g.addWidget(self.dev_in, 0, 1, 1, 2)
        g.addWidget(QLabel("Output"), 1, 0)
        self.dev_out = QComboBox()
        g.addWidget(self.dev_out, 1, 1, 1, 2)
        g.addWidget(QLabel("Buffer"), 2, 0)
        self.buf = QComboBox()
        for bsz in BUFFER_SIZES:
            self.buf.addItem(str(bsz))
        self.buf.setCurrentText("1024")
        g.addWidget(self.buf, 2, 1)
        rf = QPushButton("Rescan")
        rf.clicked.connect(self.refresh_devices)
        g.addWidget(rf, 2, 2)
        v.addWidget(gb)

        # BYPASS
        gb = QGroupBox("STAGE BYPASS")
        g = QGridLayout(gb); g.setHorizontalSpacing(10); g.setVerticalSpacing(4)
        self.bypass = {}
        stages = [("Rotator", "bypass_rotator"), ("AGC", "bypass_agc"),
                  ("EQ", "bypass_eq"), ("Bass", "bypass_bass"),
                  ("Leveler", "bypass_leveler"), ("Comp", "bypass_comp"),
                  ("HF Lim", "bypass_hf_lim"), ("WB Lim", "bypass_wb_lim"),
                  ("Clipper", "bypass_clipper"), ("Comp Clip", "bypass_comp_clip")]
        for i, (label, key) in enumerate(stages):
            cb = QCheckBox(label)
            cb.toggled.connect(lambda on, k=key: self._push_bypass(k, on))
            self.bypass[key] = cb
            g.addWidget(cb, i // 5, i % 5)
        v.addWidget(gb)

        # START / SAVE
        h = QHBoxLayout(); h.setSpacing(8)
        self.start_btn = QPushButton("START")
        self.start_btn.setObjectName("start")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self._toggle_engine)
        h.addWidget(self.start_btn, stretch=2)
        adv = QPushButton("ADVANCED")
        adv.clicked.connect(self._open_advanced)
        h.addWidget(adv, stretch=1)
        save = QPushButton("SAVE CUSTOM")
        save.clicked.connect(self._save_custom)
        h.addWidget(save, stretch=1)
        v.addLayout(h)
        h2 = QHBoxLayout(); h2.setSpacing(8)
        plg = QPushButton("PLUGINS")
        plg.clicked.connect(self._open_plugins)
        h2.addWidget(plg)
        imp = QPushButton("IMPORT PRESET (.json)")
        imp.clicked.connect(self._import_preset)
        h2.addWidget(imp)
        exp = QPushButton("EXPORT PRESET (.json)")
        exp.clicked.connect(self._export_preset)
        h2.addWidget(exp)
        v.addLayout(h2)
        v.addStretch(1)
        return col

    def _build_tweaks(self):
        box = QGroupBox("TWEAK")
        h = QHBoxLayout(box); h.setSpacing(4)
        self.knobs = {}
        spec = [
            ("INPUT",     "input_gain_db",       -12, 12, "{:+.1f}", "dB", None),
            ("BASS",      "bass_shelf_db",         0,  8, "{:+.1f}", "dB", None),
            ("EQ 100",    "eq_gains",             -6,  6, "{:+.1f}", "dB", 0),
            ("EQ 400",    "eq_gains",             -6,  6, "{:+.1f}", "dB", 1),
            ("EQ 2k",     "eq_gains",             -6,  6, "{:+.1f}", "dB", 2),
            ("EQ 8k",     "eq_gains",             -6,  6, "{:+.1f}", "dB", 3),
            ("DRIVE",     "final_drive_db",       -6, 12, "{:+.1f}", "dB", None),
            ("HF LIM",    "hf_lim_threshold_db",  -8,  0, "{:+.1f}", "dB", None),
            ("CLIP",      "clip_drive_db",         0,  6, "{:+.1f}", "dB", None),
            ("COMPOSITE", "composite_drive_db",    0,  3, "{:+.1f}", "dB", None),
            ("PILOT",     "pilot_pct",             0, 20, "{:.1f}",  "%",  None),
            ("RDS INJ",   "rds_pct",               0,  8, "{:.1f}",  "%",  None),
            ("OUTPUT",    "output_level",          0,  1, "{:.2f}",  "",   None),
        ]
        for label, key, lo, hi, fmt, suf, idx in spec:
            k = Knob(label, lo, hi, 0.0, fmt, suf)
            k.valueChanged.connect(
                lambda vv, kk=key, ii=idx: self._push_knob(kk, ii, vv))
            self.knobs[(key, idx)] = k
            h.addWidget(k)
        return box

    # ------------------------------------------------------------------ pushers
    def _pick_signature(self, name):
        if self._building:
            return
        self.c.set_preset(signature=name)
        self.sync_from_params()

    def _pick_density(self, name):
        if self._building:
            return
        self.c.set_preset(density=name)
        self.sync_from_params()

    def _push_knob(self, key, idx, v):
        if self._building:
            return
        if idx is None:
            self.c.set_trim(**{key: v})
        else:
            arr = list(self.c.current_params()[key])
            arr[idx] = v
            self.c.set_trim(**{key: arr})

    def _push_transmission(self, *_):
        if self._building:
            return
        tau = {0: 75.0, 1: 50.0, 2: 0.0}[self.preemph.currentIndex()]
        mode = {0: "filtered", 1: "raw", 2: "off"}[self.clipmode.currentIndex()]
        self.c.set_trim(preemph_us=tau, composite_clip_mode=mode,
                        monitor_deemph=self.mon_deemph.isChecked())

    def _push_bs412(self, *_):
        if self._building:
            return
        self.c.set_trim(bs412_enable=self.bs412_on.isChecked(),
                        bs412_target_dbr=float(self.bs412_tgt.value()))

    def _push_rds(self, *_):
        if self._building:
            return
        try:
            pi = int(self.rds_pi.text(), 16) & 0xFFFF
        except ValueError:
            pi = 0x1234
        self.c.set_trim(rds_enable=self.rds_on.isChecked(),
                        rds_pi=pi, rds_pty=self.rds_pty.value(),
                        rds_ps=self.rds_ps.text() or "OMEGAFM",
                        rds_rt=self.rds_rt.text(),
                        rds_dynamic_ps=self.rds_dyn.isChecked(),
                        rds_file=self.rds_file.text(),
                        rds_watch_file=self.rds_watch.isChecked())

    def _push_bypass(self, key, on):
        if self._building:
            return
        self.c.set_trim(**{key: on})

    def _browse_rds(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Now-playing text file", "", "Text (*.txt);;All (*)")
        if path:
            self.rds_file.setText(path)
            self._push_rds()

    def _save_custom(self):
        self.c.save_custom()
        self.sync_from_params()

    def _open_advanced(self):
        if self._adv_view is None:
            self._adv_panel = AdvancedPanel(self.c)
            self._adv_view = ScalableView(self._adv_panel,
                                          title="OmegaFM · Advanced")
            self._adv_view.resize(1150, 670)
        self._adv_panel.sync()
        self._adv_view.show()
        self._adv_view.raise_()
        self._adv_view.activateWindow()

    def _open_plugins(self):
        if self._plugins_view is None:
            panel = PluginsPanel(self.c)
            self._plugins_view = ScalableView(panel, title="OmegaFM · Plugins")
            self._plugins_view.resize(PLG_W, PLG_H)
            self._child_views.append(self._plugins_view)
        else:
            self._plugins_view.proxy.widget().refresh(scan=True)
        self._plugins_view.show()
        self._plugins_view.raise_()
        self._plugins_view.activateWindow()

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import preset", "", "OmegaFM preset (*.json);;All (*)")
        if not path:
            return
        try:
            name = self.c.import_preset(path)
        except ValueError as e:
            QMessageBox.warning(self, "Import preset",
                                f"Could not import this file:\n\n{e}")
            return
        self.sync_from_params()
        if self._plugins_view is not None:
            self._plugins_view.proxy.widget().refresh(scan=False)
        self.status.setText(f"preset \"{name}\" loaded")

    def _export_preset(self):
        base = (self.c.preset_name or self.c.signature).lower()
        default = f"{base}_{self.c.density.lower()}.json".replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export preset", default,
            "OmegaFM preset (*.json);;All (*)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.c.export_preset(path)
        except OSError as e:
            QMessageBox.warning(self, "Export preset",
                                f"Could not write the file:\n\n{e}")
            return
        self.status.setText(f"preset saved · {path.rsplit('/',1)[-1]}")

    def _toggle_engine(self, checked):
        if checked:
            try:
                in_dev = self.dev_in.currentData()
                out_dev = self.dev_out.currentData()
                mpx = self.outmode.currentIndex() == 1
                self.c.start(in_dev, out_dev, int(self.buf.currentText()), mpx)
                self.start_btn.setText("STOP")
            except Exception as e:
                self.start_btn.setChecked(False)
                QMessageBox.critical(
                    self, "Audio engine",
                    f"Could not start the audio stream:\n\n{e}\n\n"
                    "MPX mode needs a device that supports 192 kHz. "
                    "Check the device selection and buffer size.")
        else:
            self.c.stop()
            self.start_btn.setText("START")

    # ------------------------------------------------------------------ sync
    def refresh_devices(self):
        self.dev_in.clear(); self.dev_out.clear()
        devs = list_devices()
        if not devs:
            self.dev_in.addItem("audio backend unavailable", None)
            self.dev_out.addItem("audio backend unavailable", None)
            return
        self.dev_in.addItem("System default", None)
        self.dev_out.addItem("System default", None)
        for d in devs:
            tag = "  [ASIO]" if d["is_asio"] else ""
            name = f'{d["index"]}: {d["name"]}{tag}'
            if d["max_in"] > 0:
                self.dev_in.addItem(name, d["index"])
            if d["max_out"] > 0:
                self.dev_out.addItem(name, d["index"])

    def sync_from_params(self):
        """Reflect controller state into every widget (signals blocked)."""
        self._building = True
        p = self.c.current_params()
        for n, btn in self.sig_btns.items():
            btn.setChecked(n == self.c.signature)
        for n, btn in self.den_btns.items():
            btn.setChecked(n == self.c.density)
        label = self.c.signature
        if label == "CUSTOM" and self.c.preset_name:
            label = self.c.preset_name[:14]
        self.lamp.setText(f"{label} · {self.c.density}")

        self.preemph.setCurrentIndex({75.0: 0, 50.0: 1, 0.0: 2}.get(p["preemph_us"], 0))
        self.clipmode.setCurrentIndex(
            {"filtered": 0, "raw": 1, "off": 2}.get(p["composite_clip_mode"], 0))
        self.mon_deemph.setChecked(bool(p["monitor_deemph"]))

        self.bs412_on.setChecked(bool(p.get("bs412_enable", False)))
        self.bs412_tgt.setValue(float(p.get("bs412_target_dbr", 0.0)))
        self.rds_on.setChecked(bool(p["rds_enable"]))
        self.rds_pi.setText(f'{p["rds_pi"]:04X}')
        self.rds_pty.setValue(int(p["rds_pty"]))
        self.rds_ps.setText(p["rds_ps"])
        self.rds_rt.setText(p["rds_rt"])
        self.rds_dyn.setChecked(bool(p["rds_dynamic_ps"]))
        self.rds_file.setText(p["rds_file"])
        self.rds_watch.setChecked(bool(p["rds_watch_file"]))

        for key, cb in self.bypass.items():
            cb.setChecked(bool(p[key]))

        for (key, idx), knob in self.knobs.items():
            val = p[key][idx] if idx is not None else p[key]
            knob.set_value(val)
        self._building = False
        if self._adv_panel is not None:
            self._adv_panel.sync()

    # ------------------------------------------------------------------ timers
    def _poll_meters(self):
        m = self.c.meters.read()
        self.m_in_l.set_value(m["input_l"]); self.m_in_r.set_value(m["input_r"])
        self.m_agc.set_value(m["agc_gain"])
        for bar, v in zip(self.m_lev, m["lev_gain"]):
            bar.set_value(v)
        for bar, v in zip(self.m_cmp, m["comp_gr"]):
            bar.set_value(v)
        self.m_hf.set_value(m["hf_gr"]); self.m_wb.set_value(m["wb_gr"])
        self.m_out_l.set_value(m["out_l"]); self.m_out_r.set_value(m["out_r"])
        self.m_mpx.set_value(m["mpx_pct"])
        dbr = m.get("bs412_dbr")
        if dbr is not None and dbr > -90:
            gr = m.get("bs412_gr", 0.0)
            self.bs412_lbl.setText(f"{dbr:+.1f} dBr" +
                                   (f"  GR {gr:.1f}" if gr > 0.05 else ""))
        else:
            self.bs412_lbl.setText("-- dBr")
        if m["running"]:
            mode = "MPX 192k" if self.outmode.currentIndex() == 1 else "Stereo 48k"
            dev = m.get("devices", "")
            self.status.setText(
                f"{dev}  ·  {mode} · {m['latency_ms']:.0f} ms · "
                f"xruns {m['underruns']}")
            if self.c.engine.error:
                self.c.stop()
                self.start_btn.setChecked(False)
                self.start_btn.setText("START")
                QMessageBox.critical(self, "Audio engine",
                                     f"Stream error:\n{self.c.engine.error}")
                self.c.engine.error = None
        else:
            self.status.setText("STOPPED")
            if self.start_btn.isChecked() and not self.c.engine.running:
                self.start_btn.setChecked(False)
                self.start_btn.setText("START")

    def _poll_rds(self):
        p = self.c.current_params()
        # live now-playing file: "RT=..." / "PS=..." lines, or plain text -> RT
        if p["rds_watch_file"] and p["rds_file"]:
            try:
                f = Path(p["rds_file"])
                mt = f.stat().st_mtime
                if mt != self._rds_mtime:
                    self._rds_mtime = mt
                    txt = f.read_text(encoding="utf-8", errors="replace").strip()
                    rt, ps = None, None
                    for line in txt.splitlines():
                        if line.upper().startswith("RT="):
                            rt = line[3:].strip()
                        elif line.upper().startswith("PS="):
                            ps = line[3:].strip()
                    if rt is None and ps is None:
                        rt = txt.splitlines()[0] if txt else None
                    if rt:
                        self.c.rds_update(rt=rt)
                        self.rds_rt.setText(rt[:64])
                        if p["rds_dynamic_ps"]:
                            self._ps_chunks = [rt[i:i + 8]
                                               for i in range(0, len(rt), 8)] or ["OMEGAFM"]
                            self._ps_i = 0
                    if ps:
                        self.c.rds_update(ps=ps)
                        self.rds_ps.setText(ps[:8])
            except OSError:
                pass
        # dynamic PS rotation (8-char chunks of RT every 2 s)
        if p["rds_dynamic_ps"] and self._ps_chunks:
            if int(time.monotonic()) % 2 == 0:
                self.c.rds_update(ps=self._ps_chunks[self._ps_i])
                self._ps_i = (self._ps_i + 1) % len(self._ps_chunks)


# =========================================================================== #
#  Advanced window (per-stage / per-band engineering controls)
# =========================================================================== #

ADV_W, ADV_H = 1440, 884
BAND_NAMES = ("LO", "LM", "HM", "HI")


class AdvancedPanel(QWidget):
    """All the deep controls: attacks, releases, thresholds, ratios,
    crossover-band targets, limiter time constants, EQ freqs/Qs.

    Lives in its own ScalableView window so there is room for everything;
    scales exactly like the main panel."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        self.setObjectName("panel")
        self.setFixedSize(ADV_W, ADV_H)
        self.setStyleSheet(QSS)
        self.knobs = {}
        self._building = True
        self._build()
        self._building = False
        self.sync()

    # ------------------------------------------------------------------ knobs
    def _knob(self, label, key, lo, hi, fmt="{:+.1f}", suf="dB", idx=None):
        k = Knob(label, lo, hi, lo, fmt, suf)
        k.valueChanged.connect(lambda v, kk=key, ii=idx: self._push(kk, ii, v))
        self.knobs[(key, idx)] = k
        return k

    def _push(self, key, idx, v):
        if self._building:
            return
        if idx is None:
            self.c.set_trim(**{key: v})
        else:
            arr = list(self.c.current_params()[key])
            arr[idx] = v
            self.c.set_trim(**{key: arr})

    # ------------------------------------------------------------------ build
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(8)

        head = QHBoxLayout()
        t = QLabel("ADVANCED PROCESSING")
        t.setProperty("cls", "brand")
        st = QLabel("PER-STAGE TIME CONSTANTS · THRESHOLDS · CROSSOVER BANDS")
        st.setProperty("cls", "brandsub")
        tv = QVBoxLayout(); tv.setSpacing(0)
        tv.addWidget(t); tv.addWidget(st)
        head.addLayout(tv); head.addStretch(1)
        hint = QLabel("drag ↑↓ · scroll · double-click = center")
        hint.setProperty("cls", "status")
        head.addWidget(hint)
        root.addLayout(head)
        root.addWidget(_rule())

        # ---- row 1: AGC | WB limiter | HF limiter | Final
        r1 = QHBoxLayout(); r1.setSpacing(8)
        gb = QGroupBox("AGC  ·  GAIN RIDE")
        h = QHBoxLayout(gb); h.setSpacing(2)
        h.addWidget(self._knob("TARGET", "agc_target_db", -30, -8))
        h.addWidget(self._knob("RANGE", "agc_range_db", 0, 15, "{:.1f}"))
        h.addWidget(self._knob("ATTACK", "agc_attack_db_s", 0.5, 8, "{:.1f}", "dB/s"))
        h.addWidget(self._knob("RELEASE", "agc_release_db_s", 0.2, 4, "{:.1f}", "dB/s"))
        h.addWidget(self._knob("GATE", "agc_gate_db", -70, -30))
        h.addWidget(self._knob("INTEGR", "agc_integration_ms", 100, 1000, "{:.0f}", "ms"))
        h.addWidget(self._knob("WINDOW", "agc_window_db", 0, 4, "{:.1f}"))
        r1.addWidget(gb, stretch=7)

        gb = QGroupBox("WB LIMITER  ·  RIDING RELEASE")
        h = QHBoxLayout(gb); h.setSpacing(2)
        h.addWidget(self._knob("CEILING", "wb_lim_ceiling_db", -3, 0))
        h.addWidget(self._knob("FAST REL", "wb_lim_release_db_s", 30, 300, "{:.0f}", "dB/s"))
        h.addWidget(self._knob("PLATFORM", "wb_platform_ms", 100, 3000, "{:.0f}", "ms"))
        r1.addWidget(gb, stretch=3)

        gb = QGroupBox("HF LIMITER")
        h = QHBoxLayout(gb); h.setSpacing(2)
        h.addWidget(self._knob("SPLIT", "hf_split_hz", 2000, 9000, "{:.0f}", "Hz"))
        h.addWidget(self._knob("RELEASE", "hf_lim_release_ms", 20, 300, "{:.0f}", "ms"))
        r1.addWidget(gb, stretch=2)

        gb = QGroupBox("FINAL")
        h = QHBoxLayout(gb); h.setSpacing(2)
        h.addWidget(self._knob("CLIP KNEE", "clip_knee", 0, 0.5, "{:.2f}", ""))
        h.addWidget(self._knob("BASS HARM", "bass_harmonics", 0, 1, "{:.2f}", ""))
        r1.addWidget(gb, stretch=2)
        root.addLayout(r1)

        # ---- row 2: leveler | EQ
        r2 = QHBoxLayout(); r2.setSpacing(8)
        gb = QGroupBox("4-BAND LEVELER  ·  SPECTRUM CONSISTENCY")
        h = QHBoxLayout(gb); h.setSpacing(2)
        for i, n in enumerate(BAND_NAMES):
            h.addWidget(self._knob(f"{n} TGT", "lev_targets_db", -40, -12, idx=i))
        h.addWidget(self._knob("RANGE", "lev_range_db", 0, 8, "{:.1f}"))
        h.addWidget(self._knob("SPEED", "lev_rate_db_s", 0.3, 4, "{:.1f}", "dB/s"))
        h.addWidget(self._knob("GATE", "lev_gate_db", -70, -35))
        h.addWidget(self._knob("INTEGR", "lev_integration_ms", 100, 1000, "{:.0f}", "ms"))
        h.addWidget(self._knob("WINDOW", "lev_window_db", 0, 4, "{:.1f}"))
        h.addWidget(self._knob("COUPLE", "lev_couple_db", 0, 7, "{:.1f}"))
        r2.addWidget(gb, stretch=10)

        gb = QGroupBox("PARAMETRIC EQ  ·  FREQ / Q")
        grid = QGridLayout(gb); grid.setSpacing(2)
        franges = [(40, 250), (150, 900), (800, 4500), (3000, 14000)]
        for i, (flo, fhi) in enumerate(franges):
            grid.addWidget(self._knob(f"F{i+1}", "eq_freqs", flo, fhi,
                                      "{:.0f}", "Hz", i), 0, i)
            grid.addWidget(self._knob(f"Q{i+1}", "eq_qs", 0.4, 4,
                                      "{:.2f}", "", i), 1, i)
        r2.addWidget(gb, stretch=4)
        root.addLayout(r2)

        # ---- row 3: compressor grid (bands as rows)
        gb = QGroupBox("4-BAND COMPRESSOR  ·  THE SIGNATURE")
        grid = QGridLayout(gb)
        grid.setHorizontalSpacing(2); grid.setVerticalSpacing(0)
        for bi, n in enumerate(BAND_NAMES):
            lbl = QLabel(n); lbl.setProperty("cls", "section")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(40)
            grid.addWidget(lbl, bi, 0)
            grid.addWidget(self._knob("THRESH", "comp_thresholds_db",
                                      -40, -10, idx=bi), bi, 1)
            grid.addWidget(self._knob("RATIO", "comp_ratios",
                                      1.2, 8, "{:.1f}", ":1", idx=bi), bi, 2)
            grid.addWidget(self._knob("ATTACK", "comp_attack_ms",
                                      1, 40, "{:.0f}", "ms", idx=bi), bi, 3)
            grid.addWidget(self._knob("RELEASE", "comp_release_ms",
                                      40, 700, "{:.0f}", "ms", idx=bi), bi, 4)
            grid.addWidget(self._knob("MAKEUP", "comp_makeup_db",
                                      0, 8, "{:+.1f}", "dB", idx=bi), bi, 5)
        grid.addWidget(self._knob("KNEE", "comp_knee_db", 0, 12, "{:.1f}"),
                       0, 6, 2, 1)
        grid.addWidget(self._knob("PLATFORM", "comp_platform_ms",
                                  200, 2000, "{:.0f}", "ms"), 2, 6, 2, 1)
        grid.setColumnStretch(0, 0)
        for col in range(1, 7):
            grid.setColumnStretch(col, 3)
        root.addWidget(gb, stretch=1)

    # ------------------------------------------------------------------ sync
    def sync(self):
        self._building = True
        p = self.c.current_params()
        for (key, idx), knob in self.knobs.items():
            val = p[key][idx] if idx is not None else p[key]
            knob.set_value(val)
        self._building = False


# =========================================================================== #
#  Plugins windows
# =========================================================================== #

PLG_W, PLG_H = 780, 520


class PluginDetailPanel(QWidget):
    """Auto-generated control/metering window for one active plugin:
    knobs from its manifest params, LED bars from its manifest meters,
    values pulled live from the running instance."""

    def __init__(self, host, lp, parent=None):
        super().__init__(parent)
        self.host = host
        self.lp = lp
        self.setObjectName("panel")
        self.setStyleSheet(QSS)
        nk, nm = len(lp.params_spec), len(lp.meters_spec)
        self.setFixedSize(max(3, nk + nm) * 118 + 60, 336)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 10)
        root.setSpacing(6)
        t = QLabel(lp.name.upper())
        t.setProperty("cls", "brand")
        st = QLabel(f"v{lp.version}   ·   insert: {lp.insert}")
        st.setProperty("cls", "brandsub")
        root.addWidget(t); root.addWidget(st)
        root.addWidget(_rule())

        row = QHBoxLayout(); row.setSpacing(4)
        self.knobs = {}
        for ps in lp.params_spec:
            k = Knob(ps.get("label", ps["key"]).upper(),
                     float(ps.get("min", 0.0)), float(ps.get("max", 1.0)),
                     float(lp.params.get(ps["key"], ps.get("default", 0.0))),
                     ps.get("fmt", "{:.2f}"), ps.get("suffix", ""))
            k.valueChanged.connect(
                lambda v, key=ps["key"]: self.host.set_param(self.lp.pid, key, v))
            self.knobs[ps["key"]] = k
            row.addWidget(k)
        self.bars = {}
        for ms in lp.meters_spec:
            b = LEDBar(ms.get("label", ms["key"]).upper(),
                       ms.get("mode", "level"),
                       float(ms.get("lo", -42.0)), float(ms.get("hi", 3.0)),
                       segments=18, unit=str(ms.get("unit", "db")))
            self.bars[ms["key"]] = b
            row.addWidget(b)
        root.addLayout(row, stretch=1)

        self.stat = QLabel("")
        self.stat.setProperty("cls", "status")
        root.addWidget(self.stat)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(66)

    def _poll(self):
        inst = self.lp.instance
        if inst is None:
            self.stat.setText("AUTO-BYPASSED (error) - see PLUGINS list")
            for b in self.bars.values():
                b.set_value(b.lo if b.mode == "level" else 0.0)
            return
        self.stat.setText("active")
        m = getattr(inst, "meters", {}) or {}
        for key, b in self.bars.items():
            if key in m:
                b.set_value(float(m[key]))


class PluginsPanel(QWidget):
    """Discovered-plugin list: tick to insert live into the chain,
    double-click an active plugin for its controls and meters."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        self.host = controller.plugin_host
        self.setObjectName("panel")
        self.setFixedSize(PLG_W, PLG_H)
        self.setStyleSheet(QSS)
        self._child_views = []

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 10, 16, 10)
        root.setSpacing(6)
        t = QLabel("DSP PLUGINS")
        t.setProperty("cls", "brand")
        st = QLabel("MODULAR EXTENSIONS  ·  DROP .py FILES IN  plugins/  ·  "
                    "TICK = INSERT LIVE  ·  DOUBLE-CLICK = CONTROLS")
        st.setProperty("cls", "brandsub")
        root.addWidget(t); root.addWidget(st)
        root.addWidget(_rule())

        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(6)
        rows_holder = QWidget()
        rows_holder.setLayout(self.rows_box)
        root.addWidget(rows_holder, stretch=1)

        h = QHBoxLayout()
        rescan = QPushButton("RESCAN FOLDER")
        rescan.clicked.connect(self.refresh)
        h.addWidget(rescan)
        self.err = QLabel("")
        self.err.setProperty("cls", "status")
        h.addWidget(self.err, stretch=1)
        root.addLayout(h)
        self.refresh(scan=False)

    # ------------------------------------------------------------------ list
    def refresh(self, scan=True):
        if scan:
            self.host.scan()
        while self.rows_box.count():
            it = self.rows_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for lp in self.host.plugins.values():
            row = QWidget()
            row.setStyleSheet(f"background: {INSET}; border-radius: 6px;")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(10, 6, 10, 6)
            cb = QCheckBox()
            cb.setChecked(lp.enabled)
            cb.toggled.connect(lambda on, pid=lp.pid: self._toggled(pid, on))
            hl.addWidget(cb)
            name = QLabel(f"{lp.name}   <span style='color:{DIM}'>v{lp.version}"
                          f"   ·   {lp.insert}</span>")
            name.setTextFormat(Qt.RichText)
            hl.addWidget(name, stretch=1)
            stat = QLabel(lp.status)
            col = RED if lp.error else (GREEN if lp.enabled else DIM)
            stat.setStyleSheet(f"color: {col}; font-weight: 700;"
                               f" letter-spacing: 1px; font-size: 11px;")
            hl.addWidget(stat)
            btn = QPushButton("CONTROLS")
            btn.setEnabled(lp.enabled)
            btn.clicked.connect(lambda _, pid=lp.pid: self._open_detail(pid))
            hl.addWidget(btn)
            self.rows_box.addWidget(row)
        self.rows_box.addStretch(1)
        n_err = len(self.host.scan_errors)
        self.err.setText(f"{len(self.host.plugins)} plugin(s) found"
                         + (f"  ·  {n_err} file(s) skipped" if n_err else ""))

    def _toggled(self, pid, on):
        self.host.set_enabled(pid, on)
        self.refresh(scan=False)

    def _open_detail(self, pid):
        lp = self.host.plugins.get(pid)
        if lp is None or not lp.enabled:
            return
        panel = PluginDetailPanel(self.host, lp)
        view = ScalableView(panel, title=f"OmegaFM · {lp.name}")
        view.resize(panel.width(), panel.height())
        self._child_views.append(view)
        view.show()



class ScalableView(QGraphicsView):
    """Hosts a fixed-size panel and scales it (aspect-locked) to ANY
    window size - the whole interface grows or shrinks, never clips."""

    def __init__(self, panel: QWidget, parent=None,
                 title: str = f"OmegaFM v{__version__}"):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.proxy = self.scene_.addWidget(panel)
        self.setScene(self.scene_)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform |
                            QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor(BG))
        self.setFrameShape(QFrame.NoFrame)
        self.setWindowTitle(title)
        self.resize(1280, 720)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self.scene_.itemsBoundingRect(), Qt.KeepAspectRatio)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.fitInView(self.scene_.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, ev):
        wdg = self.proxy.widget()
        for v in list(getattr(wdg, "_child_views", []) or []):
            try:
                v.close()
            except Exception:
                pass
        adv = getattr(wdg, "_adv_view", None)
        if adv is not None:
            adv.close()
        super().closeEvent(ev)
