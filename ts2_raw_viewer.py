#!/usr/bin/env python3
"""
Live view of the Vantrue TS2 temperature data with a manually chosen display band.

The temperature rows are shown as delivered: the band dragged on the histogram
maps linearly to black..white and nothing else is applied. The histogram counts
every raw code of the frame; dashed lines mark the current gain's calibrated
range. The IR brightness rows, already processed by the camera, are shown
beside it, stretched from their frame min to max. The readout gives the raw
value, the camera's 1/64 K conversion and the value the Vantrue app would
display (vendor_tempcorr; n/a without the 'vendor' extra or the vendor files).

Run: uv run --extra gui --extra vendor ts2_raw_viewer.py
"""
import pathlib
import queue
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import p3_camera as p3
import vendor_tempcorr as vt

HERE = pathlib.Path(__file__).parent
SWITCH_WAIT_S = 6.0
STREAM_SETTLE_S = 2.0


class CameraThread(QtCore.QThread):
    """
    Owns the camera. Reads frames continuously and runs queued gain switches
    between frames, so control transfers never overlap a bulk read.
    """

    frame = QtCore.Signal(bytes, int, int)
    gain = QtCore.Signal(object, bool)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gain_requests = queue.Queue()
        self.running = True

    def run(self):
        cam = p3.P3Camera(config=self.cfg)
        cam.connect()
        cam.start_streaming()
        self.gain.emit(cam.gain_mode, False)
        switch_until = 0.0

        try:
            while self.running:
                try:
                    raw = cam.read_frame()
                except p3.FrameMarkerMismatchError:
                    continue
                self.frame.emit(raw, cam.stats.frames_read, cam.stats.frames_dropped)

                now = time.monotonic()
                if switch_until and now >= switch_until:
                    switch_until = 0.0
                    try:
                        cam.get_gain_mode()
                    except RuntimeError as e:
                        print(f"gain read after switch: {e}")
                    self.gain.emit(cam.gain_mode, False)

                try:
                    target = self.gain_requests.get_nowait()
                except queue.Empty:
                    continue
                cam.set_gain_mode(target)
                switch_until = now + SWITCH_WAIT_S
                self.gain.emit(target, True)
        finally:
            cam.stop_streaming()


class Viewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TS2 raw")
        self.cfg = p3.get_model_config(p3.Model.TS2)
        self.params = dict(vt.APP_DEFAULTS)
        self.gain_mode = None
        self.switching = False
        self.settled = False
        self.raw = self.ir = self.th = None
        self.mouse = None

        try:
            self.vc, self.tables = vt.VendorTempCorrection(), vt.load_tables()
        except (OSError, ImportError) as e:
            print(f"app column disabled: {e}")
            self.vc = self.tables = None

        self._build_ui()

        self.camera = CameraThread(self.cfg)
        self.camera.frame.connect(self.on_frame)
        self.camera.gain.connect(self.on_gain)
        self.camera.start()

    # ---- layout --------------------------------------------------------------

    def _build_ui(self):
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.images = pg.GraphicsLayoutWidget()
        self.images.addLabel("IR brightness (camera-processed, frame min..max)", row=0, col=0)
        self.images.addLabel("Temperature (band -> black..white)", row=0, col=1)
        self.ir_item, self.t_item = pg.ImageItem(), pg.ImageItem()
        for col, item in enumerate((self.ir_item, self.t_item)):
            vb = self.images.addViewBox(row=1, col=col, lockAspect=True, invertY=True, enableMouse=False)
            vb.addItem(item)
        self.images.scene().sigMouseMoved.connect(self.on_mouse)

        self.hist = pg.PlotWidget()
        self.hist.setLogMode(y=True)
        self.hist.setLabel("bottom", "temperature, camera conversion (C); one bin per raw code")
        self.hist.setLabel("left", "pixels + 1")
        self.hist.setMouseEnabled(x=True, y=False)
        self.hist_curve = self.hist.plot(stepMode="center")
        self.cal_lines = [pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=QtCore.Qt.PenStyle.DashLine)) for _ in range(2)]
        for line in self.cal_lines:
            self.hist.addItem(line)
        self.band = pg.LinearRegionItem(values=(0.0, 1.0))
        self.band.sigRegionChanged.connect(self.apply_band)
        self.hist.addItem(self.band)

        self.gain_button = QtWidgets.QPushButton("Gain: ...")
        self.gain_button.setEnabled(False)
        self.gain_button.clicked.connect(self.switch_gain)
        auto_button = QtWidgets.QPushButton("Band = frame min..max")
        auto_button.clicked.connect(self.auto_band)
        full_button = QtWidgets.QPushButton("Show full range")
        full_button.clicked.connect(self.show_full_range)
        zoom_button = QtWidgets.QPushButton("Zoom to data")
        zoom_button.clicked.connect(self.zoom_to_data)
        self.follow = QtWidgets.QCheckBox("Follow frame min..max")
        dump_button = QtWidgets.QPushButton("Dump frame")
        dump_button.clicked.connect(self.dump)

        controls = QtWidgets.QHBoxLayout()
        for w in (self.gain_button, auto_button, full_button, zoom_button, self.follow, dump_button):
            controls.addWidget(w)
        controls.addStretch()
        for name, label, lo, hi, step, decimals in (
            ("ems", "emissivity", 0.01, 1.0, 0.01, 2),
            ("dist", "distance m", 0.0, 1000.0, 0.25, 2),
            ("ta", "ambient = reflected C", -40.0, 100.0, 1.0, 1),
        ):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setValue(self.params[name])
            box.valueChanged.connect(lambda v, name=name: self.set_param(name, v))
            controls.addWidget(QtWidgets.QLabel(label))
            controls.addWidget(box)
        controls.addWidget(QtWidgets.QLabel(f"humidity {self.params['hum']:.2f}"))

        self.stats = QtWidgets.QLabel()
        self.readout = QtWidgets.QLabel()
        for label in (self.stats, self.readout):
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.images, stretch=3)
        layout.addWidget(self.hist, stretch=2)
        layout.addLayout(controls)
        layout.addWidget(self.stats)
        layout.addWidget(self.readout)
        self.resize(1300, 900)

    # ---- camera events -------------------------------------------------------

    def on_frame(self, raw, frames, dropped):
        first = self.th is None
        self.raw = raw
        full = p3.extract_full_frame(raw, self.cfg)
        self.ir = full[: self.cfg.ir_row_end]
        self.th = full[self.cfg.thermal_row_start : self.cfg.thermal_row_end]

        self.ir_item.setImage(self.ir, autoLevels=True)
        self.t_item.setImage(p3.raw_to_celsius(self.th), autoLevels=False)
        if first:
            self.auto_band()
            self.show_full_range()
        elif self.follow.isChecked():
            self.auto_band()
        else:
            self.apply_band()
        self.update_histogram()

        t_lo, t_hi = int(self.th.min()), int(self.th.max())
        self.stats.setText(
            f"IR u16 [{int(self.ir.min())}..{int(self.ir.max())}]   "
            f"T u16 [{t_lo}..{t_hi}] = [{p3.raw_to_celsius(t_lo):.2f}..{p3.raw_to_celsius(t_hi):.2f}] C   "
            f"app [{self.app_text(t_lo)}..{self.app_text(t_hi)}]   frames {frames}  dropped {dropped}"
        )
        self.update_readout()

    def on_gain(self, mode, switching):
        self.gain_mode = mode
        self.switching = switching
        if not switching and not self.settled:
            QtCore.QTimer.singleShot(int(STREAM_SETTLE_S * 1000), self.on_settled)
        if not switching:
            lo, hi = vt.GAIN_RANGE_C[int(mode)]
            for line, x in zip(self.cal_lines, (lo, hi)):
                line.setValue(x)
            if self.th is not None:
                self.auto_band()
                self.show_full_range()
        self.update_gain_button()

    def on_settled(self):
        self.settled = True
        self.update_gain_button()

    # ---- controls ------------------------------------------------------------

    def update_gain_button(self):
        name = self.gain_mode.name if self.gain_mode is not None else "..."
        if self.switching:
            self.gain_button.setText(f"Gain: switching to {name}...")
        else:
            other = "LOW" if self.gain_mode == p3.GainMode.HIGH else "HIGH"
            self.gain_button.setText(f"Gain: {name} (switch to {other})")
        self.gain_button.setEnabled(self.settled and not self.switching and self.gain_mode is not None)

    def switch_gain(self):
        target = p3.GainMode.LOW if self.gain_mode == p3.GainMode.HIGH else p3.GainMode.HIGH
        self.gain_button.setEnabled(False)
        self.camera.gain_requests.put(target)

    def auto_band(self):
        if self.th is None:
            return
        self.band.setRegion((float(p3.raw_to_celsius(self.th.min())), float(p3.raw_to_celsius(self.th.max()))))

    def show_full_range(self):
        if self.th is None or self.gain_mode is None:
            return
        lo, hi = vt.GAIN_RANGE_C[int(self.gain_mode)]
        data_lo, data_hi = float(p3.raw_to_celsius(self.th.min())), float(p3.raw_to_celsius(self.th.max()))
        self.hist.setXRange(min(lo, data_lo), max(hi, data_hi), padding=0.02)

    def zoom_to_data(self):
        if self.th is None:
            return
        lo, hi = float(p3.raw_to_celsius(self.th.min())), float(p3.raw_to_celsius(self.th.max()))
        margin = max(hi - lo, 0.5)
        self.hist.setXRange(lo - margin, hi + margin, padding=0)

    def apply_band(self):
        self.t_item.setLevels(self.band.getRegion())

    def set_param(self, name, value):
        self.params[name] = value
        if name == "ta":
            self.params["tu"] = value

    def dump(self):
        if self.raw is None:
            return
        captures = HERE / "captures"
        captures.mkdir(exist_ok=True)
        path = captures / f"ts2_{time.strftime('%Y%m%d_%H%M%S')}.bin"
        path.write_bytes(self.raw)
        print(f"saved {path} ({len(self.raw)} B)")
        full = p3.extract_full_frame(self.raw, self.cfg)
        for r in range(self.cfg.ir_row_end, self.cfg.thermal_row_start):
            print(f"meta row {r}: " + " ".join(f"{v:04x}" for v in full[r]))

    # ---- histogram and readout -----------------------------------------------

    def update_histogram(self):
        lo_c, hi_c = vt.GAIN_RANGE_C[int(self.gain_mode)] if self.gain_mode is not None else (0.0, 0.0)
        lo = min(p3.celsius_to_raw(lo_c), int(self.th.min()))
        hi = max(p3.celsius_to_raw(hi_c), int(self.th.max()))
        counts = np.bincount((self.th.ravel().astype(np.int64) - lo), minlength=hi - lo + 1)
        edges = (np.arange(lo, hi + 2, dtype=np.float64) - 0.5) / p3.TEMP_SCALE - p3.KELVIN_OFFSET
        self.hist_curve.setData(edges, counts + 1)

    def app_text(self, raw):
        if self.vc is None:
            return "n/a"
        if self.switching or self.gain_mode is None:
            return "switching"
        c = vt.app_display_c(self.vc, self.tables, float(p3.raw_to_celsius(raw)), int(self.gain_mode), **self.params)
        return f"<{vt.LOW_GAIN_FLOOR_C:.0f} C" if c is None else f"{c:.2f} C"

    def on_mouse(self, pos):
        self.mouse = pos
        self.update_readout()

    def update_readout(self):
        if self.mouse is None or self.th is None:
            return
        for item, data in ((self.ir_item, self.ir), (self.t_item, self.th)):
            p = item.mapFromScene(self.mouse)
            row, col = int(p.y()), int(p.x())
            if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
                continue
            v = int(data[row, col])
            if item is self.ir_item:
                self.readout.setText(f"IR  r{row:3d} c{col:3d}  raw 0x{v:04x} = {v}")
            else:
                self.readout.setText(
                    f"T   r{row:3d} c{col:3d}  raw {v}  = {p3.raw_to_kelvin(v):.2f} K"
                    f" = {p3.raw_to_celsius(v):.2f} C   app: {self.app_text(v)}"
                )
            return

    def closeEvent(self, event):
        self.camera.running = False
        self.camera.wait(3000)
        super().closeEvent(event)


def main():
    app = pg.mkQApp("TS2 raw")
    viewer = Viewer()
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
