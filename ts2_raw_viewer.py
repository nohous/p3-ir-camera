#!/usr/bin/env python3
"""
Live view of the Vantrue TS2 temperature data with a manually chosen display band
and a ring buffer of recent frames to step through.

The temperature rows are shown as delivered: the band dragged on the histogram
maps linearly to black..white on both panes and nothing else is applied. The
histogram counts every raw code of the live frame; dashed lines mark the current
gain's calibrated range. The left pane shows one frame from the ring buffer,
selected on the frame-mean strip below the images. The readout gives the raw
value, the camera's 1/64 K conversion and the value the Vantrue app would
display (vendor_tempcorr; n/a without the 'vendor' extra or the vendor files).

Keys: Space pause/resume the buffer, Left/Right step one buffered frame.

Run: uv run --extra gui --extra vendor ts2_raw_viewer.py
"""
import collections
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
BUFFER_FRAMES = 250


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
        self.raw = self.th = None
        self.mouse = None
        self.buffer = collections.deque(maxlen=BUFFER_FRAMES)
        self.paused = False
        self.sel = 0

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
        self.buf_title = self.images.addLabel("Buffer", row=0, col=0)
        self.images.addLabel("Live temperature (band -> black..white)", row=0, col=1)
        self.buf_item, self.t_item = pg.ImageItem(), pg.ImageItem()
        for col, item in enumerate((self.buf_item, self.t_item)):
            vb = self.images.addViewBox(row=1, col=col, lockAspect=True, invertY=True, enableMouse=False)
            vb.addItem(item)
        self.images.scene().sigMouseMoved.connect(self.on_mouse)

        self.strip = pg.PlotWidget()
        self.strip.setLabel("left", "frame mean (C)")
        self.strip.setLabel("bottom", "buffered frame (oldest .. newest)")
        self.strip.setMouseEnabled(x=False, y=False)
        self.strip_curve = self.strip.plot(pen="c", symbol="o", symbolSize=3)
        self.sel_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("y", width=2))
        self.sel_line.sigPositionChanged.connect(self.on_sel_line)
        self.strip.addItem(self.sel_line)

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

        self.pause_button = QtWidgets.QPushButton("Pause buffer (Space)")
        self.pause_button.clicked.connect(self.toggle_pause)
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
        for w in (self.pause_button, self.gain_button, auto_button, full_button, zoom_button, self.follow, dump_button):
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
        self.buf_info = QtWidgets.QLabel()
        self.readout = QtWidgets.QLabel()
        for label in (self.stats, self.buf_info, self.readout):
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))

        for key, fn in (
            (QtCore.Qt.Key.Key_Space, self.toggle_pause),
            (QtCore.Qt.Key.Key_Left, lambda: self.step(-1)),
            (QtCore.Qt.Key.Key_Right, lambda: self.step(+1)),
        ):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.images, stretch=4)
        layout.addWidget(self.strip, stretch=1)
        layout.addWidget(self.hist, stretch=2)
        layout.addLayout(controls)
        layout.addWidget(self.stats)
        layout.addWidget(self.buf_info)
        layout.addWidget(self.readout)
        self.resize(1300, 1000)

    # ---- camera events -------------------------------------------------------

    def on_frame(self, raw, frames, dropped):
        first = self.th is None
        self.raw = raw
        self.th = self.thermal(raw)
        self.t_item.setImage(p3.raw_to_celsius(self.th), autoLevels=False)

        if not self.paused:
            self.buffer.append((raw, frames, float(self.th.mean())))
            self.sel = len(self.buffer) - 1
            self.update_strip()
            self.show_buffered()

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
            f"live T u16 [{t_lo}..{t_hi}] = [{p3.raw_to_celsius(t_lo):.2f}..{p3.raw_to_celsius(t_hi):.2f}] C   "
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
        self.auto_band()
        self.update_gain_button()

    # ---- ring buffer ---------------------------------------------------------

    def thermal(self, raw):
        return p3.extract_full_frame(raw, self.cfg)[self.cfg.thermal_row_start : self.cfg.thermal_row_end]

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume buffer (Space)" if self.paused else "Pause buffer (Space)")
        if not self.paused:
            self.sel = len(self.buffer) - 1
            self.update_strip()
            self.show_buffered()

    def step(self, delta):
        if not self.paused:
            self.toggle_pause()
        self.sel = min(max(self.sel + delta, 0), len(self.buffer) - 1)
        self.show_buffered()

    def on_sel_line(self):
        if not self.paused or not self.buffer:
            return
        sel = min(max(int(round(self.sel_line.value())), 0), len(self.buffer) - 1)
        if sel != self.sel:
            self.sel = sel
            self.show_buffered()

    def update_strip(self):
        means = np.array([m for _, _, m in self.buffer])
        self.strip_curve.setData(np.arange(len(means)), p3.raw_to_celsius(means).astype(np.float64))

    def show_buffered(self):
        if not self.buffer:
            return
        raw, frame_no, mean = self.buffer[self.sel]
        th = self.thermal(raw)
        self.buf_item.setImage(p3.raw_to_celsius(th), autoLevels=False)
        self.buf_item.setLevels(self.band.getRegion())
        self.sel_line.blockSignals(True)
        self.sel_line.setValue(self.sel)
        self.sel_line.blockSignals(False)

        marker = p3.parse_marker(raw[: p3.MARKER_SIZE])
        info = (f"buffer {'PAUSED' if self.paused else 'live'}  frame {self.sel + 1}/{len(self.buffer)} (#{frame_no})"
                f"  mean {p3.raw_to_celsius(mean):.3f} C  cnt1 {int(marker['cnt1'][0])}  cnt3 {int(marker['cnt3'][0])}")
        if self.sel > 0:
            prev_raw, _, prev_mean = self.buffer[self.sel - 1]
            meta = p3.extract_full_frame(raw, self.cfg)[self.cfg.ir_row_end : self.cfg.thermal_row_start]
            prev_meta = p3.extract_full_frame(prev_raw, self.cfg)[self.cfg.ir_row_end : self.cfg.thermal_row_start]
            info += (f"   vs previous: mean {1000 * p3.raw_to_kelvin(mean - prev_mean):+.1f} mK"
                     f" ({mean - prev_mean:+.2f} codes), metadata words changed {int((meta != prev_meta).sum())}")
        self.buf_info.setText(info)
        self.buf_title.setText(f"Buffer frame {self.sel + 1}/{len(self.buffer)} ({'paused' if self.paused else 'live'})")

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
        levels = self.band.getRegion()
        self.t_item.setLevels(levels)
        self.buf_item.setLevels(levels)

    def set_param(self, name, value):
        self.params[name] = value
        if name == "ta":
            self.params["tu"] = value

    def dump(self):
        raw = self.buffer[self.sel][0] if self.paused and self.buffer else self.raw
        if raw is None:
            return
        captures = HERE / "captures"
        captures.mkdir(exist_ok=True)
        path = captures / f"ts2_{time.strftime('%Y%m%d_%H%M%S')}.bin"
        path.write_bytes(raw)
        print(f"saved {path} ({len(raw)} B)")
        full = p3.extract_full_frame(raw, self.cfg)
        for r in range(self.cfg.ir_row_end, self.cfg.thermal_row_start):
            print(f"meta row {r}: " + " ".join(f"{v:04x}" for v in full[r]))

    # ---- histogram and readout -----------------------------------------------

    def update_histogram(self):
        lo_c, hi_c = vt.GAIN_RANGE_C[int(self.gain_mode)] if self.gain_mode is not None else (0.0, 0.0)
        lo = min(p3.celsius_to_raw(lo_c), int(self.th.min()))
        hi = max(p3.celsius_to_raw(hi_c), int(self.th.max()))
        counts = np.bincount((self.th.ravel().astype(np.int64) - lo), minlength=hi - lo + 1)
        edges = p3.raw_to_celsius(np.arange(lo, hi + 2) - 0.5).astype(np.float64)
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
        panes = [(self.t_item, self.th, "live")]
        if self.buffer:
            panes.append((self.buf_item, self.thermal(self.buffer[self.sel][0]), "buffer"))
        for item, data, name in panes:
            p = item.mapFromScene(self.mouse)
            row, col = int(p.y()), int(p.x())
            if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
                continue
            v = int(data[row, col])
            self.readout.setText(
                f"{name:6} r{row:3d} c{col:3d}  raw {v}  = {p3.raw_to_kelvin(v):.2f} K"
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
