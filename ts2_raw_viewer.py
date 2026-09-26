#!/usr/bin/env python3
"""
Unprocessed live view of the Vantrue TS2: the IR and temperature rows of each
frame exactly as delivered, each mapped linearly from its own frame min to max,
with a per-pixel readout of the raw u16, the camera's 1/64 K conversion and the
value the vendor app would display (emulated vendor correction, vendor_tempcorr;
shown as n/a when vendor/ lacks the vendor files).

Keys: q quit, d dump the frame (marker + pixel bytes) to captures/ and print
the metadata rows, g toggle gain, e/E emissivity -/+, r/R distance -/+,
t/T ambient (= reflected) temperature -/+.
"""
import os
import pathlib
import time

# The Qt bundled with the opencv-python wheel ships only the xcb platform plugin.
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import p3_camera as p3  # noqa: E402
import vendor_tempcorr as vt  # noqa: E402

HERE = pathlib.Path(__file__).parent

ZOOM = 3
BAR_H = 64
WINDOW = "TS2 raw"
SWITCH_WAIT_S = 6.0
STREAM_SETTLE_S = 2.0
PARAM_KEYS = {
    ord("e"): ("ems", -0.01, 0.01, 1.0),
    ord("E"): ("ems", +0.01, 0.01, 1.0),
    ord("r"): ("dist", -0.25, 0.0, 1000.0),
    ord("R"): ("dist", +0.25, 0.0, 1000.0),
    ord("t"): ("ta", -1.0, -40.0, 100.0),
    ord("T"): ("ta", +1.0, -40.0, 100.0),
}


def to_u8(region):
    return cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def text(img, s, row, color=(255, 255, 255)):
    cv2.putText(img, s, (6, 17 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def readout(pos, ir, th, vendor):
    x, y = pos
    col = (x // ZOOM) % ir.shape[1]
    row = y // ZOOM
    if row >= ir.shape[0]:
        return ""

    if x < ir.shape[1] * ZOOM:
        v = int(ir[row, col])
        return f"IR  r{row:3d} c{col:3d}  raw 0x{v:04x} = {v}"

    v = int(th[row, col])
    return (f"T   r{row:3d} c{col:3d}  raw {v}  = {p3.raw_to_kelvin(v):.2f} K"
            f" = {p3.raw_to_celsius(v):.2f} C   app: {vendor(v)}")


def window_closed():
    # The Qt backend raises instead of returning 0 once the window is destroyed.
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    cfg = p3.get_model_config(p3.Model.TS2)
    try:
        vc, tables = vt.VendorTempCorrection(), vt.load_tables()
    except (OSError, ImportError) as e:
        print(f"app column disabled: {e}")
        vc = tables = None
    params = dict(vt.APP_DEFAULTS)
    switch_until = 0.0

    cam = p3.P3Camera(config=cfg)
    cam.connect()
    cam.start_streaming()
    stream_start = time.monotonic()

    def vendor(raw):
        if vc is None:
            return "n/a"
        if switch_until:
            return "switching"
        c = vt.app_display_c(vc, tables, float(p3.raw_to_celsius(raw)), int(cam.gain_mode), **params)
        return f"<{vt.LOW_GAIN_FLOOR_C:.0f} C" if c is None else f"{c:.2f} C"

    mouse = [0, 0]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, flags, arg: mouse.__setitem__(slice(None), (x, y)))
    captures = HERE / "captures"
    captures.mkdir(exist_ok=True)

    try:
        while True:
            try:
                raw = cam.read_frame()
            except p3.FrameMarkerMismatchError:
                continue

            now = time.monotonic()
            if switch_until and now >= switch_until:
                switch_until = 0.0
                try:
                    cam.get_gain_mode()
                except RuntimeError as e:
                    print(f"gain read after switch: {e}")

            full = p3.extract_full_frame(raw, cfg)
            ir = full[: cfg.ir_row_end]
            th = full[cfg.thermal_row_start : cfg.thermal_row_end]

            img = cv2.resize(np.hstack([to_u8(ir), to_u8(th)]), None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            bar = np.zeros((BAR_H, img.shape[1], 3), np.uint8)
            s = cam.stats
            gain = f"switching to {cam.gain_mode.name}" if switch_until else cam.gain_mode.name
            text(bar, f"IR u16 [{ir.min()}..{ir.max()}]   T u16 [{th.min()}..{th.max()}]"
                      f" = [{p3.raw_to_celsius(th.min()):.2f}..{p3.raw_to_celsius(th.max()):.2f}] C"
                      f"   app [{vendor(th.min())}..{vendor(th.max())}]"
                      f"   frames {s.frames_read}  dropped {s.frames_dropped}", 0)
            text(bar, f"gain {gain}   ems {params['ems']:.2f}  dist {params['dist']:.2f} m"
                      f"  ambient=reflected {params['ta']:.0f} C  hum {params['hum']:.2f}", 1, (160, 255, 160))
            text(bar, readout(mouse, ir, th, vendor), 2, (0, 255, 255))
            cv2.imshow(WINDOW, np.vstack([img, bar]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_closed():
                break

            if key in PARAM_KEYS:
                name, step, lo, hi = PARAM_KEYS[key]
                params[name] = round(min(hi, max(lo, params[name] + step)), 2)
                params["tu"] = params["ta"]

            if key == ord("g"):
                if switch_until or now - stream_start < STREAM_SETTLE_S:
                    print("gain switch ignored: stream settling or switch in progress")
                else:
                    target = p3.GainMode.LOW if cam.gain_mode == p3.GainMode.HIGH else p3.GainMode.HIGH
                    cam.set_gain_mode(target)
                    switch_until = now + SWITCH_WAIT_S

            if key == ord("d"):
                path = captures / f"ts2_{time.strftime('%Y%m%d_%H%M%S')}.bin"
                path.write_bytes(raw)
                print(f"saved {path} ({len(raw)} B)")
                for r in range(cfg.ir_row_end, cfg.thermal_row_start):
                    print(f"meta row {r}: " + " ".join(f"{v:04x}" for v in full[r]))
    finally:
        cam.stop_streaming()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
