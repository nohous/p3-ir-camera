"""Tests for vendor_tempcorr.py; skipped unless vendor/ holds the vendor files."""

from __future__ import annotations

import pytest

vt = pytest.importorskip("vendor_tempcorr")

if not (vt.SO_PATH.exists() and all((vt.TABLE_DIR / name).exists() for name in vt.TABLES.values())):
    pytest.skip("vendor library or tau tables missing from vendor/", allow_module_level=True)


@pytest.fixture(scope="module")
def vc():
    return vt.VendorTempCorrection()


@pytest.fixture(scope="module")
def tables():
    return vt.load_tables()


def test_vendor_rejects_out_of_range_emissivity(vc, tables):
    with pytest.raises(ValueError, match="-1100"):
        vc.correct(30.0, tables[vt.GAIN_HIGH], **{**vt.APP_DEFAULTS, "ems": 1.5})


def test_app_defaults_are_identity(vc, tables):
    for t in (0.0, 22.7, 60.0, 100.0):
        assert vc.correct(t, tables[vt.GAIN_HIGH], **vt.APP_DEFAULTS) == pytest.approx(t, abs=1e-3)


def test_correction_is_monotonic(vc, tables):
    params = {**vt.APP_DEFAULTS, "ems": 0.95, "dist": 1.0}
    out = [vc.correct(t, tables[vt.GAIN_HIGH], **params) for t in (0.0, 22.7, 36.6, 60.0, 100.0)]
    assert out == sorted(out)
    assert out[-1] > 100.0


def test_app_display_rules(vc, tables):
    assert vt.app_display_c(vc, tables, 22.709, vt.GAIN_HIGH, **vt.APP_DEFAULTS) == 22.70
    assert vt.app_display_c(vc, tables, -21.004, vt.GAIN_HIGH, **vt.APP_DEFAULTS) == -21.01
    assert vt.app_display_c(vc, tables, 149.99, vt.GAIN_LOW, **vt.APP_DEFAULTS) is None
    assert vt.app_display_c(vc, tables, 300.0, vt.GAIN_LOW, **vt.APP_DEFAULTS) == 300.0
