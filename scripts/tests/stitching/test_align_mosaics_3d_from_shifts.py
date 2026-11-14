#!/usr/bin/env python3


def test_help(script_runner):
    ret = script_runner.run(["linum-align-mosaics-3d-from-shifts", "--help"])
    assert ret.success


def _load_script_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "stitching" / "linum_align_mosaics_3d_from_shifts.py"
    spec = importlib.util.spec_from_file_location("align_mosaics_3d_from_shifts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shift_overrides_replace_the_listed_transitions_only():
    import pandas as pd
    import pytest

    module = _load_script_module()
    shifts = pd.DataFrame(
        {
            "fixed_id": [0.0, 1.0, 2.0],
            "moving_id": [1.0, 2.0, 3.0],
            "x_shift": [10.0, 20.0, -30.0],
            "y_shift": [5.0, 0.0, 15.0],
            "x_shift_mm": [0.1, 0.2, -0.3],
            "y_shift_mm": [0.05, 0.0, 0.15],
            "reliable": [1, 0, 0],
        }
    )
    overrides = pd.DataFrame({"fixed_id": [1], "moving_id": [2], "x_shift_mm": [-0.04], "y_shift_mm": [0.12]})
    assert module.apply_shift_overrides(shifts, overrides) == 1
    row = shifts.iloc[1]
    assert row["x_shift_mm"] == pytest.approx(-0.04)
    assert row["y_shift_mm"] == pytest.approx(0.12)
    # Pixel columns follow at the file's own 100 px per mm.
    assert row["x_shift"] == pytest.approx(-4.0)
    assert row["y_shift"] == pytest.approx(12.0)
    # Marked reliable, so the image-based estimate is not made again.
    assert int(row["reliable"]) == 1
    # The other unreliable transition is left for the refinement.
    assert int(shifts.iloc[2]["reliable"]) == 0
    assert shifts.iloc[2]["x_shift_mm"] == pytest.approx(-0.3)
