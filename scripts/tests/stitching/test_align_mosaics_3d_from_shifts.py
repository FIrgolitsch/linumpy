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


def _write_mosaics(folder, shapes):
    import dask.array as da
    import numpy as np

    from linumpy.io.zarr import save_omezarr

    rng = np.random.default_rng(7)
    folder.mkdir()
    for slice_id, shape in shapes.items():
        volume = rng.random(shape, dtype=np.float32) + 0.1
        save_omezarr(
            da.from_array(volume),
            folder / f"slice_z{slice_id:02d}_normalize.ome.zarr",
            (0.01, 0.01, 0.01),
            chunks=(8, 16, 16),
            n_levels=1,
        )


def test_plan_then_one_mosaic_at_a_time_matches_the_single_run(script_runner, tmp_path):
    import csv

    import numpy as np
    import pandas as pd

    from linumpy.io.zarr import read_omezarr

    mosaics = tmp_path / "mosaics"
    _write_mosaics(mosaics, {3: (4, 40, 48), 4: (4, 42, 46), 5: (4, 40, 48)})
    shifts = tmp_path / "shifts_xy.csv"
    pd.DataFrame(
        {
            "fixed_id": [3, 4],
            "moving_id": [4, 5],
            "x_shift": [3.3, -7.25],
            "y_shift": [-2.6, 4.1],
            "x_shift_mm": [0.033, -0.0725],
            "y_shift_mm": [-0.026, 0.041],
        }
    ).to_csv(shifts, index=False)

    single = tmp_path / "single"
    assert script_runner.run(["linum-align-mosaics-3d-from-shifts", mosaics, shifts, single]).success
    planned = tmp_path / "planned"
    assert script_runner.run(["linum-align-mosaics-3d-from-shifts", mosaics, shifts, planned, "--plan_only"]).success
    assert not list(planned.glob("*.ome.zarr"))

    with (planned / "common_space_plan.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["filename"] for row in rows] == [f"slice_z{i:02d}_normalize.ome.zarr" for i in (3, 4, 5)]
    for row in rows:
        out = tmp_path / "one_by_one" / row["filename"]
        out.parent.mkdir(exist_ok=True)
        ret = script_runner.run(
            [
                "linum-apply-common-space-shift",
                mosaics / row["filename"],
                out,
                f"--shift_x_px={row['shift_x_px']}",
                f"--shift_y_px={row['shift_y_px']}",
                "--shape",
                row["height"],
                row["width"],
            ]
        )
        assert ret.success
        expected, expected_res = read_omezarr(single / row["filename"])
        shifted, res = read_omezarr(out)
        assert list(res) == list(expected_res)
        assert np.array_equal(np.asarray(shifted[:]), np.asarray(expected[:]))
