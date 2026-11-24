#!/usr/bin/env python3
import csv

from linumpy.io.test_data import get_data, write_synthetic_oct_tile


def test_help(script_runner):
    ret = script_runner.run(["linum-estimate-xy-shift-from-metadata", "--help"])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data("raw_tiles")
    output = tmp_path / "xy_shifts.csv"
    ret = script_runner.run(["linum-estimate-xy-shift-from-metadata", input, output])
    assert ret.success


def _write_slice(raw, z, tiles):
    """Write the tiles of one slice; *tiles* maps the grid cell to the stage position in mm."""
    for (mx, my), (x_mm, y_mm) in tiles.items():
        made = write_synthetic_oct_tile(raw / f"src_{mx}_{my}_{z}", nx=8, ny=8, nz=4)
        info = (made / "info.txt").read_text()
        # Field of view in whole microns, as the microscope writes it.
        info = info.replace("width: 8.0", "width: 8").replace("height: 8.0", "height: 8")
        info = info.replace("stage_x_pos_mm: 0.0", f"stage_x_pos_mm: {x_mm}")
        info = info.replace("stage_y_pos_mm: 0.0", f"stage_y_pos_mm: {y_mm}")
        (made / "info.txt").write_text(info)
        made.rename(raw / f"tile_x{mx:02d}_y{my:02d}_z{z:02d}")


def test_shift_is_the_move_of_the_low_corner_in_image_axes(script_runner, tmp_path):
    """The mosaic grows by one tile on the low X side and moves 0.25 mm in Y.

    The high X edge stays where it was. The slab image starts at the low
    corner, so its origin moves by the whole tile step along the image rows
    (stage X) and by 0.25 mm along the image columns (stage Y).
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    step = 0.6
    _write_slice(raw, 0, {(mx, my): (10.0 + step * mx, 50.0 + step * my) for mx in range(2) for my in range(2)})
    _write_slice(raw, 1, {(mx, my): (10.0 - step + step * mx, 50.25 + step * my) for mx in range(3) for my in range(2)})

    output = tmp_path / "xy_shifts.csv"
    ret = script_runner.run(["linum-estimate-xy-shift-from-metadata", raw, output, "--n_processes", 1])
    assert ret.success

    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    row = rows[0]
    assert (float(row["fixed_id"]), float(row["moving_id"])) == (0.0, 1.0)
    assert abs(float(row["x_shift_mm"]) - 0.25) < 1e-9
    assert abs(float(row["y_shift_mm"]) + step) < 1e-9
    assert float(row["reliable"]) == 1.0
