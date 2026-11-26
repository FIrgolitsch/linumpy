#!/usr/bin/env python3
import json


def test_help(script_runner):
    ret = script_runner.run(["linum-estimate-slab-slant", "--help"])
    assert ret.success


def test_slant_is_measured_and_removed_by_the_shift_script(script_runner, tmp_path):
    import dask.array as da
    import numpy as np
    from scipy import ndimage as ndi

    from linumpy.io.zarr import read_omezarr, save_omezarr
    from linumpy.stack_alignment.slab_slant import plane_shift

    step, depth, slant_x = 20, 30, 0.08
    rng = np.random.default_rng(2)
    block = (ndi.gaussian_filter(rng.standard_normal((7 * step + depth, 200, 220)), (1.0, 2.0, 2.0)) * 6.0 + 2.0).clip(0.2)
    mosaics = tmp_path / "inputs"
    mosaics.mkdir()
    lines = ["slice_id,filename,shift_x_px,shift_y_px,height,width"]
    for n in range(7):
        slab = np.stack(
            [ndi.shift(block[n * step + z], (0.0, slant_x * z), order=3, mode="nearest") for z in range(depth)]
        ).astype(np.float32)
        name = f"slice_z{n:02d}_normalize.ome.zarr"
        save_omezarr(da.from_array(slab), mosaics / name, (0.01, 0.01, 0.01), chunks=(depth, 128, 128))
        lines.append(f"{n},{name},4.0,6.0,212,230")
    plan = tmp_path / "common_space_plan.csv"
    plan.write_text("\n".join(lines) + "\n")

    slant_file = tmp_path / "slab_slant.json"
    command = ["linum-estimate-slab-slant", str(mosaics), str(plan), str(slant_file), "--slicing_interval_mm", "0.2"]
    ret = script_runner.run(command)
    assert ret.success
    result = json.loads(slant_file.read_text())
    assert result["used"]
    assert result["step_planes"] == step
    assert abs(result["slant_x_px_per_plane"] - slant_x) < 0.01
    assert abs(result["slant_y_px_per_plane"]) < 0.01

    placed = []
    for n in (2, 3):
        name = f"slice_z{n:02d}_normalize.ome.zarr"
        out = tmp_path / f"placed_{n}.ome.zarr"
        ret = script_runner.run(
            [
                "linum-apply-common-space-shift",
                str(mosaics / name),
                str(out),
                "--shift_x_px=4.0",
                "--shift_y_px=6.0",
                "--shape",
                "212",
                "230",
                "--slant_file",
                str(slant_file),
            ]
        )
        assert ret.success
        assert "slant=(+0.0" in ret.stdout
        placed.append(np.asarray(read_omezarr(out)[0][:]))
    found = plane_shift(placed[0][step + 5, 20:-20, 20:-20], placed[1][5, 20:-20, 20:-20])
    assert found is not None
    assert np.abs(found[0]).max() < 0.1
