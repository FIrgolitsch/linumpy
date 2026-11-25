#!/usr/bin/env python3
"""Tests for ``scripts/stacking/linum_stack_slices_motor.py``.

The motor-stacking script is loaded via :mod:`importlib` so its pure helper
functions (canvas sizing, transform loading) can be characterized on small
synthetic inputs, without depending on real OME-Zarr slice volumes.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "stacking" / "linum_stack_slices_motor.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("linum_stack_slices_motor", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_help(script_runner):
    ret = script_runner.run(["linum-stack-slices-2d", "--help"])
    assert ret.success


def test_stack_slices_motor_help(script_runner):
    ret = script_runner.run(["linum-stack-slices-motor", "--help"])
    assert ret.success


# ---------------------------------------------------------------------------
# compute_output_shape (canvas sizing)
# ---------------------------------------------------------------------------


class TestComputeOutputShape:
    def test_single_slice_no_shift(self):
        mod = _load_module()
        first_vol_shape = (10, 20, 30)  # Z, Y, X
        ny, nx, x0, y0 = mod.compute_output_shape({0: "slice0"}, {0: (0.0, 0.0)}, first_vol_shape)
        assert (ny, nx) == (20, 30)
        assert (x0, y0) == (0, 0)

    def test_two_slices_offset_expands_canvas(self):
        mod = _load_module()
        first_vol_shape = (10, 20, 30)
        cumsum_px = {0: (0.0, 0.0), 1: (5.0, -3.0)}
        ny, nx, x0, y0 = mod.compute_output_shape({0: "slice0", 1: "slice1"}, cumsum_px, first_vol_shape)
        assert nx == 35
        assert ny == 23
        assert x0 == 0
        assert y0 == -3


# ---------------------------------------------------------------------------
# load_registration_transforms
# ---------------------------------------------------------------------------


class TestLoadRegistrationTransforms:
    def _write_transform_dir(self, tmp_path: Path, metrics: dict | None) -> None:
        transform_dir = tmp_path / "slice_z01"
        transform_dir.mkdir()
        sitk.WriteTransform(sitk.Euler2DTransform(), str(transform_dir / "transform.tfm"))
        (transform_dir / "offsets.txt").write_text("5\n10\n")
        if metrics is not None:
            (transform_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics))

    def test_loads_transform_offsets_and_confidence(self, tmp_path: Path):
        mod = _load_module()
        metrics = {
            "overall_status": "ok",
            "metrics": {
                "registration_confidence": {"value": 0.9},
                "translation_x": {"value": 1.5},
                "translation_y": {"value": -2.0},
                "z_correlation": {"value": 0.8},
                "rotation": {"value": 0.1},
            },
        }
        self._write_transform_dir(tmp_path, metrics)

        transforms, pairwise, _sources = mod.load_registration_transforms(tmp_path, [0, 1])

        assert transforms[1] is not None
        _tfm, fixed_z, moving_z, confidence = transforms[1]
        assert fixed_z == 5
        assert moving_z == 10
        assert abs(confidence - 0.9) < 1e-9
        assert pairwise[1] == (1.5, -2.0, 0.8)

    def test_missing_transform_dir_returns_none(self, tmp_path: Path):
        mod = _load_module()
        transforms, pairwise, _sources = mod.load_registration_transforms(tmp_path, [0, 1])
        assert transforms[1] is None
        assert pairwise == {}

    def test_skip_error_status_discards_transform(self, tmp_path: Path):
        mod = _load_module()
        self._write_transform_dir(tmp_path, {"overall_status": "error", "metrics": {}})

        transforms, _pairwise, _sources = mod.load_registration_transforms(tmp_path, [0, 1], skip_error_status=True)

        assert transforms[1] is None


# ---------------------------------------------------------------------------
# Depth of the joints, end to end
# ---------------------------------------------------------------------------


def _textured_block(nz: int, shape: tuple[int, int] = (160, 160)) -> np.ndarray:
    """Tissue whose fine structure changes smoothly with depth, in a round footprint."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    noise = gaussian_filter(rng.normal(size=(nz, *shape)).astype(np.float32), (4.0, 3.0, 3.0))
    noise /= noise.std()
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    inside = ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2) < (0.45 * shape[0]) ** 2
    return np.where(inside[None], 0.3 * (1.0 + 0.4 * noise), 0.0).astype(np.float32)


def _plane_correlation(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a > 0) & (b > 0)
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def test_stack_places_each_slab_at_its_measured_depth(script_runner, tmp_path):
    """Slabs that start 17, 23 and 20 planes below the one above, with slabs of unequal depth."""
    import dask.array as da
    import zarr

    from linumpy.io.zarr import read_omezarr, save_omezarr

    block = _textured_block(140)
    tops = [0, 17, 40, 60]
    depths = [44, 58, 36, 44]
    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    for index, (top, depth) in enumerate(zip(tops, depths, strict=True)):
        decay = np.exp(-0.03 * np.arange(depth, dtype=np.float32))[:, None, None]
        slab = block[top : top + depth] * decay
        save_omezarr(da.from_array(slab), slices_dir / f"slice_z{index:02d}.ome.zarr", (0.01, 0.01, 0.01))
    shifts = tmp_path / "shifts_xy.csv"
    rows = ["fixed_id,moving_id,x_shift,y_shift,x_shift_mm,y_shift_mm,reliable"]
    rows += [f"{index},{index + 1},0,0,0,0,1" for index in range(len(tops) - 1)]
    shifts.write_text("\n".join(rows) + "\n")
    out = tmp_path / "stack.ome.zarr"

    ret = script_runner.run(
        [
            "linum-stack-slices-motor",
            str(slices_dir),
            str(shifts),
            str(out),
            "--use_expected_overlap",
            "--no_xy_shift",
            "--blend",
            "--blend_cut_face_planes",
            "8",
            "--z_step_from_structure",
            "--z_step_search_vox",
            "8",
        ]
    )
    assert ret.success, ret.stderr

    # The second slab starts three planes above the interval: those planes are
    # left out and it is handed over at the interval. The others start at
    # their own top. No slab is moved because its overlap differs from the others'.
    attrs = zarr.open_group(str(out), mode="r").attrs
    assert attrs["slab_z_starts"] == [0, 20, 40, 60]
    stack, _ = read_omezarr(out, level=0)
    stack = np.asarray(stack[:])
    assert stack.shape[0] == 60 + depths[-1]
    for plane in (5, 18, 22, 35, 45, 58, 62, 90):
        assert _plane_correlation(stack[plane], block[plane]) > 0.97, plane
