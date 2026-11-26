"""Tests for linumpy/stack_alignment/slab_slant.py"""

import dask.array as da
import numpy as np
import pytest
from scipy import ndimage as ndi

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.stack_alignment.common_space import shift_mosaic_to_common_space
from linumpy.stack_alignment.slab_slant import estimate_slab_slant, plane_shift, read_slab_slant, write_slab_slant

STEP = 6
DEPTH = 14
SLANT = (0.11, -0.07)  # (x, y) px per plane


def _texture(shape, seed):
    """Smooth positive texture with structure at a few pixels."""
    noise = np.random.default_rng(seed).standard_normal(shape)
    return (ndi.gaussian_filter(noise, 2.0) * 6.0 + 2.0).clip(0.2).astype(np.float32)


def _slabs(folder, n_slabs, slant, positions):
    """Slabs cut from one block: slab n shows planes n * STEP onward, with the slant inside every slab."""
    height, width = 220, 240
    block = _texture((n_slabs * STEP + DEPTH, height + 40, width + 40), seed=11)
    block = ndi.gaussian_filter(block, (1.0, 0, 0))
    rows = []
    for n in range(n_slabs):
        pos_x, pos_y = positions[n]
        slab = np.empty((DEPTH, height, width), dtype=np.float32)
        for z in range(DEPTH):
            # The slab is a window of the block at (pos_y, pos_x); its content sits slant * z further.
            slab[z] = ndi.shift(block[n * STEP + z], (slant[1] * z - pos_y, slant[0] * z - pos_x), order=3, mode="nearest")[
                20 : 20 + height, 20 : 20 + width
            ]
        path = folder / f"slice_z{n:02d}_normalize.ome.zarr"
        save_omezarr(da.from_array(slab), path, (0.01, 0.01, 0.01), chunks=(DEPTH, 128, 128))
        # The origin of the slab is the point (pos_y, pos_x) of the block: that is where it sits in the common frame.
        rows.append((n, path, 10.0 + pos_x, 10.0 + pos_y))
    return rows


def test_plane_shift_finds_a_fraction_of_a_pixel_with_the_right_sign():
    fixed = _texture((200, 220), seed=5)
    # The moving plane is the fixed one moved by (-1.3, +0.6): it has to be moved back by (+1.3, -0.6).
    moving = ndi.shift(fixed, (-1.3, 0.6), order=3, mode="nearest")
    found = plane_shift(fixed, moving)
    assert found is not None
    shift, peak = found
    assert shift == pytest.approx([1.3, -0.6], abs=0.05)
    assert peak > 0.9


def test_plane_shift_uses_the_offset_and_gives_up_without_common_tissue():
    fixed = _texture((200, 220), seed=5)
    moving = fixed[30:, 12:]  # the origin of the moving plane sits at (30, 12) on the fixed grid
    found = plane_shift(fixed, moving, offset=(30.0, 12.0))
    assert found is not None
    assert found[0] == pytest.approx([0.0, 0.0], abs=0.03)
    assert plane_shift(fixed, np.zeros_like(fixed)) is None


def test_estimate_slab_slant_recovers_the_slant_and_ignores_a_misplaced_joint(tmp_path):
    positions = [(0.0, 0.0), (3.0, -2.0), (5.5, 1.25), (2.0, 4.0), (-1.0, 3.0), (0.5, 0.5), (4.0, -3.0), (1.0, 1.0)]
    slabs = _slabs(tmp_path, len(positions), SLANT, positions)
    # One slab is placed 3 px off: its two joints are outliers, the median holds.
    wrong = list(slabs)
    wrong[4] = (wrong[4][0], wrong[4][1], wrong[4][2] + 3.0, wrong[4][3])

    result = estimate_slab_slant(wrong, STEP, planes=(2, 4, 6), min_joints=5)

    assert result["used"]
    assert result["n_joints"] == len(positions) - 1
    assert result["slant_x_px_per_plane"] == pytest.approx(SLANT[0], abs=0.01)
    assert result["slant_y_px_per_plane"] == pytest.approx(SLANT[1], abs=0.01)

    path = tmp_path / "slab_slant.json"
    write_slab_slant(result, path)
    assert read_slab_slant(path) == pytest.approx((result["slant_x_px_per_plane"], result["slant_y_px_per_plane"]))


def test_no_slant_is_reported_from_too_few_joints_or_across_a_missing_slice(tmp_path):
    positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
    slabs = _slabs(tmp_path, 3, SLANT, positions)
    result = estimate_slab_slant(slabs, STEP, planes=(2, 4), min_joints=5)
    assert not result["used"]
    assert result["n_joints"] == 2
    assert (result["slant_x_px_per_plane"], result["slant_y_px_per_plane"]) == (0.0, 0.0)
    # Slabs 0 and 2 are not neighbours: that joint is not measured.
    assert estimate_slab_slant([slabs[0], slabs[2]], STEP, planes=(2, 4))["n_joints"] == 0


def test_straightened_slabs_agree_in_their_overlap(tmp_path):
    positions = [(0.0, 0.0), (3.0, -2.0), (5.5, 1.25), (2.0, 4.0), (-1.0, 3.0), (0.5, 0.5)]
    slabs = _slabs(tmp_path, len(positions), SLANT, positions)
    result = estimate_slab_slant(slabs, STEP, planes=(2, 4, 6), min_joints=5)
    slant = (result["slant_x_px_per_plane"], result["slant_y_px_per_plane"])

    def placed(with_slant):
        out = []
        for slice_id, path, shift_x, shift_y in slabs[:2]:
            target = tmp_path / f"{'straight' if with_slant else 'plain'}_{slice_id}.ome.zarr"
            shift_mosaic_to_common_space(path, target, shift_x, shift_y, (260, 280), slant=slant if with_slant else (0.0, 0.0))
            out.append(np.asarray(read_omezarr(target)[0][:]))
        return out

    def remaining(volumes):
        found = plane_shift(volumes[0][STEP + 4, 30:-30, 30:-30], volumes[1][4, 30:-30, 30:-30])
        assert found is not None
        return found[0]

    # As stitched, the joint shows the slant times the step; straightened, nothing is left.
    assert remaining(placed(with_slant=False)) == pytest.approx([SLANT[1] * STEP, SLANT[0] * STEP], abs=0.06)
    assert remaining(placed(with_slant=True)) == pytest.approx([0.0, 0.0], abs=0.06)
