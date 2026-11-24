"""Tests for the slab origin shifts taken from the stage positions."""

import numpy as np
import pytest

from linumpy.stack_alignment.origin import slab_origin_shifts_mm


def test_stage_y_goes_to_image_x_and_stage_x_to_image_y():
    xmin = [10.0, 10.3, 10.3]
    ymin = [50.0, 50.0, 49.2]
    x_shift, y_shift = slab_origin_shifts_mm(xmin, ymin)
    np.testing.assert_allclose(x_shift, [0.0, -0.8])
    np.testing.assert_allclose(y_shift, [0.3, 0.0])


def test_shifts_place_every_slab_at_its_stage_position():
    rng = np.random.default_rng(0)
    xmin = rng.uniform(5, 15, size=12)
    ymin = rng.uniform(40, 60, size=12)
    x_shift, y_shift = slab_origin_shifts_mm(xmin, ymin)
    # The running sum is the position of the low corner relative to the first slice.
    np.testing.assert_allclose(np.cumsum(x_shift), ymin[1:] - ymin[0])
    np.testing.assert_allclose(np.cumsum(y_shift), xmin[1:] - xmin[0])


def test_tissue_fixed_on_the_stage_lines_up():
    """A point of the sample keeps its place when the mosaic grows on the low side."""
    point_x_mm, point_y_mm = 12.0, 55.0
    xmin = np.array([11.0, 10.4])  # the second mosaic starts one tile earlier in X
    ymin = np.array([54.0, 54.0])
    x_shift, y_shift = slab_origin_shifts_mm(xmin, ymin)
    row_in_slab = point_x_mm - xmin
    col_in_slab = point_y_mm - ymin
    origin_row = np.concatenate([[0.0], np.cumsum(y_shift)])
    origin_col = np.concatenate([[0.0], np.cumsum(x_shift)])
    assert row_in_slab[0] + origin_row[0] == pytest.approx(row_in_slab[1] + origin_row[1])
    assert col_in_slab[0] + origin_col[0] == pytest.approx(col_in_slab[1] + origin_col[1])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        slab_origin_shifts_mm([0.0, 1.0], [0.0])


def _lattice(transform=((57.9, 1.0), (-1.33, 58.66))):
    from linumpy.stack_alignment.origin import TileLattice

    return TileLattice(transform=np.array(transform, dtype=float), tile_shape=(75, 75), step_mm=(0.6, 0.6))


def test_nominal_lattice_is_the_plain_pixel_size():
    lattice = _lattice(((60.0, 0.0), (0.0, 60.0)))
    # 0.3 mm along image X (stage Y) and -1.2 mm along image Y (stage X), at 10 um.
    assert lattice.displacement_px(0.3, -1.2) == pytest.approx((30.0, -120.0))
    assert lattice.first_tile_px(12, 9) == (0.0, 0.0)


def test_lattice_displacement_follows_the_fitted_steps():
    lattice = _lattice()
    # One tile step along stage X: 57.9 px down the rows and 1.33 px back along the columns.
    assert lattice.displacement_px(0.0, 0.6) == pytest.approx((-1.33, 57.9))
    # One tile step along stage Y: 58.66 px along the columns and 1 px down the rows.
    assert lattice.displacement_px(0.6, 0.0) == pytest.approx((58.66, 1.0))


def test_lattice_recovers_the_grid_from_the_stitched_shape():
    from linumpy.mosaic.motor import compute_affine_output_shape

    lattice = _lattice()
    for nx, ny in [(1, 1), (9, 8), (13, 12), (21, 29), (11, 12)]:
        _, height, width = compute_affine_output_shape(nx, ny, (4, 75, 75), lattice.transform)
        assert lattice.grid_shape((4, height, width)) == (nx, ny)
    assert lattice.grid_shape((4, 1234, 37)) is None


def test_first_tile_position_is_the_one_the_stitcher_uses():
    from linumpy.mosaic.motor import compute_affine_positions

    lattice = _lattice()
    for nx, ny in [(9, 8), (17, 21), (11, 12)]:
        positions = compute_affine_positions(nx, ny, lattice.transform)
        expected = (-min(p[0] for p in positions), -min(p[1] for p in positions))
        assert lattice.first_tile_px(nx, ny) == expected


def test_lattice_placement_lines_up_tissue_across_a_change_of_grid():
    """A point fixed on the stage lands on the same pixel of the common frame in two slices.

    The second mosaic starts elsewhere on the stage and has another grid, so
    both the conversion of the stage shift and the canvas offset matter.
    """
    lattice = _lattice()
    point = np.array([14.2, 58.3])  # stage X, Y in mm
    corners = [np.array([10.0, 54.0]), np.array([8.2, 55.8])]
    grids = [(17, 21), (11, 12)]
    seen = []
    for corner, grid in zip(corners, grids, strict=True):
        # Where the stitcher puts the point inside the slab image.
        index = (point - corner) / 0.6
        in_slab = lattice.transform @ index + np.array(lattice.first_tile_px(*grid))
        # Where the slab origin goes in the common frame.
        dx, dy = lattice.displacement_px(corner[1] - corners[0][1], corner[0] - corners[0][0])
        first_row, first_col = lattice.first_tile_px(*grid)
        origin = np.array([dy - first_row, dx - first_col])
        seen.append(origin + in_slab)
    np.testing.assert_allclose(seen[0], seen[1], atol=1e-9)
