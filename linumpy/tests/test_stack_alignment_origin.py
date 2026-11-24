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
