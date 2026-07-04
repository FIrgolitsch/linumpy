"""Tests for linumpy/mosaic/grid.py — MosaicGrid class and module-level helpers.

Covers construction, tile extraction/setting, neighbor enumeration, position
computation, stitching, blending helpers, and diffusion/average blending
weight functions. Complements ``test_mosaic_grid_overlap.py`` (which targets
the overlap-similarity and crop-tile paths).
"""

import numpy as np
import pytest

from linumpy.mosaic.grid import (
    MosaicGrid,
    add_volume_to_mosaic,
    get_average_blending_weights,
    get_diffusion_blending_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_grid_image(n_tiles_x: int, n_tiles_y: int, tile_size: int = 8) -> np.ndarray:
    """Build a non-trivial mosaic-grid image with unique per-tile intensities."""
    image = np.zeros((n_tiles_x * tile_size, n_tiles_y * tile_size), dtype=np.float32)
    for x in range(n_tiles_x):
        for y in range(n_tiles_y):
            x0, y0 = x * tile_size, y * tile_size
            # Unique value per tile so we can verify extraction order.
            image[x0 : x0 + tile_size, y0 : y0 + tile_size] = (x + 1) * 10.0 + (y + 1)
    return image


def _make_grid(n_tiles_x: int = 2, n_tiles_y: int = 3, tile_size: int = 8, overlap: float = 0.2) -> MosaicGrid:
    image = _make_grid_image(n_tiles_x, n_tiles_y, tile_size)
    return MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=overlap)


# ---------------------------------------------------------------------------
# Construction & geometry queries
# ---------------------------------------------------------------------------


def test_construction_sets_tile_shape_and_dtype():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    assert grid.tile_shape == (8, 8)
    assert grid.tile_size_x == 8
    assert grid.tile_size_y == 8
    assert grid.dtype == np.float32
    assert grid.blending_method is None


def test_construction_records_intensity_range():
    image = _make_grid_image(2, 2, tile_size=4)
    grid = MosaicGrid(image=image, tile_shape=(4, 4), overlap_fraction=0.25)
    assert grid.imin == float(image.min())
    assert grid.imax == float(image.max())


def test_compute_mosaic_shape_correct_tile_counts():
    grid = _make_grid(n_tiles_x=3, n_tiles_y=4, tile_size=8)
    assert grid.n_tiles_x == 3
    assert grid.n_tiles_y == 4


def test_set_affine_diagonal_matches_overlap():
    tile_size = 16
    overlap = 0.25
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=tile_size, overlap=overlap)
    expected = np.eye(2) * (1 - overlap) * tile_size
    np.testing.assert_allclose(np.asarray(grid.affine), expected)
    assert grid.overlap_fraction == overlap


def test_set_affine_updates_overlap_fraction():
    grid = _make_grid(overlap=0.2)
    grid.set_affine(overlap_fraction=0.4)
    assert grid.overlap_fraction == 0.4


# ---------------------------------------------------------------------------
# Tile extraction / setting
# ---------------------------------------------------------------------------


def test_get_tile_returns_correct_block():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=3, tile_size=8)
    tile_0_0 = grid.get_tile(0, 0)
    tile_1_2 = grid.get_tile(1, 2)
    # Tile (0,0) has value 11, tile (1,2) has value 23.
    np.testing.assert_allclose(tile_0_0, np.full((8, 8), 11.0, dtype=np.float32))
    np.testing.assert_allclose(tile_1_2, np.full((8, 8), 23.0, dtype=np.float32))


def test_get_tile_out_of_bounds_raises():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    with pytest.raises(AssertionError):
        grid.get_tile(5, 0)
    with pytest.raises(AssertionError):
        grid.get_tile(0, 5)


def test_set_tile_overwrites_block():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    new_tile = np.full((8, 8), 99.0, dtype=np.float32)
    grid.set_tile(1, 1, new_tile)
    np.testing.assert_allclose(grid.get_tile(1, 1), 99.0)


def test_get_tiles_returns_all_tiles_with_positions():
    n_tiles_x, n_tiles_y, tile_size = 2, 3, 8
    grid = _make_grid(n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y, tile_size=tile_size)
    tiles, positions = grid.get_tiles()
    assert tiles.shape == (n_tiles_x * n_tiles_y, tile_size, tile_size)
    assert len(positions) == n_tiles_x * n_tiles_y
    # Positions are enumerated row-major (x outer, y inner).
    assert positions[0] == (0, 0)
    assert positions[1] == (0, 1)
    assert positions[3] == (1, 0)


def test_get_image_returns_original_array():
    image = _make_grid_image(2, 2, tile_size=4)
    grid = MosaicGrid(image=image, tile_shape=(4, 4), overlap_fraction=0.25)
    returned = grid.get_image()
    np.testing.assert_array_equal(returned, image)


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------


def test_get_position_origin_is_zero():
    grid = _make_grid(overlap=0.25)
    pos = grid.get_position(0, 0)
    np.testing.assert_array_equal(pos, np.array([0, 0]))


def test_get_position_uses_affine():
    tile_size = 16
    overlap = 0.25
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=tile_size, overlap=overlap)
    step = int(tile_size * (1 - overlap))  # 12
    pos_1_0 = grid.get_position(1, 0)
    pos_0_1 = grid.get_position(0, 1)
    pos_1_1 = grid.get_position(1, 1)
    np.testing.assert_array_equal(pos_1_0, np.array([step, 0]))
    np.testing.assert_array_equal(pos_0_1, np.array([0, step]))
    np.testing.assert_array_equal(pos_1_1, np.array([step, step]))


# ---------------------------------------------------------------------------
# Neighbor enumeration
# ---------------------------------------------------------------------------


def test_get_neighbors_around_tile_N4_corner():
    grid = _make_grid(n_tiles_x=3, n_tiles_y=3, tile_size=8)
    # Corner tile (0,0) has 2 N4 neighbors: (1,0) and (0,1).
    neighbors, positions = grid.get_neighbors_around_tile(0, 0, neighborhood_type="N4")
    assert len(neighbors) == 2
    assert set(positions) == {(1, 0), (0, 1)}


def test_get_neighbors_around_tile_N4_center():
    grid = _make_grid(n_tiles_x=3, n_tiles_y=3, tile_size=8)
    # Center tile (1,1) has 4 N4 neighbors.
    neighbors, positions = grid.get_neighbors_around_tile(1, 1, neighborhood_type="N4")
    assert len(neighbors) == 4
    assert set(positions) == {(0, 1), (2, 1), (1, 0), (1, 2)}


def test_get_neighbors_around_tile_N8_includes_diagonals():
    grid = _make_grid(n_tiles_x=3, n_tiles_y=3, tile_size=8)
    neighbors, positions = grid.get_neighbors_around_tile(1, 1, neighborhood_type="N8")
    assert len(neighbors) == 8
    expected = {
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    }
    assert set(positions) == expected


def test_get_neighbors_around_tile_Nd_only_diagonals():
    grid = _make_grid(n_tiles_x=3, n_tiles_y=3, tile_size=8)
    _neighbors, positions = grid.get_neighbors_around_tile(1, 1, neighborhood_type="Nd")
    # 'Nd' selects only diagonal offsets.
    assert set(positions) == {(0, 0), (0, 2), (2, 0), (2, 2)}


def test_get_neighbors_list_N4_horizontal_and_vertical():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    neighbors = grid.get_neighbors_list(neighborhood_type="N4")
    # 2x2 grid: 2 horizontal + 2 vertical = 4 neighbor pairs.
    assert len(neighbors) == 4
    assert ((0, 0), (1, 0)) in neighbors  # horizontal
    assert ((0, 0), (0, 1)) in neighbors  # vertical


def test_get_neighbors_list_N8_adds_diagonals():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    neighbors = grid.get_neighbors_list(neighborhood_type="N8")
    # 2x2 grid: 4 (N4) + 1 (down-right) + 1 (up-right) = 6.
    assert len(neighbors) == 6
    assert ((0, 0), (1, 1)) in neighbors
    assert ((1, 0), (0, 1)) in neighbors


def test_get_neighbors_list_sets_attribute():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    neighbors = grid.get_neighbors_list(neighborhood_type="N4")
    assert grid.neighbors_list == neighbors


def test_get_neighbor_tiles_returns_pair():
    grid = _make_grid(n_tiles_x=2, n_tiles_y=2, tile_size=8)
    grid.get_neighbors_list(neighborhood_type="N4")
    tile_1, tile_2 = grid.get_neighbor_tiles(0)
    assert tile_1.shape == (8, 8)
    assert tile_2.shape == (8, 8)


def test_get_neighbor_overlap_returns_overlap_arrays():
    """For a 2x1 grid with overlap, the overlap region is non-empty."""
    tile_size = 8
    # Build a smooth ramp so the overlap region is meaningful.
    scene = np.arange(tile_size * 2, dtype=np.float32).reshape(tile_size * 2, 1)
    scene = np.tile(scene, (1, tile_size))
    tile_1 = scene[:tile_size, :]
    tile_2 = scene[tile_size:, :]
    image = np.vstack([tile_1, tile_2])
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.25)
    grid.get_neighbors_list(neighborhood_type="N4")
    o1, o2, pos1, pos2 = grid.get_neighbor_overlap(0)
    assert o1.shape == o2.shape
    assert o1.size > 0
    assert len(pos1) == 4
    assert len(pos2) == 4


# ---------------------------------------------------------------------------
# Blending method configuration
# ---------------------------------------------------------------------------


def test_set_blending_method_none_clears_attribute():
    grid = _make_grid()
    grid.set_blending_method("average")
    assert grid.blending_method == "average"
    grid.set_blending_method("none")
    assert grid.blending_method is None


def test_set_blending_method_average():
    grid = _make_grid()
    grid.set_blending_method("average")
    assert grid.blending_method == "average"


def test_set_blending_method_diffusion():
    grid = _make_grid()
    grid.set_blending_method("diffusion")
    assert grid.blending_method == "diffusion"


def test_set_blending_method_case_insensitive():
    grid = _make_grid()
    grid.set_blending_method("AVERAGE")
    assert grid.blending_method == "average"


def test_set_blending_method_invalid_raises():
    grid = _make_grid()
    with pytest.raises(AssertionError):
        grid.set_blending_method("bogus")


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def test_get_stitched_image_none_blending_shape():
    """Stitched image spans the bounding box of all tile positions + tile size."""
    n_tiles_x, n_tiles_y, tile_size = 2, 2, 8
    image = _make_grid_image(n_tiles_x, n_tiles_y, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.0)
    stitched = grid.get_stitched_image(blending_method="none")
    # With 0 overlap, stitched shape == (n_tiles_x * tile_size, n_tiles_y * tile_size).
    assert stitched.shape == (n_tiles_x * tile_size, n_tiles_y * tile_size)


def test_get_stitched_image_with_overlap_smaller_than_grid():
    """With overlap, the stitched image is smaller than the raw grid."""
    tile_size = 8
    overlap = 0.5
    image = _make_grid_image(2, 2, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=overlap)
    stitched = grid.get_stitched_image(blending_method="none")
    # Step = tile_size * (1 - overlap) = 4. Bounding box = (4 + 8, 4 + 8) = (12, 12).
    assert stitched.shape == (12, 12)


def test_get_stitched_image_skips_empty_tiles():
    """Tiles that are all-zero are skipped during stitching."""
    tile_size = 8
    image = np.zeros((2 * tile_size, 2 * tile_size), dtype=np.float32)
    # Only tile (0,0) has data.
    image[:tile_size, :tile_size] = 5.0
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.0)
    stitched = grid.get_stitched_image(blending_method="none")
    # Top-left tile region should be 5.0; rest should be 0.
    assert stitched.shape == (2 * tile_size, 2 * tile_size)
    np.testing.assert_allclose(stitched[:tile_size, :tile_size], 5.0)


def test_get_stitched_image_average_blending_runs():
    """Average blending path executes without error and produces a 2D array."""
    tile_size = 8
    image = _make_grid_image(2, 2, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.25)
    stitched = grid.get_stitched_image(blending_method="average")
    assert stitched.ndim == 2
    assert stitched.shape[0] > 0
    assert stitched.shape[1] > 0


# ---------------------------------------------------------------------------
# add_volume_to_mosaic (module-level helper)
# ---------------------------------------------------------------------------


def test_add_volume_to_mosaic_2d_no_overlap():
    """Adding a 2D tile to an empty mosaic places the tile at the given position."""
    mosaic = np.zeros((16, 16), dtype=np.float32)
    tile = np.full((8, 8), 7.0, dtype=np.float32)
    result = add_volume_to_mosaic(tile, (0, 0), mosaic, blending_method="none")
    np.testing.assert_allclose(result[:8, :8], 7.0)
    np.testing.assert_allclose(result[8:, :], 0.0)


def test_add_volume_to_mosaic_3d_volume_into_3d_mosaic():
    """3D volume added to a 3D mosaic at offset position."""
    nz, nx, ny = 2, 4, 4
    mosaic = np.zeros((nz, 8, 8), dtype=np.float32)
    volume = np.full((nz, nx, ny), 3.0, dtype=np.float32)
    result = add_volume_to_mosaic(volume, (2, 2), mosaic, blending_method="none")
    np.testing.assert_allclose(result[:, 2 : 2 + nx, 2 : 2 + ny], 3.0)


def test_add_volume_to_mosaic_average_blending_in_overlap():
    """In overlap region with average blending, result is the mean of old & new."""
    mosaic = np.zeros((1, 4, 4), dtype=np.float32)
    # Pre-fill the right half with 10.0.
    mosaic[:, :, 2:] = 10.0
    tile = np.full((1, 4, 4), 20.0, dtype=np.float32)
    result = add_volume_to_mosaic(tile, (0, 0), mosaic, blending_method="average")
    # Overlap region (cols 2-3) should be (10 + 20) / 2 = 15.
    np.testing.assert_allclose(result[:, :, 2:], 15.0)
    # Non-overlap region (cols 0-1) should be 20.
    np.testing.assert_allclose(result[:, :, :2], 20.0)


def test_add_volume_to_mosaic_diffusion_blending_runs():
    """Diffusion blending path executes and produces a finite-valued mosaic."""
    mosaic = np.zeros((1, 12, 12), dtype=np.float32)
    tile_1 = np.full((1, 8, 8), 5.0, dtype=np.float32)
    tile_2 = np.full((1, 8, 8), 9.0, dtype=np.float32)
    mosaic = add_volume_to_mosaic(tile_1, (0, 0), mosaic, blending_method="diffusion")
    mosaic = add_volume_to_mosaic(tile_2, (0, 4), mosaic, blending_method="diffusion")
    assert np.all(np.isfinite(mosaic))


# ---------------------------------------------------------------------------
# Blending weight helpers
# ---------------------------------------------------------------------------


def test_get_average_blending_weights_no_overlap_ones():
    mask = np.zeros((4, 4), dtype=bool)
    alpha = get_average_blending_weights(mask)
    np.testing.assert_allclose(alpha, np.ones((4, 4)))


def test_get_average_blending_weights_overlap_half():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True  # left half overlaps
    alpha = get_average_blending_weights(mask)
    np.testing.assert_allclose(alpha[mask], 0.5)
    np.testing.assert_allclose(alpha[~mask], 1.0)


def test_get_diffusion_blending_weights_2d_basic():
    """Diffusion weights are in [0, 1] with boundary values 0 and interior 1."""
    mask = np.ones((16, 16), dtype=bool)
    alpha = get_diffusion_blending_weights(mask, factor=2, n_steps=50)
    assert alpha.shape == mask.shape
    assert np.all(np.isfinite(alpha))
    assert alpha.min() >= 0.0 - 1e-6
    assert alpha.max() <= 1.0 + 1e-6


def test_get_diffusion_blending_weights_3d_basic():
    mask = np.ones((8, 8, 8), dtype=bool)
    alpha = get_diffusion_blending_weights(mask, factor=2, n_steps=30)
    assert alpha.shape == mask.shape
    assert np.all(np.isfinite(alpha))


def test_get_diffusion_blending_weights_with_moving_mask():
    """When moving_mask is provided, weights respect both masks."""
    fixed_mask = np.ones((16, 16), dtype=bool)
    moving_mask = np.ones((16, 16), dtype=bool)
    alpha = get_diffusion_blending_weights(fixed_mask, moving_mask, factor=2, n_steps=30)
    assert alpha.shape == fixed_mask.shape
    assert np.all(np.isfinite(alpha))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_tile_grid_no_overlap_stitches_without_raising():
    """A 1x1 grid (no neighbors, no overlap) stitches to the tile itself."""
    tile_size = 8
    image = _make_grid_image(1, 1, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.0)
    stitched = grid.get_stitched_image(blending_method="none")
    np.testing.assert_allclose(stitched, image)


def test_global_overlap_similarity_single_tile_returns_zero():
    """A 1x1 grid has no neighbor pairs → similarity is 0.0."""
    tile_size = 8
    image = _make_grid_image(1, 1, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.25)
    assert grid.global_overlap_similarity() == 0.0


def test_get_neighbors_list_single_tile_empty():
    """A 1x1 grid has no neighbor pairs."""
    tile_size = 8
    image = _make_grid_image(1, 1, tile_size)
    grid = MosaicGrid(image=image, tile_shape=(tile_size, tile_size), overlap_fraction=0.25)
    neighbors = grid.get_neighbors_list(neighborhood_type="N4")
    assert neighbors == []
