"""Tests for linumpy/mosaic/quick_stitch.py — pure-Python helpers.

The full ``quick_stitch`` and ``detect_mosaic`` functions require an OCT tile
directory on disk (``linumpy.microscope.oct.OCT``), so they are out of scope
for unit tests. This file covers the pure helpers: ``get_largest_cc``,
``get_tiles_ids_from_list``, ``get_tiles_ids`` (with a synthetic tmp_path
directory), and ``save_quickstitch``.
"""

from pathlib import Path

import numpy as np
import pytest

from linumpy.mosaic.quick_stitch import (
    DEFAULT_TILE_FILE_PATTERN,
    get_largest_cc,
    get_tiles_ids,
    get_tiles_ids_from_list,
    save_quickstitch,
)

# ---------------------------------------------------------------------------
# get_largest_cc
# ---------------------------------------------------------------------------


def test_get_largest_cc_single_blob():
    """A single connected blob is returned unchanged (as True)."""
    seg = np.zeros((20, 20), dtype=bool)
    seg[5:15, 5:15] = True
    largest = get_largest_cc(seg)
    assert largest.shape == seg.shape
    # The single blob is the largest (and only) component.
    np.testing.assert_array_equal(largest, seg)


def test_get_largest_cc_picks_bigger_of_two():
    """When two blobs exist, the larger one is returned."""
    seg = np.zeros((30, 30), dtype=bool)
    # Small blob.
    seg[2:5, 2:5] = True  # 3x3 = 9 px
    # Large blob.
    seg[10:25, 10:25] = True  # 15x15 = 225 px
    largest = get_largest_cc(seg)
    # The large blob region should be True; the small blob region should be False.
    assert largest[10:25, 10:25].all()
    assert not largest[2:5, 2:5].any()


def test_get_largest_cc_diagonal_blob_is_connected():
    """Diagonally-touching pixels are 8-connected → single component."""
    seg = np.zeros((10, 10), dtype=bool)
    seg[0, 0] = True
    seg[1, 1] = True
    seg[2, 2] = True
    largest = get_largest_cc(seg)
    # All three pixels are 8-connected → one component.
    assert largest.sum() == 3


def test_get_largest_cc_empty_raises():
    """An all-zero segmentation has no components → assertion error."""
    seg = np.zeros((10, 10), dtype=bool)
    with pytest.raises(AssertionError):
        get_largest_cc(seg)


# ---------------------------------------------------------------------------
# get_tiles_ids_from_list
# ---------------------------------------------------------------------------


def _make_tile_path(name: str, parent: Path) -> Path:
    """Create a fake tile directory entry (not a real file)."""
    p = parent / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_get_tiles_ids_from_list_parses_xyz():
    """Tile (x, y, z) IDs are parsed from directory names matching the pattern."""
    tmp = Path(__file__).parent
    names = ["tile_x01_y02_z03", "tile_x00_y00_z00", "tile_x10_y05_z07"]
    paths = [_make_tile_path(n, tmp / "_tmp_quick_stitch_test") for n in names]
    try:
        ids = get_tiles_ids_from_list(paths)
        assert ids == [(0, 0, 0), (1, 2, 3), (10, 5, 7)]  # sorted by name
    finally:
        for p in paths:
            p.rmdir()
        (tmp / "_tmp_quick_stitch_test").rmdir()


def test_get_tiles_ids_from_list_custom_pattern():
    """A custom file_pattern with named groups is honored."""
    tmp = Path(__file__).parent
    pattern = r"t_(?P<x>\d+)_(?P<y>\d+)_(?P<z>\d+)"
    names = ["t_5_6_7", "t_1_2_3"]
    paths = [_make_tile_path(n, tmp / "_tmp_quick_stitch_test_custom") for n in names]
    try:
        ids = get_tiles_ids_from_list(paths, file_pattern=pattern)
        assert ids == [(1, 2, 3), (5, 6, 7)]
    finally:
        for p in paths:
            p.rmdir()
        (tmp / "_tmp_quick_stitch_test_custom").rmdir()


def test_get_tiles_ids_from_list_returns_sorted():
    """Output is sorted by path name (lexicographic)."""
    tmp = Path(__file__).parent
    names = ["tile_x05_y05_z05", "tile_x01_y01_z01", "tile_x03_y03_z03"]
    paths = [_make_tile_path(n, tmp / "_tmp_quick_stitch_sort") for n in names]
    try:
        ids = get_tiles_ids_from_list(paths)
        # Sorted by name: x01 < x03 < x05.
        assert ids[0] == (1, 1, 1)
        assert ids[1] == (3, 3, 3)
        assert ids[2] == (5, 5, 5)
    finally:
        for p in paths:
            p.rmdir()
        (tmp / "_tmp_quick_stitch_sort").rmdir()


# ---------------------------------------------------------------------------
# get_tiles_ids
# ---------------------------------------------------------------------------


def test_get_tiles_ids_finds_tile_directories(tmp_path):
    """get_tiles_ids discovers tile_* directories and parses their IDs."""
    for name in ["tile_x00_y00_z00", "tile_x01_y00_z00", "tile_x00_y01_z00"]:
        (tmp_path / name).mkdir()
    # Also add a non-tile directory that should be ignored.
    (tmp_path / "other_dir").mkdir()
    # And a regular file that should be ignored.
    (tmp_path / "tile_x99_y99_z99").write_text("not a dir")
    tiles, tile_ids = get_tiles_ids(tmp_path)
    assert len(tiles) == 3
    assert len(tile_ids) == 3
    assert set(tile_ids) == {(0, 0, 0), (1, 0, 0), (0, 1, 0)}


def test_get_tiles_ids_filters_by_z(tmp_path):
    """When z is provided, only tile_*z{z:02d} directories are returned."""
    for z in [0, 1, 2]:
        (tmp_path / f"tile_x00_y00_z{z:02d}").mkdir()
    tiles_z1, ids_z1 = get_tiles_ids(tmp_path, z=1)
    assert len(tiles_z1) == 1
    assert ids_z1 == [(0, 0, 1)]


def test_get_tiles_ids_empty_directory(tmp_path):
    """An empty directory yields empty tile list and IDs."""
    tiles, tile_ids = get_tiles_ids(tmp_path)
    assert tiles == []
    assert tile_ids == []


def test_default_tile_file_pattern_constant():
    """The module-level DEFAULT_TILE_FILE_PATTERN has the expected named groups."""
    import re

    match = re.match(DEFAULT_TILE_FILE_PATTERN, "tile_x12_y34_z56")
    assert match is not None
    assert match.group("x") == "12"
    assert match.group("y") == "34"
    assert match.group("z") == "56"


# ---------------------------------------------------------------------------
# save_quickstitch
# ---------------------------------------------------------------------------


def test_save_quickstitch_png_normalizes_to_uint8(tmp_path):
    """Saving as PNG produces a uint8 image in [0, 255]."""
    from imageio import imread

    img = np.zeros((20, 20), dtype=np.float32)
    img[5:15, 5:15] = 100.0
    out = tmp_path / "mosaic.png"
    save_quickstitch(img, out)
    assert out.exists()
    loaded = imread(out)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (20, 20)


def test_save_quickstitch_jpg_normalizes_to_uint8(tmp_path):
    """Saving as JPG produces a uint8 image."""
    from imageio import imread

    img = np.zeros((20, 20), dtype=np.float32)
    img[5:15, 5:15] = 100.0
    out = tmp_path / "mosaic.jpg"
    save_quickstitch(img, out)
    assert out.exists()
    loaded = imread(out)
    assert loaded.dtype == np.uint8


def test_save_quickstitch_tiff_keeps_float32(tmp_path):
    """Saving as TIFF preserves float32 intensity."""
    from imageio import imread

    img = np.zeros((20, 20), dtype=np.float32)
    img[5:15, 5:15] = 100.0
    out = tmp_path / "mosaic.tiff"
    save_quickstitch(img, out)
    assert out.exists()
    loaded = imread(out)
    # imageio may return float32 or float64 depending on backend.
    assert np.issubdtype(loaded.dtype, np.floating)


def test_save_quickstitch_creates_parent_directory(tmp_path):
    """Parent directories are created if they don't exist."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[2:8, 2:8] = 50.0
    out = tmp_path / "nested" / "subdir" / "mosaic.png"
    save_quickstitch(img, out)
    assert out.exists()


def test_save_quickstitch_all_zero_image_raises(tmp_path):
    """An all-zero image has an empty mask → save_quickstitch raises ValueError.

    This documents the pre-existing behavior: ``img[mask].min()`` on an empty
    mask raises ``ValueError: zero-size array to reduction operation minimum``.
    The function does not guard against all-zero input.
    """
    img = np.zeros((10, 10), dtype=np.float32)
    out = tmp_path / "empty.png"
    with pytest.raises(ValueError, match="zero-size array"):
        save_quickstitch(img, out)
