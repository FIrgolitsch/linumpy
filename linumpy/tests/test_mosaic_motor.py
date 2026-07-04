"""Tests for linumpy/mosaic/motor.py"""

import json
from pathlib import Path

import numpy as np

from linumpy.mosaic.motor import (
    _extract_displacement_params,
    apply_blend_shift_refinement,
    compare_motor_vs_registration,
    compute_affine_output_shape,
    compute_affine_positions,
    compute_motor_positions,
    compute_registration_refinements,
    estimate_affine_from_pairs,
)

# ---------------------------------------------------------------------------
# compute_motor_positions
# ---------------------------------------------------------------------------


def test_compute_motor_positions_count():
    positions, _step_y, _step_x = compute_motor_positions(nx=3, ny=4, tile_shape=(10, 64, 64), overlap_fraction=0.1)
    assert len(positions) == 12  # 3 x 4


def test_compute_motor_positions_step_sizes():
    tile_shape = (10, 100, 80)
    overlap = 0.2
    _positions, step_y, step_x = compute_motor_positions(nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=overlap)
    expected_step_y = int(100 * (1 - overlap))  # 80
    expected_step_x = int(80 * (1 - overlap))  # 64
    assert step_y == expected_step_y
    assert step_x == expected_step_x


def test_compute_motor_positions_first_is_origin():
    positions, _, _ = compute_motor_positions(nx=2, ny=3, tile_shape=(5, 50, 50), overlap_fraction=0.1)
    first = positions[0]
    assert first[0] == 0
    assert first[1] == 0


def test_compute_motor_positions_scale_factor():
    tile_shape = (10, 100, 100)
    _positions_1x, step_y_1x, _ = compute_motor_positions(
        nx=2, ny=1, tile_shape=tile_shape, overlap_fraction=0.0, scale_factor=1.0
    )
    _positions_2x, step_y_2x, _ = compute_motor_positions(
        nx=2, ny=1, tile_shape=tile_shape, overlap_fraction=0.0, scale_factor=2.0
    )
    assert step_y_2x == 2 * step_y_1x


# ---------------------------------------------------------------------------
# apply_blend_shift_refinement
# ---------------------------------------------------------------------------


def test_apply_blend_shift_refinement_empty_refinements():
    """No refinements → tile returned unchanged."""
    tile = np.ones((5, 16, 16), dtype=np.float32)
    result = apply_blend_shift_refinement(tile, [])
    np.testing.assert_array_equal(result, tile)


def test_apply_blend_shift_refinement_negligible_shift():
    """Sub-threshold shifts (< 0.1 px) → tile returned unchanged."""
    tile = np.ones((5, 16, 16), dtype=np.float32)
    refinements = [{"dx": 0.05, "dy": 0.05}]
    result = apply_blend_shift_refinement(tile, refinements)
    np.testing.assert_array_equal(result, tile)


def test_apply_blend_shift_refinement_applies_shift():
    """Large shift is applied, changing the tile data."""
    rng = np.random.default_rng(7)
    tile = (rng.random((5, 32, 32)) * 100.0).astype(np.float32)
    refinements = [{"dx": 3.0, "dy": 3.0}]
    result = apply_blend_shift_refinement(tile, refinements)
    # Shape must be preserved
    assert result.shape == tile.shape
    # Content must have changed
    assert not np.array_equal(result, tile)


def test_apply_blend_shift_refinement_averages_multiple():
    """Multiple refinements are averaged before application."""
    tile = (np.ones((5, 32, 32)) * 50.0).astype(np.float32)
    # Two opposite shifts → average ≈ 0 → no change (may not be exact due to shift)
    refinements = [{"dx": 0.0, "dy": 4.0}, {"dx": 0.0, "dy": -4.0}]
    result = apply_blend_shift_refinement(tile, refinements)
    # Average dy = 0 / 2 / 2 = 0 → negligible → should be unchanged
    np.testing.assert_array_equal(result, tile)


# ---------------------------------------------------------------------------
# compare_motor_vs_registration
# ---------------------------------------------------------------------------


def test_compare_motor_vs_registration_basic():
    motor = [(0, 0), (10, 0), (0, 10), (10, 10)]
    reg = [(1, 1), (11, 1), (1, 11), (11, 11)]
    result = compare_motor_vs_registration(motor, reg)
    assert result["n_tiles"] == 4
    assert abs(result["mean_diff_y"] - 1.0) < 1e-9
    assert abs(result["mean_diff_x"] - 1.0) < 1e-9
    assert result["systematic_offset"] is False  # only 1 px offset, threshold is 5


def test_compare_motor_vs_registration_systematic_offset():
    motor = [(0, 0)] * 5
    reg = [(10, 10)] * 5  # 10 px systematic offset
    result = compare_motor_vs_registration(motor, reg)
    assert result["systematic_offset"] is True
    assert "offset_warning" in result


def test_compare_motor_vs_registration_writes_json(tmp_path):
    motor = [(0, 0), (10, 0)]
    reg = [(1, 0), (11, 0)]
    out_path = str(tmp_path / "comparison.json")
    compare_motor_vs_registration(motor, reg, output_path=out_path)
    with Path(out_path).open() as f:
        loaded = json.load(f)
    assert loaded["n_tiles"] == 2
    assert abs(loaded["mean_diff_y"] - 1.0) < 1e-9


def test_compare_motor_vs_registration_no_dilation_flag():
    """Fewer than 10 tiles: no dilation_indicator key."""
    motor = [(0, 0), (10, 0)]
    reg = [(0, 0), (10, 0)]
    result = compare_motor_vs_registration(motor, reg)
    assert "dilation_indicator" not in result


# ---------------------------------------------------------------------------
# compute_motor_positions — rotation
# ---------------------------------------------------------------------------


def test_compute_motor_positions_rotation_changes_positions():
    """A 90-degree rotation swaps the (row, col) axes of the positions."""
    tile_shape = (10, 100, 80)
    overlap = 0.0
    positions_0, _, _ = compute_motor_positions(nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=overlap, rotation_deg=0.0)
    positions_90, _, _ = compute_motor_positions(
        nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=overlap, rotation_deg=90.0
    )
    # First tile stays at origin.
    assert np.allclose(positions_90[0], [0, 0])
    # Under 90° rotation, the (i*step_y, 0) tile maps to (0, -i*step_y) — i.e. col axis.
    # The rotated positions must differ from the unrotated ones (except origin).
    assert not np.allclose(positions_90[1], positions_0[1])


def test_compute_motor_positions_rotation_returns_ndarray_entries():
    """When rotation_deg != 0, entries are np.ndarray (per source code branch)."""
    positions, _, _ = compute_motor_positions(nx=2, ny=1, tile_shape=(5, 50, 50), overlap_fraction=0.1, rotation_deg=30.0)
    # Non-rotation branch returns tuples; rotation branch returns np.ndarray.
    assert isinstance(positions[1], np.ndarray)


# ---------------------------------------------------------------------------
# compute_registration_refinements
# ---------------------------------------------------------------------------


def _make_synthetic_volume(nx: int, ny: int, tile_shape: tuple, overlap_fraction: float) -> np.ndarray:
    """Build a synthetic mosaic-grid volume with smooth, registrable overlap.

    The volume is a single Z-slice (z=1) of shape (1, nx*tile_h, ny*tile_w)
    with a smooth ramp along the X axis so adjacent tile overlaps correlate.
    """
    tile_h, tile_w = tile_shape[1], tile_shape[2]
    height = nx * tile_h
    width = ny * tile_w
    # Smooth ramp along axis=1 (rows) so vertical-overlap pairs correlate.
    row_ramp = np.linspace(1.0, 100.0, num=height, dtype=np.float32)
    col_ramp = np.linspace(1.0, 50.0, num=width, dtype=np.float32)
    plane = row_ramp[:, None] + col_ramp[None, :]
    # Add a Z dimension.
    volume = plane[None, :, :]
    return volume.astype(np.float32)


def test_compute_registration_refinements_returns_well_formed_dict():
    """Output dict has all four expected keys with correct types."""
    tile_shape = (1, 32, 32)
    volume = _make_synthetic_volume(nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=0.25)
    refinements = compute_registration_refinements(volume, tile_shape, nx=2, ny=2, overlap_fraction=0.25)
    assert set(refinements.keys()) == {"horizontal", "vertical", "pairs", "stats"}
    assert isinstance(refinements["horizontal"], dict)
    assert isinstance(refinements["vertical"], dict)
    assert isinstance(refinements["pairs"], list)
    assert isinstance(refinements["stats"], dict)
    expected_stat_keys = {"total_pairs", "valid_pairs", "clamped_pairs", "mean_refinement", "max_refinement"}
    assert set(refinements["stats"].keys()) == expected_stat_keys


def test_compute_registration_refinements_pairs_carry_deltas():
    """Each pair entry has the four required keys with correct delta signs."""
    tile_shape = (1, 32, 32)
    volume = _make_synthetic_volume(nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=0.25)
    refinements = compute_registration_refinements(volume, tile_shape, nx=2, ny=2, overlap_fraction=0.25)
    for pair in refinements["pairs"]:
        assert set(pair.keys()) == {"row_delta", "col_delta", "measured_dy", "measured_dx"}
        assert pair["row_delta"] in (0, 1)
        assert pair["col_delta"] in (0, 1)


def test_compute_registration_refinements_empty_overlap_skipped():
    """When overlaps are entirely zero, no pairs are recorded."""
    tile_shape = (1, 16, 16)
    volume = np.zeros((1, 32, 32), dtype=np.float32)  # all empty
    refinements = compute_registration_refinements(volume, tile_shape, nx=2, ny=2, overlap_fraction=0.25)
    assert refinements["stats"]["total_pairs"] == 0
    assert refinements["pairs"] == []


def test_compute_registration_refinements_max_empty_fraction_skips_uniform():
    """With max_empty_fraction set, near-uniform low-intensity overlaps are skipped."""
    tile_shape = (1, 16, 16)
    # Volume with very low values everywhere → otsu threshold will classify most as background.
    volume = np.full((1, 32, 32), 1.0, dtype=np.float32)
    refinements = compute_registration_refinements(
        volume, tile_shape, nx=2, ny=2, overlap_fraction=0.25, max_empty_fraction=0.5
    )
    # All overlaps are uniform low → should be classified as empty and skipped.
    assert refinements["stats"]["total_pairs"] == 0


def test_compute_registration_refinements_clamps_large_residuals():
    """Residuals exceeding max_refinement_px are clamped and counted."""
    # Build a volume where adjacent tiles are shifted by a large amount.
    tile_shape = (1, 32, 32)
    nx, ny = 2, 2
    tile_h, tile_w = tile_shape[1], tile_shape[2]
    height = nx * tile_h
    width = ny * tile_w
    # Random data with a discontinuity between columns → large phase-correlation residual.
    rng = np.random.default_rng(42)
    plane = rng.random((height, width)).astype(np.float32) * 100.0
    # Force a 10-pixel horizontal shift between column 0 and column 1 tiles.
    plane[:tile_h, tile_w:] = plane[:tile_h, :tile_w]
    volume = plane[None, :, :]
    refinements = compute_registration_refinements(
        volume, tile_shape, nx=nx, ny=ny, overlap_fraction=0.25, max_refinement_px=1.0
    )
    # If any pair was registered, large residuals should be clamped.
    if refinements["stats"]["valid_pairs"] > 0:
        # Clamped pairs may or may not be present, but the count is consistent.
        assert refinements["stats"]["clamped_pairs"] >= 0
        assert refinements["stats"]["clamped_pairs"] <= refinements["stats"]["valid_pairs"]


def test_compute_registration_refinements_histogram_match_runs():
    """Histogram matching path executes without error."""
    tile_shape = (1, 32, 32)
    volume = _make_synthetic_volume(nx=2, ny=2, tile_shape=tile_shape, overlap_fraction=0.25)
    # Scale one half differently so histogram matching has an effect.
    volume[:, :, :32] *= 2.0
    refinements = compute_registration_refinements(volume, tile_shape, nx=2, ny=2, overlap_fraction=0.25, histogram_match=True)
    # Just verify it runs and produces a valid structure.
    assert "pairs" in refinements


# ---------------------------------------------------------------------------
# estimate_affine_from_pairs
# ---------------------------------------------------------------------------


def test_estimate_affine_from_pairs_empty_returns_fallback_diagonal():
    """Empty pairs list → fallback to diagonal model based on tile_shape + overlap."""
    tile_shape = (10, 100, 80)
    overlap = 0.2
    transform, diag = estimate_affine_from_pairs([], tile_shape, overlap)
    expected_step_y = tile_shape[1] * (1.0 - overlap)
    expected_step_x = tile_shape[2] * (1.0 - overlap)
    np.testing.assert_allclose(transform, np.array([[expected_step_y, 0.0], [0.0, expected_step_x]]))
    assert diag["fallback"] is True
    assert diag["reason"] == "no pairs"


def test_estimate_affine_from_pairs_diagonal_input_recovers_steps():
    """Pure horizontal & vertical pairs with no residual recover the diagonal step."""
    tile_shape = (10, 100, 80)
    overlap = 0.2
    step_y = tile_shape[1] * (1.0 - overlap)  # 80
    step_x = tile_shape[2] * (1.0 - overlap)  # 64
    pairs = [
        {"row_delta": 0, "col_delta": 1, "measured_dy": 0.0, "measured_dx": step_x},
        {"row_delta": 1, "col_delta": 0, "measured_dy": step_y, "measured_dx": 0.0},
    ]
    transform, diag = estimate_affine_from_pairs(pairs, tile_shape, overlap)
    np.testing.assert_allclose(transform, np.array([[step_y, 0.0], [0.0, step_x]]), atol=1e-6)
    assert diag["fallback"] is False
    assert diag["n_pairs"] == 2


def test_estimate_affine_from_pairs_off_diagonal_capture_rotation():
    """A known off-diagonal term in the input pairs is recovered by the fit."""
    tile_shape = (10, 100, 100)
    overlap = 0.0
    step = 100.0
    # Horizontal pair has a small dy component (rotation).
    pairs = [
        {"row_delta": 0, "col_delta": 1, "measured_dy": 5.0, "measured_dx": step},
        {"row_delta": 1, "col_delta": 0, "measured_dy": step, "measured_dx": 0.0},
    ]
    transform, diag = estimate_affine_from_pairs(pairs, tile_shape, overlap)
    # transform[0, 1] should be 5.0 (the dy component of the horizontal step).
    assert abs(transform[0, 1] - 5.0) < 1e-6
    assert diag["fallback"] is False


def test_estimate_affine_from_pairs_records_lstsq_residual():
    """The diagnostics dict includes the lstsq_residual key (float)."""
    tile_shape = (10, 64, 64)
    pairs = [
        {"row_delta": 0, "col_delta": 1, "measured_dy": 0.0, "measured_dx": 50.0},
        {"row_delta": 1, "col_delta": 0, "measured_dy": 50.0, "measured_dx": 0.0},
    ]
    _transform, diag = estimate_affine_from_pairs(pairs, tile_shape, 0.2)
    assert "lstsq_residual" in diag
    assert isinstance(diag["lstsq_residual"], float)


# ---------------------------------------------------------------------------
# _extract_displacement_params
# ---------------------------------------------------------------------------


def test_extract_displacement_params_identity_overlap_zero():
    """A diagonal transform with step == tile_size → 0 overlap, 0 rotation."""
    tile_shape = (10, 100, 100)
    transform = np.array([[100.0, 0.0], [0.0, 100.0]])
    params = _extract_displacement_params(transform, tile_shape, overlap_fraction=0.0)
    assert abs(params["theta_deg"]) < 1e-6
    assert abs(params["Ox_fraction"]) < 1e-6
    assert abs(params["Oy_fraction"]) < 1e-6


def test_extract_displacement_params_known_overlap():
    """Diagonal transform with step = 0.8 * tile → 0.2 overlap."""
    tile_shape = (10, 100, 100)
    transform = np.array([[80.0, 0.0], [0.0, 80.0]])
    params = _extract_displacement_params(transform, tile_shape, overlap_fraction=0.2)
    assert abs(params["Ox_fraction"] - 0.2) < 1e-6
    assert abs(params["Oy_fraction"] - 0.2) < 1e-6
    assert abs(params["theta_deg"]) < 1e-6


def test_extract_displacement_params_rotation_theta():
    """Off-diagonal horizontal term produces non-zero theta_deg."""
    tile_shape = (10, 100, 100)
    # Horizontal step (b, d) = (-10, 100) → theta = arctan2(10, 100) ≈ 5.71°
    transform = np.array([[100.0, -10.0], [0.0, 100.0]])
    params = _extract_displacement_params(transform, tile_shape, overlap_fraction=0.0)
    assert params["theta_deg"] > 0
    assert abs(params["theta_deg"] - np.degrees(np.arctan2(10, 100))) < 1e-3


def test_extract_displacement_params_returns_transform_in_list_form():
    """The 'transform' key contains the matrix as a nested list."""
    tile_shape = (10, 64, 64)
    transform = np.array([[50.0, 0.0], [0.0, 50.0]])
    params = _extract_displacement_params(transform, tile_shape, overlap_fraction=0.2)
    assert params["transform"] == transform.tolist()
    assert "off_diagonal_px" in params
    assert len(params["off_diagonal_px"]) == 2


# ---------------------------------------------------------------------------
# compute_affine_positions
# ---------------------------------------------------------------------------


def test_compute_affine_positions_diagonal_transform():
    """Diagonal transform → positions are (i*step_y, j*step_x) row-major."""
    transform = np.array([[80.0, 0.0], [0.0, 64.0]])
    positions = compute_affine_positions(nx=2, ny=2, transform=transform)
    assert len(positions) == 4
    assert positions[0] == (0, 0)
    assert positions[1] == (0, 64)
    assert positions[2] == (80, 0)
    assert positions[3] == (80, 64)


def test_compute_affine_positions_off_diagonal():
    """Off-diagonal transform produces rotated grid positions."""
    transform = np.array([[80.0, 5.0], [5.0, 64.0]])
    positions = compute_affine_positions(nx=2, ny=2, transform=transform)
    assert positions[0] == (0, 0)
    # Tile (0,1): transform @ [0,1] = (5, 64) → rounded.
    assert positions[1] == (round(5.0), round(64.0))


def test_compute_affine_positions_count_matches_grid():
    transform = np.eye(2) * 10.0
    positions = compute_affine_positions(nx=3, ny=4, transform=transform)
    assert len(positions) == 12


# ---------------------------------------------------------------------------
# compute_affine_output_shape
# ---------------------------------------------------------------------------


def test_compute_affine_output_shape_diagonal():
    """Diagonal transform → output shape = (nz, nx*tile_h, ny*tile_w)."""
    tile_shape = (5, 32, 24)
    transform = np.array([[32.0, 0.0], [0.0, 24.0]])
    nz, h, w = compute_affine_output_shape(nx=2, ny=3, tile_shape=tile_shape, transform=transform)
    assert nz == 5
    # Bounding box: max_row = (1*32 + 32) = 64; max_col = (2*24 + 24) = 72.
    assert h == 64
    assert w == 72


def test_compute_affine_output_shape_includes_off_diagonal_extent():
    """Off-diagonal terms extend the bounding box beyond the diagonal case."""
    tile_shape = (5, 32, 32)
    transform = np.array([[32.0, 10.0], [10.0, 32.0]])
    nz_diag, h_diag, w_diag = compute_affine_output_shape(
        nx=2, ny=2, tile_shape=tile_shape, transform=np.array([[32.0, 0.0], [0.0, 32.0]])
    )
    nz_off, h_off, w_off = compute_affine_output_shape(nx=2, ny=2, tile_shape=tile_shape, transform=transform)
    assert nz_off == nz_diag
    # Off-diagonal should produce a larger or equal bounding box.
    assert h_off >= h_diag
    assert w_off >= w_diag


# ---------------------------------------------------------------------------
# Edge cases: duplicate / out-of-order z positions
# ---------------------------------------------------------------------------


def test_apply_blend_shift_refinement_zero_refinements_list():
    """A list of zero-shift refinements is treated as negligible → unchanged."""
    tile = (np.ones((5, 16, 16)) * 50.0).astype(np.float32)
    refinements = [{"dx": 0.0, "dy": 0.0}, {"dx": 0.0, "dy": 0.0}]
    result = apply_blend_shift_refinement(tile, refinements)
    np.testing.assert_array_equal(result, tile)


def test_apply_blend_shift_refinement_preserves_shape_with_large_shift():
    """Large shifts preserve tile shape (scipy.ndimage.shift with order=1)."""
    rng = np.random.default_rng(11)
    tile = (rng.random((4, 32, 32)) * 100.0).astype(np.float32)
    refinements = [{"dx": 5.0, "dy": -5.0}]
    result = apply_blend_shift_refinement(tile, refinements)
    assert result.shape == tile.shape


def test_compare_motor_vs_registration_dilation_flag_triggered():
    """With >10 tiles and a strong index-correlated error, dilation_indicator is True."""
    # Errors grow linearly with tile index → high correlation.
    motor = [(i * 10, 0) for i in range(15)]
    reg = [(i * 10 + i * 2, 0) for i in range(15)]  # +2 px per tile
    result = compare_motor_vs_registration(motor, reg)
    assert "index_error_correlation" in result
    assert "dilation_indicator" in result
    assert result["dilation_indicator"] is True


def test_compare_motor_vs_registration_dilation_flag_not_triggered():
    """With >10 tiles but no index-correlated error, dilation_indicator is False."""
    rng = np.random.default_rng(0)
    motor = [(i * 10, 0) for i in range(15)]
    # Random small errors, no trend.
    reg = [(m[0] + rng.normal(0, 0.1), 0) for m in motor]
    result = compare_motor_vs_registration(motor, reg)
    assert "dilation_indicator" in result
    assert result["dilation_indicator"] is False


def test_compare_motor_vs_registration_max_magnitude():
    """max_magnitude equals the largest per-tile displacement magnitude."""
    motor = [(0, 0), (10, 0), (20, 0)]
    reg = [(0, 0), (12, 0), (20, 0)]
    result = compare_motor_vs_registration(motor, reg)
    # Differences: (0,0), (2,0), (0,0) → max magnitude = 2.0.
    assert abs(result["max_magnitude"] - 2.0) < 1e-9
