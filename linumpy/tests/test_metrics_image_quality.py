"""Tests for :mod:`linumpy.metrics.image_quality` (CPU image-quality metrics)."""

import numpy as np
import pytest

from linumpy.metrics.image_quality import (
    assess_slice_quality,
    compute_edge_score,
    compute_normalized_cross_correlation,
    compute_quality_report,
    compute_ssim_2d,
    compute_ssim_3d,
    compute_variance_score,
    detect_calibration_slice,
    normalize_image,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sharp_image() -> np.ndarray:
    """A small synthetic sharp image (a step edge with noise)."""
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float32)
    img[:, 16:] = 1.0
    img += rng.normal(0.0, 0.01, img.shape).astype(np.float32)
    return img


@pytest.fixture
def blurred_image(sharp_image: np.ndarray) -> np.ndarray:
    """A blurred version of ``sharp_image`` (Gaussian-like smoothing)."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(sharp_image, sigma=2.0).astype(np.float32)


@pytest.fixture
def small_volume() -> np.ndarray:
    """A small (Z, Y, X) volume with a ramp along Z."""
    rng = np.random.default_rng(1)
    vol = np.zeros((6, 16, 16), dtype=np.float32)
    for z in range(6):
        vol[z] = float(z) / 5.0
    vol += rng.normal(0.0, 0.01, vol.shape).astype(np.float32)
    return vol


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage:
    def test_normalize_to_unit_range(self) -> None:
        img = np.array([[0.0, 10.0], [5.0, 20.0]], dtype=np.float32)
        out = normalize_image(img)
        assert out.dtype == np.float32
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_constant_image_unchanged(self) -> None:
        img = np.full((4, 4), 7.5, dtype=np.float32)
        out = normalize_image(img)
        assert out.shape == img.shape
        # When max == min, normalize_image skips division — returns the constant as float32
        assert np.all(out == 7.5)

    def test_preserves_shape_and_dtype(self) -> None:
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = normalize_image(img)
        assert out.shape == (3, 4)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_normalized_cross_correlation
# ---------------------------------------------------------------------------


class TestNormalizedCrossCorrelation:
    def test_identical_images_correlation_one(self, sharp_image: np.ndarray) -> None:
        ncc = compute_normalized_cross_correlation(sharp_image, sharp_image)
        assert ncc == pytest.approx(1.0, abs=1e-6)

    def test_independent_images_low_correlation(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
        b = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
        ncc = compute_normalized_cross_correlation(a, b)
        assert -0.5 < ncc < 0.5

    def test_mismatched_shapes_crops_to_common(self) -> None:
        a = np.ones((10, 12), dtype=np.float32)
        b = np.ones((8, 14), dtype=np.float32)
        ncc = compute_normalized_cross_correlation(a, b)
        # Constant images -> NaN after normalize (zero variance)
        assert np.isnan(ncc) or ncc == 0.0

    def test_empty_input_returns_nan(self) -> None:
        ncc = compute_normalized_cross_correlation(np.zeros((0, 4)), np.zeros((0, 4)))
        assert np.isnan(ncc)


# ---------------------------------------------------------------------------
# compute_ssim_2d
# ---------------------------------------------------------------------------


class TestSSIM2D:
    def test_identical_images_ssim_one(self, sharp_image: np.ndarray) -> None:
        score = compute_ssim_2d(sharp_image, sharp_image)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_sharp_gt_blurred(self, sharp_image: np.ndarray, blurred_image: np.ndarray) -> None:
        score_same = compute_ssim_2d(sharp_image, sharp_image)
        score_diff = compute_ssim_2d(sharp_image, blurred_image)
        assert score_same > score_diff
        assert 0.0 <= score_diff <= 1.0

    def test_mismatched_shapes_crops(self) -> None:
        a = np.zeros((16, 20), dtype=np.float32)
        a[:, 10:] = 1.0
        b = np.zeros((14, 18), dtype=np.float32)
        b[:, 9:] = 1.0
        score = compute_ssim_2d(a, b)
        assert 0.0 <= score <= 1.0

    def test_tiny_image_falls_back(self) -> None:
        # 2x2 image — too small for SSIM window, should fall back to NCC
        a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        score = compute_ssim_2d(a, a)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_ssim_3d
# ---------------------------------------------------------------------------


class TestSSIM3D:
    def test_identical_volumes_ssim_one(self, small_volume: np.ndarray) -> None:
        score = compute_ssim_3d(small_volume, small_volume)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_sample_depth_subset(self, small_volume: np.ndarray) -> None:
        score_full = compute_ssim_3d(small_volume, small_volume, sample_depth=0)
        score_subset = compute_ssim_3d(small_volume, small_volume, sample_depth=3)
        assert 0.0 <= score_subset <= 1.0
        assert score_full == pytest.approx(1.0, abs=1e-6)

    def test_xy_roi_center_crop(self, small_volume: np.ndarray) -> None:
        score = compute_ssim_3d(small_volume, small_volume, xy_roi=8)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_edge_score
# ---------------------------------------------------------------------------


class TestEdgeScore:
    def test_identical_edges_score_high(self, sharp_image: np.ndarray) -> None:
        score = compute_edge_score(sharp_image, sharp_image)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_different_images_lower_score(self, sharp_image: np.ndarray, blurred_image: np.ndarray) -> None:
        score_same = compute_edge_score(sharp_image, sharp_image)
        score_diff = compute_edge_score(sharp_image, blurred_image)
        assert score_same >= score_diff

    def test_3d_volume_uses_middle_slice(self, small_volume: np.ndarray) -> None:
        score = compute_edge_score(small_volume, small_volume)
        assert 0.0 <= score <= 1.0

    def test_3d_volume_explicit_sample_z(self, small_volume: np.ndarray) -> None:
        score = compute_edge_score(small_volume, small_volume, sample_z=0)
        assert 0.0 <= score <= 1.0

    def test_constant_image_returns_zero(self) -> None:
        const = np.zeros((16, 16), dtype=np.float32)
        score = compute_edge_score(const, const)
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_variance_score
# ---------------------------------------------------------------------------


class TestVarianceScore:
    def test_equal_variance_score_one(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 2.0, (16, 16)).astype(np.float32)
        b = rng.normal(0.0, 2.0, (16, 16)).astype(np.float32)
        score = compute_variance_score(a, b)
        assert 0.0 <= score <= 1.0
        # Similar variances -> high score
        assert score > 0.5

    def test_zero_reference_returns_zero(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float32)
        assert compute_variance_score(a, b) == 0.0

    def test_very_different_variance_low_score(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
        b = rng.normal(0.0, 100.0, (16, 16)).astype(np.float32)
        score = compute_variance_score(a, b)
        assert 0.0 <= score <= 1.0
        assert score < 0.5


# ---------------------------------------------------------------------------
# assess_slice_quality
# ---------------------------------------------------------------------------


class TestAssessSliceQuality:
    def test_no_neighbors_returns_zero_overall(self, small_volume: np.ndarray) -> None:
        score, metrics = assess_slice_quality(small_volume, None, None)
        assert score == 0.0
        assert metrics["ssim_mean"] == 0.0
        assert metrics["has_data"] is True

    def test_with_neighbors(self, small_volume: np.ndarray) -> None:
        before = small_volume.copy()
        after = small_volume.copy()
        score, metrics = assess_slice_quality(small_volume, before, after)
        assert 0.0 <= score <= 1.0
        assert metrics["ssim_before"] > 0.0
        assert metrics["ssim_after"] > 0.0
        assert "overall" in metrics

    def test_no_data_volume(self) -> None:
        const = np.full((4, 8, 8), 3.0, dtype=np.float32)
        score, metrics = assess_slice_quality(const, None, None)
        assert score == 0.0
        assert metrics["has_data"] is False

    def test_custom_weights(self, small_volume: np.ndarray) -> None:
        before = small_volume.copy()
        weights = {"ssim": 0.7, "edge": 0.2, "variance": 0.1}
        score, _metrics = assess_slice_quality(small_volume, before, None, weights=weights)
        assert 0.0 <= score <= 1.0

    def test_xy_roi_crop(self, small_volume: np.ndarray) -> None:
        before = small_volume.copy()
        score, metrics = assess_slice_quality(small_volume, before, None, xy_roi=8)
        assert 0.0 <= score <= 1.0
        assert "overall" in metrics


# ---------------------------------------------------------------------------
# detect_calibration_slice
# ---------------------------------------------------------------------------


class TestDetectCalibrationSlice:
    def test_no_calibration_in_uniform_volumes(self) -> None:
        volumes = {i: np.zeros((10, 4, 4), dtype=np.float32) for i in range(5)}
        result = detect_calibration_slice(volumes)
        assert result == []

    def test_detects_thick_first_slice(self) -> None:
        volumes = {0: np.zeros((30, 4, 4), dtype=np.float32)}
        for i in range(1, 6):
            volumes[i] = np.zeros((10, 4, 4), dtype=np.float32)
        result = detect_calibration_slice(volumes, thickness_ratio=1.5)
        assert 0 in result

    def test_empty_dict_returns_empty(self) -> None:
        assert detect_calibration_slice({}) == []

    def test_zero_depths_filtered(self) -> None:
        volumes = {0: np.zeros((0, 4, 4), dtype=np.float32), 1: np.zeros((10, 4, 4), dtype=np.float32)}
        assert detect_calibration_slice(volumes) == []


# ---------------------------------------------------------------------------
# compute_quality_report
# ---------------------------------------------------------------------------


class TestComputeQualityReport:
    def test_empty_report(self) -> None:
        report = compute_quality_report({})
        assert "error" in report

    def test_basic_report(self) -> None:
        qualities = {
            0: {"overall": 0.9, "has_data": True},
            1: {"overall": 0.4, "has_data": True},
            2: {"overall": 0.0, "has_data": False},
        }
        report = compute_quality_report(qualities, min_quality=0.5)
        assert report["n_slices"] == 3
        assert report["mean_quality"] == pytest.approx(0.4333, abs=1e-3)
        assert 1 in report["low_quality_slices"]
        assert 2 in report["no_data_slices"]
        assert 0 not in report["low_quality_slices"]

    def test_all_above_threshold(self) -> None:
        qualities = {i: {"overall": 0.8, "has_data": True} for i in range(3)}
        report = compute_quality_report(qualities, min_quality=0.5)
        assert report["low_quality_slices"] == []
        assert report["no_data_slices"] == []
