"""Tests for :mod:`linumpy.metrics.collectors` (step-specific metric collectors).

Exercises each ``collect_*`` function with small synthetic numpy arrays /
dicts and verifies the returned :class:`PipelineMetrics` has the expected
shape, info fields, and metric values. Uses ``tmp_path`` so ``finalize()``
can write its JSON file.
"""

import numpy as np
import pandas as pd
import pytest

from linumpy.metrics.collectors import (
    collect_auto_exclude_metrics,
    collect_common_space_metrics,
    collect_interface_crop_metrics,
    collect_normalization_metrics,
    collect_pairwise_registration_metrics,
    collect_psf_compensation_metrics,
    collect_quality_assessment_metrics,
    collect_rehoming_metrics,
    collect_slice_interpolation_metrics,
    collect_stack_metrics,
    collect_stitch_3d_metrics,
    collect_xy_transform_metrics,
)

# ---------------------------------------------------------------------------
# collect_normalization_metrics
# ---------------------------------------------------------------------------


def test_collect_normalization_metrics_basic(tmp_path):
    vol = np.ones((4, 4, 4), dtype=np.float32) * 100
    agarose_mask = np.ones((4, 4, 4), dtype=bool)
    otsu = 50.0
    bg_thresholds = np.array([10.0, 12.0, 11.0, 13.0])
    out = tmp_path / "vol.zarr"
    metrics = collect_normalization_metrics(vol, agarose_mask, otsu, bg_thresholds, out)
    assert metrics.step_name == "normalize_intensities"
    assert "agarose_coverage" in metrics.metrics
    np.testing.assert_allclose(metrics.metrics["agarose_coverage"]["value"], 1.0)
    assert metrics.metrics["otsu_threshold"]["value"] == 50.0
    assert metrics.metrics["mean_background"]["value"] == pytest.approx(11.5)
    assert "output_volume" in metrics.metrics


def test_collect_normalization_metrics_with_input_and_params(tmp_path):
    vol = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2, 2), dtype=bool)
    out = tmp_path / "vol.zarr"
    metrics = collect_normalization_metrics(
        vol, mask, 30.0, np.array([5.0, 5.0]), out, input_path=tmp_path / "in.zarr", params={"foo": "bar"}
    )
    assert metrics.metrics["agarose_coverage"]["value"] == 0.0
    assert "input_volume" in metrics.metrics
    assert metrics.metrics["foo"]["value"] == "bar"


def test_collect_normalization_metrics_zero_array(tmp_path):
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=bool)
    out = tmp_path / "vol.zarr"
    # all-zero mask → coverage 0; should not raise
    metrics = collect_normalization_metrics(vol, mask, 0.0, np.zeros(4), out)
    assert metrics.metrics["agarose_coverage"]["value"] == 0.0


# ---------------------------------------------------------------------------
# collect_xy_transform_metrics
# ---------------------------------------------------------------------------


def test_collect_xy_transform_metrics_basic(tmp_path):
    transform = np.array([[80.0, 0.0], [0.0, 80.0]])
    residuals = np.array([0.1, 0.2, 0.3])
    out = tmp_path / "transform.json"
    metrics = collect_xy_transform_metrics(transform, 12, (100, 100), residuals, out)
    assert metrics.step_name == "xy_transform_estimation"
    assert metrics.metrics["tile_pairs_used"]["value"] == 12
    assert metrics.metrics["transform_00"]["value"] == 80.0
    assert metrics.metrics["rms_residual"]["value"] == pytest.approx(np.sqrt(np.mean(residuals)))
    # overlap = 1 - 80/100 = 0.2
    np.testing.assert_allclose(metrics.metrics["estimated_overlap_x"]["value"], 0.2)


def test_collect_xy_transform_metrics_with_tile_counts(tmp_path):
    transform = np.array([[80.0, 0.0], [0.0, 80.0]])
    residuals = np.array([0.5])
    out = tmp_path / "t.json"
    metrics = collect_xy_transform_metrics(
        transform, 6, (100, 100), residuals, out, n_tiles_x=5, n_tiles_y=4, params={"initial_overlap": 0.2}
    )
    assert "n_tiles_x" in metrics.metrics
    assert "accumulated_systematic_error_px" in metrics.metrics
    assert "accumulated_random_error_px" in metrics.metrics


def test_collect_xy_transform_metrics_empty_residuals(tmp_path):
    transform = np.eye(2) * 50.0
    residuals = np.array([])
    out = tmp_path / "t.json"
    metrics = collect_xy_transform_metrics(transform, 1, (50, 50), residuals, out)
    assert "rms_residual" not in metrics.metrics


# ---------------------------------------------------------------------------
# collect_pairwise_registration_metrics
# ---------------------------------------------------------------------------


def test_collect_pairwise_registration_metrics_basic(tmp_path):
    out = tmp_path / "reg"
    metrics = collect_pairwise_registration_metrics(
        registration_error=0.1, tx=5.0, ty=3.0, rotation_deg=0.5, best_z_index=10, expected_z_index=10, output_path=out
    )
    assert metrics.step_name == "pairwise_registration"
    assert metrics.metrics["translation_x"]["value"] == 5.0
    assert metrics.metrics["translation_y"]["value"] == 3.0
    assert metrics.metrics["z_drift"]["value"] == 0
    assert "registration_confidence" in metrics.metrics


def test_collect_pairwise_registration_metrics_with_paths_and_params(tmp_path):
    out = tmp_path / "reg"
    metrics = collect_pairwise_registration_metrics(
        0.2,
        10.0,
        5.0,
        1.0,
        8,
        10,
        out,
        fixed_path=tmp_path / "fixed",
        moving_path=tmp_path / "moving",
        params={"max_translation_px": 100.0},
        z_correlation=0.8,
    )
    assert "fixed_volume" in metrics.metrics
    assert "moving_volume" in metrics.metrics
    assert metrics.metrics["z_correlation"]["value"] == 0.8
    # confidence should be high with good z_corr and small translation
    assert metrics.metrics["registration_confidence"]["value"] > 0.5


def test_collect_pairwise_registration_metrics_negative_z_corr_clamped(tmp_path):
    out = tmp_path / "reg"
    metrics = collect_pairwise_registration_metrics(0.1, 1.0, 1.0, 0.0, 5, 5, out, z_correlation=-0.5)
    assert metrics.metrics["z_correlation"]["value"] == 0.0


def test_collect_pairwise_registration_metrics_z_drift(tmp_path):
    out = tmp_path / "reg"
    metrics = collect_pairwise_registration_metrics(0.1, 1.0, 1.0, 0.0, 12, 10, out)
    assert metrics.metrics["z_drift"]["value"] == 2


# ---------------------------------------------------------------------------
# collect_interface_crop_metrics
# ---------------------------------------------------------------------------


def test_collect_interface_crop_metrics_ok(tmp_path):
    out = tmp_path / "vol.zarr"
    metrics = collect_interface_crop_metrics(
        detected_interface=50,
        crop_depth_px=100,
        start_idx=0,
        end_idx=100,
        input_shape=(200, 10, 10),
        output_shape=(100, 10, 10),
        resolution_um=6.5,
        output_path=out,
    )
    assert metrics.step_name == "crop_interface"
    assert metrics.metrics["detected_interface_depth"]["value"] == 50
    assert metrics.metrics["detected_interface_depth_um"]["value"] == pytest.approx(325.0)
    assert metrics.metrics["interface_quality"]["value"] == "ok"


def test_collect_interface_crop_metrics_warning_too_shallow(tmp_path):
    out = tmp_path / "vol.zarr"
    metrics = collect_interface_crop_metrics(
        detected_interface=1,
        crop_depth_px=100,
        start_idx=0,
        end_idx=100,
        input_shape=(200, 10, 10),
        output_shape=(100, 10, 10),
        resolution_um=6.5,
        output_path=out,
    )
    assert metrics.metrics["interface_quality"]["value"] == "warning"


def test_collect_interface_crop_metrics_warning_too_deep(tmp_path):
    out = tmp_path / "vol.zarr"
    metrics = collect_interface_crop_metrics(
        detected_interface=180,
        crop_depth_px=100,
        start_idx=0,
        end_idx=100,
        input_shape=(200, 10, 10),
        output_shape=(100, 10, 10),
        resolution_um=6.5,
        output_path=out,
    )
    assert metrics.metrics["interface_quality"]["value"] == "warning"


def test_collect_interface_crop_metrics_with_input_path(tmp_path):
    out = tmp_path / "vol.zarr"
    metrics = collect_interface_crop_metrics(
        50,
        100,
        0,
        100,
        (200, 10, 10),
        (100, 10, 10),
        6.5,
        out,
        input_path=tmp_path / "in.zarr",
        padding_needed=True,
    )
    assert "input_volume" in metrics.metrics
    assert metrics.metrics["padding_needed"]["value"] is True


# ---------------------------------------------------------------------------
# collect_psf_compensation_metrics
# ---------------------------------------------------------------------------


def test_collect_psf_compensation_metrics_good(tmp_path):
    psf = np.zeros(100)
    psf[50] = 1.0
    out = tmp_path / "vol.zarr"
    metrics = collect_psf_compensation_metrics(psf, 0.5, out)
    assert metrics.step_name == "psf_compensation"
    assert metrics.metrics["psf_max"]["value"] == 1.0
    assert metrics.metrics["psf_peak_depth"]["value"] == 50
    assert metrics.metrics["profile_quality"]["value"] == "good"


def test_collect_psf_compensation_metrics_poor(tmp_path):
    psf = np.full(100, 0.01)
    out = tmp_path / "vol.zarr"
    metrics = collect_psf_compensation_metrics(psf, 0.5, out)
    assert metrics.metrics["profile_quality"]["value"] == "poor"


def test_collect_psf_compensation_metrics_warning_peak_depth(tmp_path):
    psf = np.zeros(100)
    psf[2] = 1.0  # peak near start
    out = tmp_path / "vol.zarr"
    metrics = collect_psf_compensation_metrics(psf, 0.5, out)
    assert metrics.metrics["profile_quality"]["value"] == "warning"


def test_collect_psf_compensation_metrics_with_input_and_gaussian(tmp_path):
    psf = np.zeros(50)
    psf[25] = 1.0
    out = tmp_path / "vol.zarr"
    metrics = collect_psf_compensation_metrics(psf, 0.3, out, input_path=tmp_path / "in", fit_gaussian=True)
    assert "input_volume" in metrics.metrics
    assert metrics.metrics["fit_gaussian"]["value"] is True


# ---------------------------------------------------------------------------
# collect_stack_metrics
# ---------------------------------------------------------------------------


def test_collect_stack_metrics_basic(tmp_path):
    out = tmp_path / "stack.zarr"
    z_offsets = np.array([10.0, 11.0, 12.0, 10.5])
    metrics = collect_stack_metrics((100, 10, 10), z_offsets, 4, [6.5, 6.5, 6.5], out)
    assert metrics.step_name == "stack_slices"
    assert metrics.metrics["total_z_depth"]["value"] == 100
    assert metrics.metrics["mean_z_offset"]["value"] == pytest.approx(np.mean(z_offsets))
    assert metrics.metrics["z_offset_range"]["value"] == pytest.approx(2.0)


def test_collect_stack_metrics_with_z_matches_df(tmp_path):
    out = tmp_path / "stack.zarr"
    z_offsets = np.array([10.0, 11.0])
    df = pd.DataFrame({"correlation": [0.9, 0.8, 0.7]})
    metrics = collect_stack_metrics((50, 5, 5), z_offsets, 2, [6.5, 6.5, 6.5], out, z_matches_df=df)
    assert "mean_z_correlation" in metrics.metrics
    assert metrics.metrics["mean_z_correlation"]["value"] == pytest.approx(0.8)


def test_collect_stack_metrics_with_decisions_df(tmp_path):
    out = tmp_path / "stack.zarr"
    z_offsets = np.array([10.0])
    df = pd.DataFrame(
        {
            "transform_loaded": [True, False, True],
            "manual_override": [False, False, True],
            "overlap_source": ["motor", "image", "motor"],
        }
    )
    metrics = collect_stack_metrics((50, 5, 5), z_offsets, 3, [6.5, 6.5, 6.5], out, decisions_df=df)
    assert metrics.metrics["n_transform_loaded"]["value"] == 2
    assert metrics.metrics["n_transform_missing"]["value"] == 1
    assert metrics.metrics["n_manual_override"]["value"] == 1


def test_collect_stack_metrics_empty_dfs(tmp_path):
    out = tmp_path / "stack.zarr"
    z_offsets = np.array([10.0])
    metrics = collect_stack_metrics(
        (50, 5, 5),
        z_offsets,
        1,
        [6.5, 6.5, 6.5],
        out,
        z_matches_df=pd.DataFrame(),
        decisions_df=pd.DataFrame(),
    )
    assert "mean_z_correlation" not in metrics.metrics


# ---------------------------------------------------------------------------
# collect_quality_assessment_metrics
# ---------------------------------------------------------------------------


def test_collect_quality_assessment_metrics_basic(tmp_path):
    out = tmp_path / "slice_config.csv"
    quality_results = {0: {"overall": 0.9, "has_data": True}, 1: {"overall": 0.4, "has_data": True}}
    metrics = collect_quality_assessment_metrics(out, quality_results, [1], 0.5)
    assert metrics.step_name == "slice_quality_assessment"
    assert metrics.metrics["mean_quality"]["value"] == pytest.approx(0.65)
    assert metrics.metrics["min_quality"]["value"] == pytest.approx(0.4)
    assert metrics.metrics["n_below_threshold"]["value"] == 1


def test_collect_quality_assessment_metrics_empty(tmp_path):
    out = tmp_path / "slice_config.csv"
    metrics = collect_quality_assessment_metrics(out, {}, [], 0.5)
    assert "mean_quality" not in metrics.metrics


def test_collect_quality_assessment_metrics_no_data_skipped(tmp_path):
    out = tmp_path / "slice_config.csv"
    quality_results = {0: {"overall": 0.9, "has_data": False}}
    metrics = collect_quality_assessment_metrics(out, quality_results, [], 0.5)
    assert "mean_quality" not in metrics.metrics


# ---------------------------------------------------------------------------
# collect_rehoming_metrics
# ---------------------------------------------------------------------------


def test_collect_rehoming_metrics_basic(tmp_path):
    out = tmp_path / "shifts.csv"
    metrics = collect_rehoming_metrics(out, 10, [1, 2], [3], 1, max_correction_mm=0.5)
    assert metrics.step_name == "rehoming_detection"
    assert metrics.metrics["n_total_transitions"]["value"] == 10
    assert metrics.metrics["n_tile_corrected"]["value"] == 2
    assert metrics.metrics["n_spike_corrected"]["value"] == 1
    assert metrics.metrics["n_unreliable"]["value"] == 1
    assert metrics.metrics["max_correction_mm"]["value"] == 0.5


def test_collect_rehoming_metrics_no_max_correction(tmp_path):
    out = tmp_path / "shifts.csv"
    metrics = collect_rehoming_metrics(out, 5, [], [], 0)
    assert "max_correction_mm" not in metrics.metrics


# ---------------------------------------------------------------------------
# collect_auto_exclude_metrics
# ---------------------------------------------------------------------------


def test_collect_auto_exclude_metrics_basic(tmp_path):
    out = tmp_path / "slice_config.csv"
    metrics = collect_auto_exclude_metrics(out, 20, [3, 4, 5], 2, 0.5, 3)
    assert metrics.step_name == "auto_exclude"
    assert metrics.metrics["num_total_slices"]["value"] == 20
    assert metrics.metrics["num_auto_excluded"]["value"] == 3
    assert metrics.metrics["num_clusters"]["value"] == 2
    assert metrics.metrics["z_corr_threshold"]["value"] == 0.5
    assert metrics.metrics["consecutive_threshold"]["value"] == 3


# ---------------------------------------------------------------------------
# collect_common_space_metrics
# ---------------------------------------------------------------------------


def test_collect_common_space_metrics_basic(tmp_path):
    out = tmp_path / "aligned"
    metrics = collect_common_space_metrics(out, 10, 2, 1, 5, 1)
    assert metrics.step_name == "common_space_alignment"
    assert metrics.metrics["n_selected_slices"]["value"] == 10
    assert metrics.metrics["n_excluded_slices"]["value"] == 2


def test_collect_common_space_metrics_with_discrepancies(tmp_path):
    out = tmp_path / "aligned"
    metrics = collect_common_space_metrics(out, 10, 2, 1, 5, 1, refine_discrepancies_px=[1.0, 2.0, 3.0])
    assert "mean_refine_discrepancy_px" in metrics.metrics
    assert metrics.metrics["mean_refine_discrepancy_px"]["value"] == pytest.approx(2.0)


def test_collect_common_space_metrics_empty_discrepancies(tmp_path):
    out = tmp_path / "aligned"
    metrics = collect_common_space_metrics(out, 10, 2, 1, 5, 1, refine_discrepancies_px=[])
    assert "mean_refine_discrepancy_px" not in metrics.metrics


# ---------------------------------------------------------------------------
# collect_slice_interpolation_metrics
# ---------------------------------------------------------------------------


def test_collect_slice_interpolation_metrics_basic(tmp_path):
    out = tmp_path / "slice_config.csv"
    metrics = collect_slice_interpolation_metrics(out, 3, ["z00", "z01"], ["z02"])
    assert metrics.step_name == "slice_interpolation"
    assert metrics.metrics["n_fragments"]["value"] == 3
    assert metrics.metrics["n_interpolated"]["value"] == 2
    assert metrics.metrics["n_failed"]["value"] == 1


def test_collect_slice_interpolation_metrics_with_histograms(tmp_path):
    out = tmp_path / "slice_config.csv"
    metrics = collect_slice_interpolation_metrics(
        out,
        3,
        ["z00"],
        ["z01"],
        fallback_reasons={"low_ncc": 1},
        method_counts={"zmorph": 1},
    )
    assert metrics.metrics["fallback_reasons"]["value"] == {"low_ncc": 1}
    assert metrics.metrics["method_counts"]["value"] == {"zmorph": 1}


# ---------------------------------------------------------------------------
# collect_stitch_3d_metrics
# ---------------------------------------------------------------------------


def test_collect_stitch_3d_metrics_basic(tmp_path):
    out = tmp_path / "stitched.zarr"
    metrics = collect_stitch_3d_metrics((100, 100, 100), (80, 80, 80), 4, [6.5, 6.5, 6.5], out)
    assert metrics.step_name == "stitch_3d"
    assert metrics.metrics["overlap_reduction"]["value"] == pytest.approx(1.0 - (80**3 / 100**3))
    assert metrics.metrics["num_tiles"]["value"] == 4


def test_collect_stitch_3d_metrics_with_input(tmp_path):
    out = tmp_path / "stitched.zarr"
    metrics = collect_stitch_3d_metrics(
        (100, 100, 100), (80, 80, 80), 4, [6.5, 6.5, 6.5], out, input_path=tmp_path / "in", blending_method="linear"
    )
    assert "input_volume" in metrics.metrics
    assert metrics.metrics["blending_method"]["value"] == "linear"


def test_collect_stitch_3d_metrics_zero_input_pixels(tmp_path):
    out = tmp_path / "stitched.zarr"
    metrics = collect_stitch_3d_metrics((0, 0, 0), (80, 80, 80), 4, [6.5, 6.5, 6.5], out)
    assert metrics.metrics["overlap_reduction"]["value"] == 0.0


# ---------------------------------------------------------------------------
# Edge case: all-zero / degenerate inputs
# ---------------------------------------------------------------------------


def test_collect_normalization_metrics_degenerate_all_zero(tmp_path):
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=bool)
    out = tmp_path / "vol.zarr"
    # mean of zeros is 0; should not raise
    metrics = collect_normalization_metrics(vol, mask, 0.0, np.zeros(4), out)
    assert metrics.metrics["mean_background"]["value"] == 0.0
