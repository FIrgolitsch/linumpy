"""Tests for linumpy.stack_alignment.motor_stack."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from linumpy.stack_alignment.motor_stack import (
    accumulate_pairwise_translations,
    common_space_xy_policy,
    compute_output_shape,
    load_registration_transforms,
)

# ---------------------------------------------------------------------------
# compute_output_shape
# ---------------------------------------------------------------------------


def test_compute_output_shape_single_slice_no_shift():
    ny, nx, x0, y0 = compute_output_shape({0: "slice0"}, {0: (0.0, 0.0)}, (10, 20, 30))
    assert (ny, nx) == (20, 30)
    assert (x0, y0) == (0, 0)


def test_compute_output_shape_expands_for_negative_offset():
    cumsum_px = {0: (0.0, 0.0), 1: (5.0, -3.0)}
    ny, nx, x0, y0 = compute_output_shape({0: "slice0", 1: "slice1"}, cumsum_px, (10, 20, 30))
    assert (ny, nx) == (23, 35)
    assert (x0, y0) == (0, -3)


# ---------------------------------------------------------------------------
# load_registration_transforms
# ---------------------------------------------------------------------------


def _write_transform_dir(tmp_path: Path, slice_id: int, metrics: dict | None) -> None:
    transform_dir = tmp_path / f"slice_z{slice_id:02d}"
    transform_dir.mkdir()
    sitk.WriteTransform(sitk.Euler2DTransform(), str(transform_dir / "transform.tfm"))
    (transform_dir / "offsets.txt").write_text("4\n8\n")
    if metrics is not None:
        (transform_dir / "pairwise_registration_metrics.json").write_text(json.dumps(metrics))


def test_load_registration_transforms_basic(tmp_path: Path):
    metrics = {
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 0.75},
            "translation_x": {"value": 2.0},
            "translation_y": {"value": -1.0},
            "z_correlation": {"value": 0.6},
            "rotation": {"value": 0.2},
        },
    }
    _write_transform_dir(tmp_path, 1, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1])

    assert transforms[1] is not None
    _tfm, fixed_z, moving_z, confidence = transforms[1]
    assert fixed_z == 4
    assert moving_z == 8
    assert abs(confidence - 0.75) < 1e-9
    assert pairwise[1] == (2.0, -1.0, 0.6)


def test_load_registration_transforms_metric_gating_rejects_low_zcorr(tmp_path: Path):
    metrics = {
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 0.9},
            "translation_x": {"value": 1.0},
            "translation_y": {"value": 1.0},
            "z_correlation": {"value": 0.1},
            "rotation": {"value": 0.1},
        },
    }
    _write_transform_dir(tmp_path, 1, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1], load_min_zcorr=0.5, load_max_rotation=1.0)

    assert transforms[1] is None
    # Translation is still recovered for accumulation even though the
    # transform itself was gated out.
    assert pairwise[1] == (1.0, 1.0, 0.1)


def test_load_registration_transforms_missing_dir_is_none(tmp_path: Path):
    transforms, pairwise = load_registration_transforms(tmp_path, [0, 1])
    assert transforms[1] is None
    assert pairwise == {}


def test_load_registration_transforms_skips_id_step_gap(tmp_path: Path):
    """z49→z51 has no shared cut face; do not apply that pairwise .tfm."""
    metrics = {
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 1.0},
            "translation_x": {"value": 200.0},
            "translation_y": {"value": 0.0},
            "z_correlation": {"value": 0.9},
            "rotation": {"value": 5.0},
        },
    }
    _write_transform_dir(tmp_path, 51, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [49, 51])

    assert transforms[51] is None
    assert 51 not in pairwise


def test_load_registration_transforms_keeps_manual_across_gap(tmp_path: Path):
    """A hand-aligned z51 transform is the gap correction; do not drop it."""
    metrics = {
        "source": "manual",
        "overall_status": "ok",
        "metrics": {
            "registration_confidence": {"value": 1.0},
            "translation_x": {"value": -74.0},
            "translation_y": {"value": -64.0},
            "z_correlation": {"value": 0.0},
            "rotation": {"value": -1.0},
        },
    }
    _write_transform_dir(tmp_path, 51, metrics)

    transforms, pairwise = load_registration_transforms(tmp_path, [49, 51])

    assert transforms[51] is not None
    assert abs(pairwise[51][0] - (-74.0)) < 1e-9
    assert abs(pairwise[51][1] - (-64.0)) < 1e-9


# ---------------------------------------------------------------------------
# accumulate_pairwise_translations
# ---------------------------------------------------------------------------


def test_accumulate_pairwise_translations_basic_cumsum():
    available_ids = [0, 1, 2]
    all_pairwise_translations = {1: (1.0, 2.0, 0.9), 2: (1.0, 2.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
    )

    assert accumulated[1] == (1.0, 2.0)
    assert accumulated[2] == (2.0, 4.0)


def test_accumulate_pairwise_translations_zcorr_filter_skips_low_confidence():
    available_ids = [0, 1, 2]
    all_pairwise_translations = {1: (1.0, 2.0, 0.05), 2: (1.0, 2.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        translation_min_zcorr=0.2,
    )

    # Slice 1's translation is below threshold and skipped -> stays at 0.
    assert accumulated[1] == (0.0, 0.0)
    # Slice 2's translation is accepted and accumulated on top of slice 1's zero.
    assert accumulated[2] == (1.0, 2.0)


def test_accumulate_pairwise_translations_boundary_exclusion():
    available_ids = [0, 1]
    all_pairwise_translations = {1: (10.0, 0.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        max_pairwise_translation=10.0,
    )

    # 10.0 >= 0.95 * 10.0 boundary -> excluded (zeroed).
    assert accumulated[1] == (0.0, 0.0)


def test_accumulate_pairwise_translations_drift_cap_clamps_magnitude():
    available_ids = [0, 1]
    all_pairwise_translations = {1: (10.0, 0.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        max_cumulative_drift_px=5.0,
    )

    ox, oy = accumulated[1]
    assert abs(np.sqrt(ox**2 + oy**2) - 5.0) < 1e-9


def test_accumulate_pairwise_translations_skips_id_step_gap():
    """A leftover z49→z51 translation must not shift z51 on the canvas."""
    available_ids = [49, 51]
    all_pairwise_translations = {51: (200.0, 0.0, 0.9)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={51: None},
        all_pairwise_translations=all_pairwise_translations,
    )

    assert accumulated[51] == (0.0, 0.0)


def test_accumulate_pairwise_translations_keeps_manual_gap():
    """z51's manual delta is added on top of z49, not discarded."""
    available_ids = [49, 51]
    all_pairwise_translations = {51: (-74.0, -64.0, 0.0)}

    accumulated = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={51: None},
        all_pairwise_translations=all_pairwise_translations,
        translation_min_zcorr=0.0,
        keep_gap_slice_ids={51},
    )

    assert accumulated[51] == (-74.0, -64.0)


def test_accumulate_pairwise_translations_smooths_slice_jitter():
    available_ids = list(range(9))
    all_pairwise_translations = {i: (40.0 if i % 2 == 0 else -40.0, 0.0, 0.9) for i in available_ids[1:]}

    raw = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        translation_min_zcorr=0.0,
    )
    smoothed = accumulate_pairwise_translations(
        available_ids,
        registration_transforms={},
        all_pairwise_translations=all_pairwise_translations,
        translation_min_zcorr=0.0,
        translation_smooth_sigma=3.0,
    )

    def max_step(acc: dict) -> float:
        ids = sorted(acc)
        return max(abs(acc[b][0] - acc[a][0]) for a, b in pairwise(ids))

    assert abs(max_step(raw) - 40.0) < 1e-9
    assert max_step(smoothed) < 15.0


def test_common_space_xy_policy_accumulates_on_common_space():
    accumulate, apply_tfm, rotation_only = common_space_xy_policy(
        no_xy_shift=True, accumulate_translations=True, rotation_only=False
    )
    assert accumulate is True
    assert apply_tfm is True
    assert rotation_only is True


def test_common_space_xy_policy_accumulates_without_common_space():
    accumulate, apply_tfm, rotation_only = common_space_xy_policy(
        no_xy_shift=False, accumulate_translations=True, rotation_only=False
    )
    assert accumulate is True
    assert apply_tfm is True
    assert rotation_only is True
