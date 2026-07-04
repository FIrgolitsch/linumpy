"""Tests for linumpy/registration/sitk.py — SimpleITK registration wrappers.

Uses small synthetic SimpleITK images built from numpy arrays (constant and
gradient volumes) so the tests run without CUDA or real OCT data. Covers
``itk_registration``, ``align_images_sitk``, ``register_2d_images_sitk``,
``apply_transform``, ``_numpy_to_sitk_image``, ``_crop_to_bbox``,
``_tissue_centroid_mm``, and ``command_iteration``.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from linumpy.registration.sitk import (
    _crop_to_bbox,
    _numpy_to_sitk_image,
    _tissue_centroid_mm,
    align_images_sitk,
    apply_transform,
    command_iteration,
    itk_registration,
    register_2d_images_sitk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gradient_2d(shape=(16, 16)) -> np.ndarray:
    """A smooth 2D gradient image — easy for SimpleITK to register."""
    rows, cols = shape
    r = np.linspace(0, 100, num=rows, dtype=np.float32)
    c = np.linspace(0, 50, num=cols, dtype=np.float32)
    return r[:, None] + c[None, :]


def _make_gradient_3d(shape=(8, 16, 16)) -> np.ndarray:
    """A smooth 3D gradient volume (Z, Y, X)."""
    nz, ny, nx = shape
    z = np.linspace(0, 80, num=nz, dtype=np.float32)
    y = np.linspace(0, 40, num=ny, dtype=np.float32)
    x = np.linspace(0, 20, num=nx, dtype=np.float32)
    return z[:, None, None] + y[None, :, None] + x[None, None, :]


def _make_shifted(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Roll a 2D array by (dy, dx) to simulate a known translation."""
    return np.roll(np.roll(arr, shift=dy, axis=0), shift=dx, axis=1)


# ---------------------------------------------------------------------------
# _numpy_to_sitk_image
# ---------------------------------------------------------------------------


def test_numpy_to_sitk_image_sets_spacing_zyx_to_xyz():
    """Spacing is given in numpy (sz, sy, sx) order; SITK stores (sx, sy, sz)."""
    vol = np.ones((4, 8, 16), dtype=np.float32)
    spacing = (2.0, 4.0, 8.0)  # (sz, sy, sx)
    img = _numpy_to_sitk_image(vol, spacing)
    sitk_spacing = img.GetSpacing()  # returns (sx, sy, sz)
    assert sitk_spacing == (8.0, 4.0, 2.0)


def test_numpy_to_sitk_image_origin_is_zero():
    vol = np.ones((4, 4, 4), dtype=np.float32)
    img = _numpy_to_sitk_image(vol, (1.0, 1.0, 1.0))
    assert img.GetOrigin() == (0.0, 0.0, 0.0)


def test_numpy_to_sitk_image_direction_is_identity():
    vol = np.ones((4, 4, 4), dtype=np.float32)
    img = _numpy_to_sitk_image(vol, (1.0, 1.0, 1.0))
    direction = img.GetDirection()
    np.testing.assert_allclose(list(direction), [1, 0, 0, 0, 1, 0, 0, 0, 1])


def test_numpy_to_sitk_image_preserves_shape():
    vol = np.ones((3, 5, 7), dtype=np.float32)
    img = _numpy_to_sitk_image(vol, (1.0, 1.0, 1.0))
    # SITK GetSize returns (sx, sy, sz) = (7, 5, 3).
    assert img.GetSize() == (7, 5, 3)


# ---------------------------------------------------------------------------
# itk_registration (3D and 2D)
# ---------------------------------------------------------------------------


def test_itk_registration_3d_returns_three_deltas():
    """3D registration returns a 3-element delta list and a finite metric."""
    fixed = _make_gradient_3d((8, 16, 16))
    moving = np.roll(fixed, shift=1, axis=0)  # small Z shift
    deltas, metric = itk_registration(fixed, moving, metric="MSQ")
    assert len(deltas) == 3
    assert all(isinstance(d, float) for d in deltas)
    assert np.isfinite(metric)


def test_itk_registration_2d_returns_two_deltas():
    """2D registration returns a 2-element delta list."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=2)
    deltas, metric = itk_registration(fixed, moving, metric="MSQ")
    assert len(deltas) == 2
    assert np.isfinite(metric)


def test_itk_registration_identical_images_near_zero_delta():
    """Identical fixed/moving images → deltas near zero."""
    fixed = _make_gradient_3d((8, 16, 16))
    deltas, _metric = itk_registration(fixed, fixed, metric="MSQ")
    assert all(abs(d) < 1.0 for d in deltas)


def test_itk_registration_metric_corr_runs():
    """The 'corr' metric path executes without error."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=1, dx=0)
    deltas, metric = itk_registration(fixed, moving, metric="corr")
    assert len(deltas) == 2
    assert np.isfinite(metric)


def test_itk_registration_metric_jhmi_runs():
    """The 'JHMI' metric path executes."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    deltas, _metric = itk_registration(fixed, moving, metric="JHMI")
    assert len(deltas) == 2


def test_itk_registration_metric_mmi_runs():
    """The 'MMI' metric path executes."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    deltas, _metric = itk_registration(fixed, moving, metric="MMI")
    assert len(deltas) == 2


def test_itk_registration_metric_antscorr_runs():
    """The 'ANTsCorr' metric path executes."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    deltas, _metric = itk_registration(fixed, moving, metric="ANTsCorr")
    assert len(deltas) == 2


def test_itk_registration_unknown_metric_falls_back_to_corr():
    """An unknown metric string falls through to the correlation metric."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    deltas, _metric = itk_registration(fixed, moving, metric="bogus")
    assert len(deltas) == 2  # did not raise; fell through to corr


def test_itk_registration_match_histograms_runs():
    """The match_histograms=True path executes without error."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1) * 2.0 + 5.0
    deltas, _metric = itk_registration(fixed, moving, metric="MSQ", match_histograms=True)
    assert len(deltas) == 2


def test_itk_registration_with_masks_runs():
    """Fixed and moving masks are accepted and the registration runs."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    mask = np.ones_like(fixed, dtype=np.float32)
    mask[:8, :] = 0  # mask out top half
    deltas, _metric = itk_registration(fixed, moving, metric="MSQ", mask_fixed=mask, mask_moving=mask)
    assert len(deltas) == 2


# ---------------------------------------------------------------------------
# align_images_sitk
# ---------------------------------------------------------------------------


def test_align_images_sitk_returns_two_deltas():
    """align_images_sitk returns a 2-element delta list and a finite metric."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=2)
    deltas, metric = align_images_sitk(fixed, moving)
    assert len(deltas) == 2
    assert all(isinstance(d, float) for d in deltas)
    assert np.isfinite(metric)


def test_align_images_sitk_identical_images_near_zero():
    """Identical images → deltas near zero."""
    fixed = _make_gradient_2d((16, 16))
    deltas, _metric = align_images_sitk(fixed, fixed)
    assert all(abs(d) < 1.0 for d in deltas)


# ---------------------------------------------------------------------------
# register_2d_images_sitk
# ---------------------------------------------------------------------------


def test_register_2d_images_sitk_returns_transform_and_stop_condition():
    """Returns (Transform, stop_condition_str, metric_float).

    Uses method='euler' (default) — the 'translation' method without
    initial_translation hits a SimpleITK CenteredTransformInitializer
    limitation (TranslationTransform has no center).
    """
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, stop, metric = register_2d_images_sitk(fixed, moving, method="euler", max_iterations=50)
    assert isinstance(transform, sitk.Transform)
    assert isinstance(stop, str)
    assert isinstance(metric, float)


def test_register_2d_images_sitk_euler_method():
    """The 'euler' method produces an Euler2DTransform by default."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(fixed, moving, method="euler", max_iterations=50)
    # Default (return_3d_transform=False) → 2D transform.
    assert transform.GetDimension() == 2


def test_register_2d_images_sitk_affine_method():
    """The 'affine' method runs without error."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(fixed, moving, method="affine", max_iterations=50)
    assert isinstance(transform, sitk.Transform)


def test_register_2d_images_sitk_translation_method_with_initial_translation():
    """The 'translation' method runs when initial_translation is provided.

    Without initial_translation, the CenteredTransformInitializer path fails
    for TranslationTransform (no center) — a pre-existing SimpleITK
    limitation. Providing initial_translation uses SetOffset instead.
    """
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(
        fixed, moving, method="translation", max_iterations=50, initial_translation=(0.0, 1.0)
    )
    assert isinstance(transform, sitk.Transform)


def test_register_2d_images_sitk_unknown_method_raises():
    """An unknown method string raises ValueError."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    with pytest.raises(ValueError, match="Unknown method"):
        register_2d_images_sitk(fixed, moving, method="bogus", max_iterations=10)


def test_register_2d_images_sitk_unknown_metric_raises():
    """An unknown metric string raises ValueError."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    with pytest.raises(ValueError, match="Unknown metric"):
        register_2d_images_sitk(fixed, moving, method="translation", metric="bogus", max_iterations=10)


def test_register_2d_images_sitk_metric_mse():
    """The 'MSE' metric path runs."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(fixed, moving, metric="MSE", max_iterations=50)


def test_register_2d_images_sitk_metric_mi():
    """The 'MI' metric path runs."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(fixed, moving, metric="MI", max_iterations=50)


def test_register_2d_images_sitk_metric_antscc():
    """The 'AntsCC' metric path runs."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(fixed, moving, metric="AntsCC", max_iterations=50)


def test_register_2d_images_sitk_metric_cc():
    """The 'CC' metric path runs."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(fixed, moving, metric="CC", max_iterations=50)


def test_register_2d_images_sitk_with_masks():
    """Fixed and moving masks are accepted."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    mask = np.ones_like(fixed, dtype=np.uint8)
    mask[:8, :] = 0
    _transform, _stop, _metric = register_2d_images_sitk(
        fixed, moving, method="euler", max_iterations=50, fixed_mask=mask, moving_mask=mask
    )


def test_register_2d_images_sitk_return_3d_transform_euler():
    """return_3d_transform=True with euler method returns a 3D transform."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(
        fixed, moving, method="euler", max_iterations=50, return_3d_transform=True
    )
    assert transform.GetDimension() == 3


def test_register_2d_images_sitk_return_3d_transform_translation():
    """return_3d_transform=True with translation method returns a 3D transform.

    Uses initial_translation to avoid the CenteredTransformInitializer
    limitation for TranslationTransform.
    """
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(
        fixed,
        moving,
        method="translation",
        max_iterations=50,
        return_3d_transform=True,
        initial_translation=(0.0, 1.0),
    )
    assert transform.GetDimension() == 3


def test_register_2d_images_sitk_return_3d_transform_affine():
    """return_3d_transform=True with affine method returns a 3D transform."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    transform, _stop, _metric = register_2d_images_sitk(
        fixed, moving, method="affine", max_iterations=50, return_3d_transform=True
    )
    assert transform.GetDimension() == 3


def test_register_2d_images_sitk_initial_translation():
    """Providing initial_translation runs the centered-init path."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(
        fixed, moving, method="euler", max_iterations=50, initial_translation=(1.0, 0.0)
    )


def test_register_2d_images_sitk_initial_step():
    """Providing initial_step overrides the default step size."""
    fixed = _make_gradient_2d((16, 16))
    moving = _make_shifted(fixed, dy=0, dx=1)
    _transform, _stop, _metric = register_2d_images_sitk(fixed, moving, method="euler", max_iterations=50, initial_step=2.0)


def test_register_2d_images_sitk_identical_images():
    """Identical fixed/moving images → near-zero translation."""
    fixed = _make_gradient_2d((16, 16))
    transform, _stop, _metric = register_2d_images_sitk(fixed, fixed, method="euler", max_iterations=50)
    params = transform.GetParameters()
    # Euler2DTransform params: (angle, tx, ty). For identical images all near 0.
    assert all(abs(p) < 2.0 for p in params)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


def test_apply_transform_identity_preserves_image():
    """Applying an identity translation transform returns the same image."""
    moving = _make_gradient_2d((16, 16))
    identity = sitk.TranslationTransform(2, (0.0, 0.0))
    result = apply_transform(moving, identity)
    assert result.shape == moving.shape
    np.testing.assert_allclose(result, moving, atol=1e-3)


def test_apply_transform_3d_identity_preserves_volume():
    """3D identity transform preserves the volume."""
    moving = _make_gradient_3d((8, 16, 16))
    identity = sitk.TranslationTransform(3, (0.0, 0.0, 0.0))
    result = apply_transform(moving, identity)
    assert result.shape == moving.shape
    np.testing.assert_allclose(result, moving, atol=1e-3)


def test_apply_transform_with_reference_image_requires_spacing():
    """reference_image without spacing raises ValueError."""
    moving = _make_gradient_2d((16, 16))
    identity = sitk.TranslationTransform(2, (0.0, 0.0))
    with pytest.raises(ValueError, match="reference_spacing and moving_spacing are required"):
        apply_transform(moving, identity, reference_image=moving)


def test_apply_transform_with_reference_and_spacing_runs():
    """apply_transform with reference_image + spacing runs without error.

    Note: _numpy_to_sitk_image expects 3D spacing (sz, sy, sx), so this test
    uses 3D volumes.
    """
    moving = _make_gradient_3d((8, 16, 16))
    reference = _make_gradient_3d((8, 16, 16))
    identity = sitk.TranslationTransform(3, (0.0, 0.0, 0.0))
    result = apply_transform(
        moving,
        identity,
        reference_image=reference,
        reference_spacing=(1.0, 1.0, 1.0),
        moving_spacing=(1.0, 1.0, 1.0),
    )
    assert result.shape == reference.shape


# ---------------------------------------------------------------------------
# _crop_to_bbox
# ---------------------------------------------------------------------------


def test_crop_to_bbox_all_zero_returns_original():
    """An all-zero volume is returned unchanged with zero offset."""
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    cropped, offset = _crop_to_bbox(vol, spacing=(1.0, 1.0, 1.0), margin_voxels=2)
    assert cropped.shape == vol.shape
    assert offset == (0.0, 0.0, 0.0)


def test_crop_to_bbox_crops_to_nonzero_region():
    """A volume with a small non-zero region is cropped to that region + margin."""
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[8:12, 8:12, 8:12] = 1.0  # 4x4x4 block in the center
    cropped, offset = _crop_to_bbox(vol, spacing=(2.0, 4.0, 8.0), margin_voxels=2)
    # Nonzero indices are 8..11. Bounding box: [6, 14) in each axis
    # (max(0, 8-2)=6, min(20, 11+2+1)=14) → size 8.
    assert cropped.shape == (8, 8, 8)
    # Offset in mm: (6*2, 6*4, 6*8) = (12, 24, 48).
    assert offset == (12.0, 24.0, 48.0)


def test_crop_to_bbox_margin_zero():
    """margin_voxels=0 crops to the exact nonzero bounding box."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[3:7, 3:7, 3:7] = 1.0
    cropped, offset = _crop_to_bbox(vol, spacing=(1.0, 1.0, 1.0), margin_voxels=0)
    assert cropped.shape == (4, 4, 4)
    assert offset == (3.0, 3.0, 3.0)


# ---------------------------------------------------------------------------
# _tissue_centroid_mm
# ---------------------------------------------------------------------------


def test_tissue_centroid_mm_all_zero_falls_back_to_geometric_center():
    """An all-zero volume falls back to the geometric center."""
    vol = np.zeros((8, 16, 32), dtype=np.float32)
    centroid = _tissue_centroid_mm(vol, spacing=(2.0, 4.0, 8.0))
    # Geometric center: (4*2, 8*4, 16*8) = (8, 32, 128).
    assert centroid == (8.0, 32.0, 128.0)


def test_tissue_centroid_mm_single_voxel():
    """A single non-zero voxel has its centroid at that voxel (in mm)."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[4, 5, 6] = 1.0
    centroid = _tissue_centroid_mm(vol, spacing=(1.0, 1.0, 1.0))
    assert centroid == (4.0, 5.0, 6.0)


def test_tissue_centroid_mm_uses_spacing():
    """Centroid coordinates are scaled by voxel spacing."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2, 2, 2] = 1.0
    centroid = _tissue_centroid_mm(vol, spacing=(2.0, 4.0, 8.0))
    assert centroid == (4.0, 8.0, 16.0)


# ---------------------------------------------------------------------------
# command_iteration
# ---------------------------------------------------------------------------


def test_command_iteration_prints(capsys):
    """command_iteration prints iteration info without raising."""

    # Build a minimal fake registration method object.
    class FakeReg:
        def GetOptimizerIteration(self):
            return 3

        def GetMetricValue(self):
            return 0.12345

        def GetOptimizerPosition(self):
            return [1.0, 2.0]

        def GetOptimizerScales(self):
            return [1.0, 1.0]

    command_iteration(FakeReg())  # ty: ignore[invalid-argument-type]
    out = capsys.readouterr().out
    assert "3" in out
