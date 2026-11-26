"""Tests for linumpy/imaging/transform.py"""

import numpy as np
import pytest
import SimpleITK as sitk

from linumpy.imaging.transform import apply_xy_shift


def _shift_with_reference_image(img: np.ndarray, reference: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The shift as resampled onto an image made from the reference array."""
    fixed = sitk.GetImageFromArray(reference)
    translation = [0.0] * fixed.GetDimension()
    translation[0], translation[1] = dx, dy
    transform = sitk.TranslationTransform(fixed.GetDimension())
    transform.SetParameters(translation)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    nonzero = img[img > 0]
    resampler.SetDefaultPixelValue(float(np.percentile(nonzero, 1)) if len(nonzero) else 0.0)
    resampler.SetTransform(transform)
    return sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(img)))


@pytest.mark.parametrize(("shape", "ref_shape"), [((4, 20, 24), (4, 30, 28)), ((20, 24), (26, 30))])
@pytest.mark.parametrize("n_threads", [None, 2])
def test_apply_xy_shift_uses_only_the_shape_of_the_reference(shape, ref_shape, n_threads):
    img = np.random.default_rng(6).random(shape, dtype=np.float32) + 0.1
    reference = np.zeros(ref_shape, dtype=np.float32)
    shifted = apply_xy_shift(img, reference, -3.5, -2.25, n_threads=n_threads)
    expected = _shift_with_reference_image(img, reference, -3.5, -2.25)
    assert shifted.shape == ref_shape
    assert shifted.dtype == img.dtype
    assert np.array_equal(shifted, expected)


def test_apply_xy_shift_removes_a_slant():
    # A line that drifts 0.25 px in x and -0.5 px in y per plane of depth.
    slant = (0.25, -0.5)
    depth, height, width = 9, 40, 44
    zz, yy, xx = np.meshgrid(np.arange(depth), np.arange(height), np.arange(width), indexing="ij")
    x0, y0 = 20.0, 22.0
    img = (0.1 + np.exp(-((xx - x0 - slant[0] * zz) ** 2 + (yy - y0 - slant[1] * zz) ** 2) / 8.0)).astype(np.float32)
    reference = np.zeros((depth, height + 6, width + 6), dtype=np.float32)

    # The content is put 3 px further in x and 2 px further in y, as the common space does.
    straight = apply_xy_shift(img, reference, -3.0, -2.0, slant=slant)

    assert straight.shape == reference.shape
    for z in range(depth):
        row, col = np.unravel_index(int(straight[z].argmax()), straight[z].shape)
        assert (row, col) == (int(y0) + 2, int(x0) + 3)
    # Plane 0 is the reference of the slant: it equals the plain shift.
    assert np.array_equal(straight[0], apply_xy_shift(img, reference, -3.0, -2.0)[0])


def test_apply_xy_shift_without_slant_is_unchanged_and_a_slant_needs_a_volume():
    img = np.random.default_rng(3).random((5, 20, 24), dtype=np.float32) + 0.1
    reference = np.zeros((5, 26, 30), dtype=np.float32)
    assert np.array_equal(
        apply_xy_shift(img, reference, -1.5, 2.0, slant=(0.0, 0.0)), apply_xy_shift(img, reference, -1.5, 2.0)
    )
    with pytest.raises(ValueError, match="volume"):
        apply_xy_shift(img[0], reference[0], -1.5, 2.0, slant=(0.1, 0.0))
