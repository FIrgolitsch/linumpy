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
