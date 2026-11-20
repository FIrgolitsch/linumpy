"""Image transform helpers (normalization, padding, XY shifts)."""

import numpy as np
import SimpleITK as sitk


def normalize(img: np.ndarray, saturation: float = 99.7) -> np.ndarray:
    """Normalize an image between 0 and 1.

    Parameters
    ----------
    img : np.ndarray
        The image to normalize.
    saturation : float, optional
        The saturation value for the normalization

    Returns
    -------
    np.ndarray
        The normalized image.
    """
    imin = img.min()
    imax = np.percentile(img, saturation)
    img = (img.astype(np.float32) - imin) / (imax - imin)
    img[img > 1] = 1
    return img


def match_shape(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the shape of two images by padding the smallest one.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The two images with the same shape.
    """
    nr1, nc1 = img1.shape
    nr2, nc2 = img2.shape
    n_rows = max(nr1, nr2)
    n_cols = max(nc1, nc2)

    padded_images = []
    for img in [img1, img2]:
        pad_r_0 = max((n_rows - img.shape[0]) // 2, 0)
        pad_r_1 = max((n_rows - img.shape[0] - pad_r_0), 0)
        pad_c_0 = max((n_cols - img.shape[1]) // 2, 0)
        pad_c_1 = max((n_cols - img.shape[1] - pad_c_0), 0)
        padded_images.append(np.pad(img, ((pad_r_0, pad_r_1), (pad_c_0, pad_c_1))))

    return padded_images[0], padded_images[1]


def apply_xy_shift(img: np.ndarray, reference: np.ndarray, dx: float, dy: float, n_threads: int | None = None) -> np.ndarray:
    """Apply a shift to the image in the xy plane.

    Parameters
    ----------
    img : np.ndarray
        The image to shift.
    reference : np.ndarray
        The reference image. Only its shape is used.
    dx : float
        The shift in x.
    dy : float
        The shift in y.
    n_threads : int or None
        Threads the resampling may use. ``None`` keeps the SimpleITK default.
    """
    moving = sitk.GetImageFromArray(img)

    # Only the shape of the reference is used: the output grid has unit
    # spacing and its origin at zero, as an image made from the array would.
    # Copying the reference voxels into an image for that cost as much as
    # the resampling.
    shape = np.shape(reference)
    ndim = len(shape)
    translation = [0.0] * ndim
    translation[0] = dx
    translation[1] = dy

    transform = sitk.TranslationTransform(ndim)
    transform.SetParameters(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(n) for n in reversed(shape)])
    resampler.SetOutputSpacing([1.0] * ndim)
    resampler.SetOutputOrigin([0.0] * ndim)
    resampler.SetOutputDirection(np.eye(ndim).ravel().tolist())
    if n_threads is not None:
        resampler.SetNumberOfThreads(max(1, int(n_threads)))
    resampler.SetInterpolator(sitk.sitkLinear)

    # Use a small positive value instead of zero to avoid black dots at boundaries
    nonzero_vals = img[img > 0]
    default_val = float(np.percentile(nonzero_vals, 1)) if len(nonzero_vals) > 0 else 0.0
    resampler.SetDefaultPixelValue(default_val)
    resampler.SetTransform(transform)
    warped_moving_image = resampler.Execute(moving)
    img_warped = sitk.GetArrayFromImage(warped_moving_image)
    return img_warped
