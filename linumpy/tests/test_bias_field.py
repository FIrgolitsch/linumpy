"""Tests for linumpy/intensity/bias_field.py (and gpu/bias_field.py)."""

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.intensity.bias_field import (
    apply_bias_field,
    compute_tissue_mask,
    n4_correct,
    n4_correct_per_section,
    tissue_mask_silhouette_xy,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_phantom(
    shape: tuple[int, int, int] = (20, 32, 32),
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uniform tissue phantom, known multiplicative bias field).

    The bias field is a smooth gradient (1.0 at the top, 2.0 at the bottom),
    which mimics axial attenuation in OCT data.
    """
    rng = np.random.default_rng(rng_seed)
    nz, _ny, _nx = shape

    # Flat tissue signal + small noise
    tissue = np.ones(shape, dtype=np.float32) * 0.5 + rng.random(shape).astype(np.float32) * 0.05

    # Bias: exponential gradient along Z (depth-dependent attenuation)
    z_coords = np.linspace(1.0, 2.0, nz, dtype=np.float32)
    bias_field = z_coords[:, np.newaxis, np.newaxis] * np.ones(shape, dtype=np.float32)

    corrupted = tissue * bias_field
    return corrupted, bias_field


# ---------------------------------------------------------------------------
# compute_tissue_mask
# ---------------------------------------------------------------------------


def test_compute_tissue_mask_shape():
    vol, _ = _make_phantom((10, 24, 24))
    mask = compute_tissue_mask(vol)
    assert mask.shape == vol.shape


def test_compute_tissue_mask_is_boolean():
    vol, _ = _make_phantom()
    mask = compute_tissue_mask(vol)
    assert mask.dtype == bool


def test_compute_tissue_mask_nonempty_volume():
    """A clearly structured volume should produce a non-trivial mask."""
    rng = np.random.default_rng(1)
    vol = rng.random((10, 24, 24)).astype(np.float32) * 0.1  # agarose
    vol[:, 8:16, 8:16] += 0.6  # tissue block
    mask = compute_tissue_mask(vol, smoothing_sigma=1.0)
    assert mask.any() and not mask.all()


def test_compute_tissue_mask_per_section_differs():
    """Per-section masking captures tissue location varying along Z."""
    rng = np.random.default_rng(2)
    vol = rng.random((20, 24, 24)).astype(np.float32) * 0.1  # agarose
    # First section: tissue on the left; second section: tissue on the right.
    vol[:10, 8:16, 4:12] += 0.6
    vol[10:, 8:16, 12:20] += 0.6
    # Disable Z-closing so section masks remain independent.
    mask = compute_tissue_mask(vol, smoothing_sigma=1.0, n_serial_slices=2, z_closing_sections=0)
    assert not np.array_equal(mask[0], mask[-1])


def test_compute_tissue_mask_oblique_section():
    """Oblique tissue: mask shape must follow Z (top != bottom of a section)."""
    rng = np.random.default_rng(3)
    vol = rng.random((20, 32, 32)).astype(np.float32) * 0.1  # agarose
    # Tissue block translates linearly across Z (45° slant in X).
    for z in range(20):
        x_start = 4 + z  # shifts by 1 px per Z
        vol[z, 10:22, x_start : x_start + 8] += 0.6
    mask = compute_tissue_mask(vol, smoothing_sigma=1.0, n_serial_slices=1, z_closing_sections=0)
    # Mask centroid in X must shift between top and bottom of the volume.
    top_xs = np.argwhere(mask[0])[:, 1]
    bot_xs = np.argwhere(mask[-1])[:, 1]
    assert top_xs.size > 0 and bot_xs.size > 0
    assert bot_xs.mean() > top_xs.mean() + 5  # large oblique displacement


def test_tissue_mask_silhouette_keeps_overlap_drops_agarose():
    """OR-Z silhouette fills empty overlap planes inside the tissue footprint."""
    mask = np.zeros((8, 24, 24), dtype=bool)
    mask[:3, 6:18, 6:18] = True
    sil = tissue_mask_silhouette_xy(mask)
    assert sil.shape == (24, 24)
    assert sil[6:18, 6:18].all()
    assert not sil[0, 0]
    extruded = np.broadcast_to(sil[None], mask.shape)
    assert extruded[5, 12, 12]  # overlap plane inside XY footprint
    assert not extruded[5, 0, 0]  # agarose


def test_tissue_mask_silhouette_fills_xy_holes():
    mask = np.zeros((4, 20, 20), dtype=bool)
    mask[:, 4:16, 4:16] = True
    mask[:, 8:12, 8:12] = False
    sil = tissue_mask_silhouette_xy(mask)
    assert sil[10, 10]


def test_tissue_mask_silhouette_dilate_expands():
    mask = np.zeros((4, 20, 20), dtype=bool)
    mask[:, 8:12, 8:12] = True
    sil0 = tissue_mask_silhouette_xy(mask, dilate_px=0)
    sil2 = tissue_mask_silhouette_xy(mask, dilate_px=2)
    assert sil2.sum() > sil0.sum()
    assert sil2[6, 10] and not sil0[6, 10]


def test_tissue_mask_silhouette_rejects_bad_rank():
    with pytest.raises(ValueError, match="3-D"):
        tissue_mask_silhouette_xy(np.ones((8, 8), dtype=bool))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_tissue_mask_silhouette_gpu_mask_multiplies_cupy_volume():
    """Silhouette must live on device so CuPy in-place multiply does not crash."""
    import cupy as cp

    from linumpy.gpu import is_cupy_array

    mask = cp.zeros((4, 16, 16), dtype=bool)
    mask[:, 4:12, 4:12] = True
    sil = tissue_mask_silhouette_xy(mask)
    assert is_cupy_array(sil)
    vol = cp.ones((4, 16, 16), dtype=cp.float32)
    vol *= sil
    assert float(vol[:, 0, 0].max()) == 0.0
    assert float(vol[:, 8, 8].min()) == 1.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_bias_save_writes_cupy_volume(tmp_path):
    """zarr cannot ingest CuPy arrays; _save must host-copy first."""
    import argparse

    import cupy as cp

    from scripts.illumination.linum_correct_bias_field import _save

    arr = cp.ones((4, 8, 8), dtype=cp.float32)
    args = argparse.Namespace(n_levels=0, pyramid_resolutions=None, make_isotropic=False)
    _save(arr, str(tmp_path / "out.ome.zarr"), [0.01, 0.01, 0.01], args)


# ---------------------------------------------------------------------------
# n4_correct
# ---------------------------------------------------------------------------


def test_n4_correct_output_shape():
    vol, _ = _make_phantom((10, 20, 20))
    corrected, bias = n4_correct(vol, shrink_factor=2, n_iterations=[10, 10])
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape


def test_n4_correct_bias_field_positive():
    vol, _ = _make_phantom((10, 20, 20))
    _, bias = n4_correct(vol, shrink_factor=2, n_iterations=[10, 10])
    assert float(bias.min()) > 0


def test_n4_correct_reduces_gradient():
    """After correction the axial mean gradient should be smaller."""
    vol, _ = _make_phantom((16, 20, 20))

    # Measure gradient before: mean per Z-plane
    means_before = vol.mean(axis=(1, 2))
    gradient_before = float(means_before[-1] - means_before[0])

    corrected, _ = n4_correct(vol, shrink_factor=2, n_iterations=[20, 20])

    means_after = corrected.mean(axis=(1, 2))
    gradient_after = float(means_after[-1] - means_after[0])

    # The N4-corrected gradient should be smaller in absolute terms
    assert abs(gradient_after) < abs(gradient_before), (
        f"Expected N4 to reduce axial gradient; before={gradient_before:.3f}, after={gradient_after:.3f}"
    )


# ---------------------------------------------------------------------------
# apply_bias_field
# ---------------------------------------------------------------------------


def test_apply_bias_field_inverse():
    """Dividing by the known bias field should recover the original signal."""
    vol, bias = _make_phantom((10, 20, 20))
    # vol = tissue * bias → tissue = vol / bias
    recovered = apply_bias_field(vol, bias)
    residual_std = float(np.std(recovered - (vol / bias)))
    assert residual_std < 1e-5


def test_apply_bias_field_floor():
    """Near-zero bias values must not produce Inf/NaN."""
    vol = np.ones((4, 8, 8), dtype=np.float32)
    bias = np.zeros((4, 8, 8), dtype=np.float32)  # all zeros
    result = apply_bias_field(vol, bias)
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# n4_correct_per_section
# ---------------------------------------------------------------------------


def _make_per_section_phantom(n_sections: int = 4, z_per_section: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Phantom with a different bias gradient per section (piecewise)."""
    rng = np.random.default_rng(7)
    ny, nx = 20, 20
    chunks = []
    biases = []
    for i in range(n_sections):
        # Each section has its own scale (models per-section laser drift)
        scale = 1.0 + 0.5 * i
        flat = rng.random((z_per_section, ny, nx)).astype(np.float32) * 0.02
        tissue = np.ones((z_per_section, ny, nx), dtype=np.float32) * 0.5 + flat
        z_coords = np.linspace(scale, scale * 1.5, z_per_section, dtype=np.float32)
        bias = z_coords[:, np.newaxis, np.newaxis] * np.ones((z_per_section, ny, nx), dtype=np.float32)
        chunks.append(tissue * bias)
        biases.append(bias)
    return np.concatenate(chunks, axis=0), np.concatenate(biases, axis=0)


def test_n4_correct_per_section_output_shape():
    vol, _ = _make_per_section_phantom(n_sections=2, z_per_section=5)
    corrected, bias = n4_correct_per_section(vol, n_serial_slices=2, n_processes=1, shrink_factor=2, n_iterations=[10, 10])
    assert corrected.shape == vol.shape
    assert bias.shape == vol.shape


def test_n4_correct_per_section_serial_equals_parallel():
    """n_processes=1 and n_processes=2 must produce identical results."""
    vol, _ = _make_per_section_phantom(n_sections=2, z_per_section=5)

    corrected_1, _ = n4_correct_per_section(vol, n_serial_slices=2, n_processes=1, shrink_factor=2, n_iterations=[10, 10])
    corrected_2, _ = n4_correct_per_section(vol, n_serial_slices=2, n_processes=2, shrink_factor=2, n_iterations=[10, 10])

    np.testing.assert_allclose(corrected_1, corrected_2, atol=1e-5, rtol=0)

    np.testing.assert_allclose(corrected_1, corrected_2, atol=1e-5, rtol=0)


def test_n4_correct_per_section_reduces_section_gradient():
    """Per-section correction should flatten intra-section axial gradients."""
    n_sections, z_per = 2, 8
    vol, _ = _make_per_section_phantom(n_sections=n_sections, z_per_section=z_per)

    corrected, _ = n4_correct_per_section(
        vol, n_serial_slices=n_sections, n_processes=1, shrink_factor=2, n_iterations=[20, 20]
    )

    nz = vol.shape[0]
    for s in range(n_sections):
        z_start = s * z_per
        z_end = min(z_start + z_per, nz)
        grad_before = abs(float(vol[z_end - 1].mean()) - float(vol[z_start].mean()))
        grad_after = abs(float(corrected[z_end - 1].mean()) - float(corrected[z_start].mean()))
        assert grad_after < grad_before, (
            f"Section {s}: expected reduced gradient; before={grad_before:.3f}, after={grad_after:.3f}"
        )


# ---------------------------------------------------------------------------
# GPU helpers (skipped when GPU_AVAILABLE is False)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_apply_bias_field_matches_cpu():
    """GPU and CPU apply_bias_field must agree to within 1e-4 max abs diff."""
    from linumpy.gpu.bias_field import apply_bias_field_gpu

    vol, bias = _make_phantom((10, 20, 20))
    cpu_result = apply_bias_field(vol, bias)
    gpu_result = apply_bias_field_gpu(vol, bias, use_gpu=True)
    assert np.max(np.abs(gpu_result - cpu_result)) < 1e-4


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_downsample_shape():
    from linumpy.gpu.bias_field import downsample_gpu

    vol = np.ones((20, 32, 32), dtype=np.float32)
    shrunk = downsample_gpu(vol, shrink_factor=4, use_gpu=True)
    assert shrunk.shape == (5, 8, 8)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_upsample_shape():
    from linumpy.gpu.bias_field import upsample_bias_gpu

    bias_low = np.ones((5, 8, 8), dtype=np.float32)
    upsampled = upsample_bias_gpu(bias_low, target_shape=(20, 32, 32), use_gpu=True)
    assert upsampled.shape == (20, 32, 32)
