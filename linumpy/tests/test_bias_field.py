"""Tests for linumpy/intensity/bias_field.py (and gpu/bias_field.py)."""

from itertools import pairwise

import numpy as np
import pytest

from linumpy.gpu import GPU_AVAILABLE
from linumpy.intensity.bias_field import (
    apply_bias_field,
    compute_tissue_mask,
    drop_small_mask_components,
    drop_thin_mask_parts,
    n4_correct,
    n4_correct_per_section,
    tissue_mask_between_cut_faces,
    tissue_mask_section_silhouettes,
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


def test_compute_tissue_mask_threshold_scale_keeps_dim_tissue():
    """Zeroed background leaves Otsu between dim and bright tissue; the scale recovers the dim rim."""
    rng = np.random.default_rng(0)
    vol = np.zeros((6, 64, 64), dtype=np.float32)
    vol[:, 8:56, 8:56] = 0.15  # dim tissue
    vol[:, 20:44, 20:44] = 0.6  # bright core
    vol[vol > 0] += rng.normal(0, 0.01, int((vol > 0).sum())).astype(np.float32)

    otsu_mask = compute_tissue_mask(vol, smoothing_sigma=1.0, closing_radius=0, z_closing_sections=0)
    scaled_mask = compute_tissue_mask(vol, smoothing_sigma=1.0, closing_radius=0, z_closing_sections=0, threshold_scale=0.3)

    assert not otsu_mask[3, 12, 12]
    assert scaled_mask[3, 12, 12]
    assert scaled_mask[3, 32, 32]
    assert not scaled_mask[3, 2, 2]


def test_compute_tissue_mask_rejects_non_positive_threshold_scale():
    with pytest.raises(ValueError, match="threshold_scale"):
        compute_tissue_mask(np.ones((2, 8, 8), dtype=np.float32), threshold_scale=0.0)


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


def test_section_silhouettes_drop_other_slice_agarose_fov():
    """Agarose in slice 0's unique FOV must not survive on slice 1."""
    mask = np.zeros((8, 24, 24), dtype=bool)
    mask[:4, 4:20, 4:20] = True
    mask[4:, 8:16, 8:16] = True
    mask[:4, 2:4, 4:20] = True  # slice-0-only agarose FOV edge
    out = tissue_mask_section_silhouettes(mask, n_serial_slices=2, dilate_px=0)
    assert out[:4, 10, 10].all()
    assert out[4:, 10, 10].all()
    assert out[:4, 2, 10].all()
    assert not out[4:, 2, 10].any()
    global_sil = tissue_mask_silhouette_xy(mask)
    assert global_sil[2, 10]  # global OR keeps slice-0 agarose on all Z


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


def test_section_silhouettes_follow_given_bounds():
    from linumpy.intensity.bias_field import tissue_mask_section_silhouettes

    mask = np.zeros((12, 8, 8), dtype=bool)
    mask[:4, 1:3, 1:3] = True  # section A, unequal depth
    mask[4:12, 5:7, 5:7] = True  # section B
    out = tissue_mask_section_silhouettes(mask, 2, bounds=[(0, 4), (4, 12)])
    assert out[:4, 1:3, 1:3].all() and not out[:4, 5:7, 5:7].any()
    assert out[4:, 5:7, 5:7].all() and not out[4:, 1:3, 1:3].any()
    # Equal chunks (6/6) would have leaked section B's footprint into z4:6.
    equal = tissue_mask_section_silhouettes(mask, 2)
    assert equal[4:6, 1:3, 1:3].any()


def test_slab_bounds_round_trip(tmp_path):
    import zarr

    from linumpy.mosaic.stacking import read_slab_bounds, write_slab_z_starts

    zarr.open_group(str(tmp_path / "v.ome.zarr"), mode="w")
    assert read_slab_bounds(tmp_path / "v.ome.zarr", 12) is None
    write_slab_z_starts(tmp_path / "v.ome.zarr", [0, 4], 12)
    assert read_slab_bounds(tmp_path / "v.ome.zarr", 12) == [(0, 4), (4, 12)]
    assert read_slab_bounds(tmp_path / "v.ome.zarr", 6) == [(0, 2), (2, 6)]


def _receding_block_mask() -> np.ndarray:
    """Two 10-plane sections. Tissue ends at x=40 on the first cut face and x=20 on the second.

    The first section carries signal over its whole cut-face footprint at every
    depth, as a slab does under a receding edge.
    """
    mask = np.zeros((20, 30, 60), dtype=bool)
    mask[:10, 5:25, 5:40] = True
    mask[10:, 5:25, 5:20] = True
    return mask


def test_cut_face_outline_tapers_a_receding_edge():
    mask = _receding_block_mask()
    out = tissue_mask_between_cut_faces(mask, 2)

    right_edge = [int(np.nonzero(out[z, 15])[0].max()) for z in range(10)]
    assert right_edge[0] == 39
    assert all(a >= b for a, b in pairwise(right_edge))
    assert right_edge[-1] <= 23
    # The last section has no following cut face and stays as it was.
    assert np.array_equal(out[10:], mask[10:])
    assert not (out & ~mask).any()


def test_cut_face_outline_keeps_tissue_that_widens_with_depth():
    mask = _receding_block_mask()[::-1].copy()
    out = tissue_mask_between_cut_faces(mask, 2)
    assert np.array_equal(out, mask)


def test_cut_face_outline_margin_keeps_more():
    mask = _receding_block_mask()
    tight = tissue_mask_between_cut_faces(mask, 2)
    loose = tissue_mask_between_cut_faces(mask, 2, margin_px=4)
    assert loose.sum() > tight.sum()
    assert not (tight & ~loose).any()


def test_cut_face_outline_skips_a_section_where_tissue_stops():
    """A footprint far smaller than the section's planes is not an outline."""
    mask = np.zeros((20, 30, 60), dtype=bool)
    mask[:10, 5:25, 5:40] = True
    mask[10:13, 5:25, 5:40] = True  # tissue ends three planes into the second section
    out = tissue_mask_between_cut_faces(mask, 2)
    assert np.array_equal(out, mask)


def test_drop_small_mask_components_removes_debris_only():
    mask = np.zeros((16, 40, 40), dtype=bool)
    mask[2:14, 5:30, 5:30] = True
    mask[4:6, 36:38, 36:38] = True  # speck in the agarose
    mask[8, 31, 20] = True  # within a block of the tissue: treated as part of it
    out = drop_small_mask_components(mask.copy(), 0.01)
    assert out[8, 15, 15]
    assert not out[4:6, 36:38, 36:38].any()
    assert out[8, 31, 20]
    assert np.array_equal(drop_small_mask_components(mask.copy(), 0.0), mask)


def _body_with_attachments():
    mask = np.zeros((48, 96, 96), dtype=bool)
    mask[8:40, 16:64, 16:64] = True  # the brain
    mask[20:24, 64:84, 36:40] = True  # thin strand hanging on its side
    mask[18:30, 84:94, 30:44] = True  # blob at the end of the strand
    mask[22:26, 4:8, 4:8] = True  # loose speck
    return mask


def test_drop_thin_mask_parts_cuts_a_strand_and_what_hangs_on_it():
    mask = _body_with_attachments()
    out = drop_thin_mask_parts(mask.copy(), opening_px=8, min_fraction=0.05)
    assert out[24, 40, 40]
    # A stub of a few cells stays at the foot of the strand.
    assert not out[22, 78:84, 37].any()
    assert not out[18:30, 86:94, 30:44].any()
    assert not out[22:26, 4:8, 4:8].any()
    # The body keeps its surface: nothing is shaved off its faces.
    assert out[8:40, 16:64, 16:64].all()


def test_drop_thin_mask_parts_keeps_a_second_large_body():
    mask = _body_with_attachments()
    mask[8:40, 70:94, 60:92] = True  # second solid piece, joined by nothing
    out = drop_thin_mask_parts(mask.copy(), opening_px=8, min_fraction=0.05)
    assert out[24, 80, 76]
    assert not drop_thin_mask_parts(mask.copy(), opening_px=8, min_fraction=0.9)[24, 80, 76]


def test_drop_thin_mask_parts_zero_distance_is_a_no_op():
    mask = _body_with_attachments()
    assert np.array_equal(drop_thin_mask_parts(mask.copy(), opening_px=0), mask)


def test_drop_thin_mask_parts_rejects_bad_arguments():
    mask = np.zeros((4, 8, 8), dtype=bool)
    with pytest.raises(ValueError, match="opening_px"):
        drop_thin_mask_parts(mask, opening_px=-1)
    with pytest.raises(ValueError, match="min_fraction"):
        drop_thin_mask_parts(mask, opening_px=4, min_fraction=1.0)
