"""Tests for residual-bound policy in linumpy.registration.refinement."""

import numpy as np
import pytest

from linumpy.registration.refinement import apply_refinement_bounds, tissue_ncc


def test_scale_projects_runaway_onto_bound():
    tx, ty, rot, rejected = apply_refinement_bounds(60.0, 0.0, 0.0, max_translation_px=10.0, max_rotation_deg=2.0)
    assert not rejected
    assert abs((tx**2 + ty**2) ** 0.5 - 10.0) < 1e-9
    assert tx == 10.0
    assert rot == 0.0


def test_reject_returns_identity_on_runaway():
    tx, ty, rot, rejected = apply_refinement_bounds(
        60.0, 0.0, 0.5, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert rejected
    assert (tx, ty, rot) == (0.0, 0.0, 0.0)


def test_reject_returns_identity_on_rotation_bound():
    tx, ty, rot, rejected = apply_refinement_bounds(
        1.0, 0.0, 5.0, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert rejected
    assert (tx, ty, rot) == (0.0, 0.0, 0.0)


def test_reject_keeps_small_residual():
    tx, ty, rot, rejected = apply_refinement_bounds(
        2.0, -1.0, 0.3, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert not rejected
    assert (tx, ty, rot) == (2.0, -1.0, 0.3)


def test_tissue_ncc_identical_is_one():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = rng.random((16, 16)).astype(np.float32) + 0.2
    assert tissue_ncc(img, img) == pytest.approx(1.0)


def test_tissue_ncc_shifted_is_lower():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = rng.random((16, 16)).astype(np.float32) + 0.2
    shifted = np.roll(img, 8, axis=1)
    assert tissue_ncc(img, shifted) < 0.5
