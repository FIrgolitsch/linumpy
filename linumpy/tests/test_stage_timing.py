"""Tests for env-gated stage timing."""

import json

import numpy as np
import pytest

from linumpy.mosaic.grid import add_volume_to_mosaic, get_diffusion_blending_weights
from linumpy.utils import stage_timing


@pytest.fixture(autouse=True)
def _isolated_stage_timing(monkeypatch, tmp_path):
    drain = tmp_path / "_drain.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(drain))
    stage_timing.flush()
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    yield
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(drain))
    stage_timing.flush()


def test_unset_is_noop(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    sentinel = object()
    with stage_timing.span("diffusion_blend", device="cpu"):
        observed = sentinel
    assert observed is sentinel
    stage_timing.flush()
    assert not timing_path.exists()


def test_diffusion_values_unchanged(tmp_path, monkeypatch):
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, :8] = True
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    unset = get_diffusion_blending_weights(mask, factor=1, n_steps=2)
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(tmp_path / "direct.jsonl"))
    timed = get_diffusion_blending_weights(mask, factor=1, n_steps=2)
    np.testing.assert_array_equal(unset, timed)

    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    volume = np.ones((1, 16, 16), dtype=np.float32)
    mosaic = np.zeros((1, 20, 20), dtype=np.float32)
    mosaic[0, 0:16, 0:8] = 1.0
    add_volume_to_mosaic(volume, (0, 0, 0), mosaic, blending_method="diffusion", factor=1)
    stage_timing.flush()
    assert timing_path.is_file()
    lines = [line for line in timing_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["stage"] == "diffusion_blend"
    assert record["device"] == "cpu"
    stage_timing.flush()
    assert timing_path.read_text(encoding="utf-8").splitlines() == lines
