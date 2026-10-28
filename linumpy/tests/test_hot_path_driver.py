"""Tests for the hot-path timing driver helpers."""

import ast
import inspect
import types

import pytest

from scripts.timing import linum_time_hot_path as driver

_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def _imported_modules(module: types.ModuleType) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", maxsplit=1)[0])
    return names


def test_thread_env_smaller_host():
    assert "numpy" not in _imported_modules(driver)
    assert "linumpy.config.threads" not in inspect.getsource(driver)
    env = driver.thread_env(16, max_cpus=44, processes=8)
    assert env.get("OMP_NUM_THREADS") == "2"
    assert env.get("LINUMPY_MAX_CPUS") == "16"
    for name in _THREAD_NAMES:
        assert env.get(name) == "2"
    assert "5" not in env.values()


def test_thread_env_full_host():
    env = driver.thread_env(44, max_cpus=44, processes=8)
    assert env.get("OMP_NUM_THREADS") == "5"
    assert env.get("LINUMPY_MAX_CPUS") == "44"
    for name in _THREAD_NAMES:
        assert env.get(name) == "5"


def test_output_prefix_rejects_escape(tmp_path):
    root = tmp_path / "timing"
    root.mkdir()
    inside = root / "hot_path.json"
    inside.write_text("{}", encoding="utf-8")
    resolved = driver.assert_under_timing(inside, root=root)
    assert resolved == inside.resolve()

    outside = tmp_path / "output"
    outside.mkdir()
    with pytest.raises(ValueError):
        driver.assert_under_timing(outside, root=root)

    sibling = tmp_path / "timing-evil"
    sibling.mkdir()
    with pytest.raises(ValueError):
        driver.assert_under_timing(sibling, root=root)

    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        driver.assert_under_timing(link, root=root)
