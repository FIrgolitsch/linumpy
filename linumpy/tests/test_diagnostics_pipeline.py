"""Direct-import tests for ``linumpy.diagnostics.pipeline``.

Exercises the extracted diagnostics library directly (no ``importlib`` script
loading), covering the same gate-check/aggregation behavior locked by
``scripts/tests/diagnostics/test_diagnose_pipeline.py`` plus additional
coverage for header printing, CPU/package checks, and Nextflow suggestions
(Plan 07.1-08).
"""

from unittest import mock

from linumpy.diagnostics.pipeline import SystemDiagnostics, get_terminal_width, print_header, print_subheader


def _make_diag(total_cores, total_gb, gpu_available=False):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = total_cores
    diag.results["memory"]["total_gb"] = total_gb
    diag.results["gpu"]["available"] = gpu_available
    return diag


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


def test_get_terminal_width_returns_positive_int():
    width = get_terminal_width()
    assert isinstance(width, int)
    assert width > 0


def test_get_terminal_width_falls_back_on_oserror(monkeypatch):
    import os

    def _raise(*_a, **_kw):
        raise OSError("no tty")

    monkeypatch.setattr(os, "get_terminal_size", _raise)
    assert get_terminal_width() == 80


def test_print_header_outputs_title(capsys):
    print_header("My Section")
    out = capsys.readouterr().out
    assert "My Section" in out
    assert "=" in out


def test_print_subheader_outputs_title(capsys):
    print_subheader("Sub")
    out = capsys.readouterr().out
    assert "--- Sub ---" in out


# ---------------------------------------------------------------------------
# Nextflow suggestions
# ---------------------------------------------------------------------------


def test_nextflow_suggestions_reserved_cpus_floor():
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 2  # floor at 2, not 8 // 12 == 0


def test_nextflow_suggestions_reserved_cpus_scales():
    diag = _make_diag(total_cores=48, total_gb=64.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.reserved_cpus"] == 4  # 48 // 12
    assert suggestions["params.processes"] == 14  # min(16, max(1, (48 - 4) // 3))


def test_nextflow_suggestions_capped_at_16_processes():
    diag = _make_diag(total_cores=256, total_gb=512.0)
    diag.check_nextflow_config()
    suggestions = diag.results["nextflow"]["suggestions"]
    assert suggestions["params.processes"] == 16


def test_nextflow_suggestions_use_gpu_reflects_availability():
    diag = _make_diag(total_cores=8, total_gb=32.0, gpu_available=True)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["suggestions"]["params.use_gpu"] is True


def test_nextflow_suggestions_enable_cpu_limits_always_true():
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["suggestions"]["params.enable_cpu_limits"] is True


def test_nextflow_suggestions_min_one_process():
    diag = _make_diag(total_cores=2, total_gb=4.0)
    diag.check_nextflow_config()
    # reserved=2, available=0, max(1, 0//3)=1
    assert diag.results["nextflow"]["suggestions"]["params.processes"] == 1


def test_nextflow_installed_flag_set_when_subprocess_succeeds(monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        return mock.Mock(returncode=0, stdout="nextflow version 23.10.0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["installed"] is True


def test_nextflow_installed_false_when_not_found(monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        raise FileNotFoundError("nextflow not found")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["installed"] is False


def test_nextflow_installed_true_when_returncode_nonzero(monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        return mock.Mock(returncode=1, stdout="", stderr="oops")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = _make_diag(total_cores=8, total_gb=32.0)
    diag.check_nextflow_config()
    assert diag.results["nextflow"]["installed"] is True


# ---------------------------------------------------------------------------
# CPU check
# ---------------------------------------------------------------------------


def test_check_cpu_records_total_cores(capsys):
    diag = SystemDiagnostics()
    total = diag.check_cpu()
    assert isinstance(total, int)
    assert total >= 1
    assert diag.results["cpu"]["total_cores"] == total


def test_check_cpu_records_thread_env_vars(capsys):
    diag = SystemDiagnostics()
    diag.check_cpu()
    # All thread vars should be recorded (either value or "(not set)")
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "LINUMPY_MAX_CPUS"]:
        assert var in diag.results["cpu"]


# ---------------------------------------------------------------------------
# Python packages check
# ---------------------------------------------------------------------------


def test_check_python_packages_records_versions(capsys):
    diag = SystemDiagnostics()
    diag.check_python_packages()
    assert "version" in diag.results["python"]
    # numpy should always be installed in the test env
    assert diag.results["python"]["numpy"] is not None


def test_check_python_packages_records_missing_as_none(capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "pqdm":
            raise ImportError("no pqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    diag = SystemDiagnostics()
    diag.check_python_packages()
    assert diag.results["python"]["pqdm"] is None
    assert any("pqdm" in i for i in diag.results["issues"])


# ---------------------------------------------------------------------------
# BaSiC error handling
# ---------------------------------------------------------------------------


def test_handle_basic_error_missing_basicpy(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'basicpy'")
    out = capsys.readouterr().out
    assert "basicpy not installed" in out


def test_handle_basic_error_missing_torch(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("ModuleNotFoundError: No module named 'torch'")
    out = capsys.readouterr().out
    assert "PyTorch not installed" in out


def test_handle_basic_error_cuda_oom(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("RuntimeError: CUDA error: out of memory")
    out = capsys.readouterr().out
    assert "GPU out of memory" in out


def test_handle_basic_error_generic(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("some traceback\nERROR:something weird happened\nmore text")
    out = capsys.readouterr().out
    assert "something weird happened" in out


def test_handle_basic_error_no_error_prefix(capsys):
    diag = SystemDiagnostics()
    diag._handle_basic_error("just a generic failure message")
    out = capsys.readouterr().out
    assert "just a generic failure message" in out


def test_handle_basic_error_verbose_shows_full_output(capsys):
    diag = SystemDiagnostics(verbose=True)
    diag._handle_basic_error("ERROR:boom\nline2\nline3")
    out = capsys.readouterr().out
    assert "boom" in out
    assert "Complete output" in out


# ---------------------------------------------------------------------------
# GPU checks (mocked — no CUDA in CI)
# ---------------------------------------------------------------------------


def test_check_cupy_not_installed(capsys):
    diag = SystemDiagnostics()
    diag._check_cupy()
    out = capsys.readouterr().out
    assert "CuPy not installed" in out
    assert diag.results["gpu"]["cupy_version"] is None
    assert any("CuPy not installed" in i for i in diag.results["issues"])


def test_check_linumpy_gpu_import_error(capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "linumpy.gpu":
            raise ImportError("no gpu module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    diag = SystemDiagnostics()
    diag._check_linumpy_gpu()
    out = capsys.readouterr().out
    assert "Cannot import linumpy.gpu" in out
    assert diag.results["gpu"]["linumpy_gpu_available"] is False


def test_check_linumpy_gpu_records_availability(capsys):
    diag = SystemDiagnostics()
    diag._check_linumpy_gpu()
    # linumpy.gpu imports successfully in test env (CPU fallback)
    assert "linumpy_gpu_available" in diag.results["gpu"]


def test_check_gpu_no_nvidia_smi(capsys, monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = SystemDiagnostics()
    diag.check_gpu()
    out = capsys.readouterr().out
    assert "nvidia-smi not found" in out
    assert diag.results["gpu"]["available"] is False


def test_check_gpu_nvidia_smi_timeout(capsys, monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=30)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = SystemDiagnostics()
    diag.check_gpu()
    out = capsys.readouterr().out
    assert "timed out" in out
    assert diag.results["gpu"]["available"] is False


def test_check_gpu_nvidia_smi_no_output(capsys, monkeypatch):
    import subprocess

    def _fake_run(*_a, **_kw):
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diag = SystemDiagnostics()
    diag.check_gpu()
    assert diag.results["gpu"]["available"] is False


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def test_generate_report_lists_issues(capsys):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 8
    diag.results["memory"]["total_gb"] = 16.0
    diag.results["gpu"]["available"] = False
    diag.results["gpu"]["cupy_working"] = False
    diag.results["issues"].append("Low available memory: 2.0 GB")

    results = diag.generate_report()
    out = capsys.readouterr().out

    assert results is diag.results
    assert "Low available memory: 2.0 GB" in out
    assert "CPU cores: 8" in out
    assert "NVIDIA GPU: Not available" in out


def test_generate_report_no_issues_skips_section(capsys):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 4
    diag.results["memory"]["total_gb"] = 8.0
    diag.results["gpu"]["available"] = True
    diag.results["gpu"]["cupy_working"] = True

    diag.generate_report()
    out = capsys.readouterr().out

    assert "Issues Found" not in out
    assert "NVIDIA GPU: Available" in out
    assert "CuPy GPU (linumpy): Working" in out


def test_generate_report_gpu_not_working(capsys):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 4
    diag.results["memory"]["total_gb"] = 8.0
    diag.results["gpu"]["available"] = True
    diag.results["gpu"]["cupy_working"] = False

    diag.generate_report()
    out = capsys.readouterr().out

    assert "NVIDIA GPU: Available" in out
    assert "CuPy GPU (linumpy): Not available" in out


def test_generate_report_zero_memory(capsys):
    diag = SystemDiagnostics()
    diag.results["cpu"]["total_cores"] = 4
    diag.results["memory"]["total_gb"] = 0
    diag.results["gpu"]["available"] = False
    diag.results["gpu"]["cupy_working"] = False

    diag.generate_report()
    out = capsys.readouterr().out
    assert "Total RAM: 0.0 GB" in out


# ---------------------------------------------------------------------------
# Edge cases: empty input
# ---------------------------------------------------------------------------


def test_system_diagnostics_initial_structure():
    diag = SystemDiagnostics()
    assert diag.results["cpu"] == {}
    assert diag.results["memory"] == {}
    assert diag.results["gpu"] == {}
    assert diag.results["python"] == {}
    assert diag.results["nextflow"] == {}
    assert diag.results["linumpy"] == {}
    assert diag.results["issues"] == []
    assert diag.results["recommendations"] == []
    assert "timestamp" in diag.results


def test_system_diagnostics_verbose_flag():
    diag = SystemDiagnostics(verbose=True)
    assert diag.verbose is True
    diag2 = SystemDiagnostics()
    assert diag2.verbose is False
