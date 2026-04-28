import subprocess
from pathlib import Path

import pytest

from bnode_core.utils import hydra_mlflow_decorator


def test_resolve_git_hash_from_original_cwd(monkeypatch):
    expected_hash = "abc123def456"

    monkeypatch.setattr(
        hydra_mlflow_decorator.hydra.utils,
        "get_original_cwd",
        lambda: "/tmp/project-root",
    )

    def fake_run(cmd, check, capture_output, text, timeout):
        assert cmd[:3] == ["git", "-C", "/tmp/project-root"]
        assert cmd[3:] == ["rev-parse", "HEAD"]
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{expected_hash}\n", stderr="")

    monkeypatch.setattr(hydra_mlflow_decorator.subprocess, "run", fake_run)

    result = hydra_mlflow_decorator._resolve_git_hash()

    assert result == expected_hash


def test_resolve_git_hash_falls_back_to_unknown(monkeypatch):
    monkeypatch.setattr(
        hydra_mlflow_decorator.hydra.utils,
        "get_original_cwd",
        lambda: "/tmp/project-root",
    )

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(hydra_mlflow_decorator.subprocess, "run", fake_run)

    result = hydra_mlflow_decorator._resolve_git_hash()

    assert result == "unknown"


def test_resolve_git_hash_uses_cwd_when_original_cwd_unavailable(monkeypatch):
    expected_hash = "987zyx"
    cwd = Path("/tmp/current-cwd")

    def raise_original_cwd_error():
        raise RuntimeError("Hydra runtime not initialized")

    monkeypatch.setattr(
        hydra_mlflow_decorator.hydra.utils,
        "get_original_cwd",
        raise_original_cwd_error,
    )
    monkeypatch.setattr(hydra_mlflow_decorator.Path, "cwd", lambda: cwd)

    def fake_run(cmd, check, capture_output, text, timeout):
        assert cmd[:3] == ["git", "-C", str(cwd)]
        return subprocess.CompletedProcess(cmd, 0, stdout=expected_hash, stderr="")

    monkeypatch.setattr(hydra_mlflow_decorator.subprocess, "run", fake_run)

    result = hydra_mlflow_decorator._resolve_git_hash()

    assert result == expected_hash


def test_resolve_resume_run_context_requires_restart_run_id(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()

    with pytest.raises(ValueError, match="missing mlflow_run_id"):
        hydra_mlflow_decorator._resolve_resume_run_context(
            cfg_run_id=None,
            restart_metadata={
                "hydra_output_dir": str(hydra_output_dir),
                "mlflow_run_id": None,
                "mlflow_tracking_uri": "file:///mlruns",
                "mlflow_experiment_name": "exp",
            },
            hydra_output_dir=hydra_output_dir,
            explicit_restart_state_path=None,
            current_tracking_uri="file:///mlruns",
            current_experiment_name="exp",
        )


def test_resolve_resume_run_context_rejects_tracking_uri_mismatch(tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()

    with pytest.raises(ValueError, match="tracking URI"):
        hydra_mlflow_decorator._resolve_resume_run_context(
            cfg_run_id=None,
            restart_metadata={
                "hydra_output_dir": str(hydra_output_dir),
                "mlflow_run_id": "run-123",
                "mlflow_tracking_uri": "file:///saved-mlruns",
                "mlflow_experiment_name": "exp",
            },
            hydra_output_dir=hydra_output_dir,
            explicit_restart_state_path=None,
            current_tracking_uri="file:///current-mlruns",
            current_experiment_name="exp",
        )


def test_resolve_resume_run_context_allows_explicit_restart_artifact_new_output_dir(tmp_path):
    hydra_output_dir = tmp_path / "target-run"
    hydra_output_dir.mkdir()
    source_output_dir = tmp_path / "source-run"
    source_output_dir.mkdir()

    resolved_run_id, allow_hydra_output_param_mismatch = (
        hydra_mlflow_decorator._resolve_resume_run_context(
            cfg_run_id=None,
            restart_metadata={
                "hydra_output_dir": str(source_output_dir),
                "mlflow_run_id": "run-123",
                "mlflow_tracking_uri": "file:///mlruns",
                "mlflow_experiment_name": "exp",
            },
            hydra_output_dir=hydra_output_dir,
            explicit_restart_state_path=str(source_output_dir / "training_restart.pt"),
            current_tracking_uri="file:///mlruns/",
            current_experiment_name="exp",
        )
    )

    assert resolved_run_id == "run-123"
    assert allow_hydra_output_param_mismatch is True
