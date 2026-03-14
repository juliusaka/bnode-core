import subprocess
from pathlib import Path

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
