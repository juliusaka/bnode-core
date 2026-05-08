import pytest
from types import SimpleNamespace

from bnode_core import filepaths
from bnode_core.filepaths import config_dir_auto_recognize

def test_returns_config_dir_when_bnode_and_config_exist(tmp_path, monkeypatch):
    # create .bnode_project file and ./config/ directory
    (tmp_path / ".bnode_project").write_text("")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    res = config_dir_auto_recognize()
    assert res.resolve() == config_dir.resolve()

def test_raises_when_nothing_found(tmp_path, monkeypatch):
    # neither .bnode_project nor resources/config nor ./config exist
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as exc:
        config_dir_auto_recognize()
    assert "Please ensure you are in a correct working directory" in str(exc.value)


def test_training_restart_paths_use_restart_filenames(monkeypatch, tmp_path):
    hydra_output_dir = tmp_path / "hydra-run"
    hydra_output_dir.mkdir()
    hydra_cfg = SimpleNamespace(runtime=SimpleNamespace(output_dir=str(hydra_output_dir)))
    monkeypatch.setattr(filepaths.hydra.core.hydra_config.HydraConfig, "get", lambda: hydra_cfg)

    assert (
        filepaths.filepath_training_outer_restart_state_current_hydra_output()
        == hydra_output_dir / "training_outer_restart.pt"
    )
    assert (
        filepaths.filepath_training_inner_restart_state_current_hydra_output()
        == hydra_output_dir / "training_inner_restart.pt"
    )
    assert (
        filepaths.filepath_lr_schedulers_current_hydra_output()
        == hydra_output_dir / "lr_schedulers.pt"
    )
    assert (
        filepaths.filepath_grad_scaler_current_hydra_output()
        == hydra_output_dir / "grad_scaler.pt"
    )
