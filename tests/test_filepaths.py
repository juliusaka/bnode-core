import pytest
from pathlib import Path
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