import os
import io
import argparse
from pathlib import Path
import yaml
import h5py
import numpy as np
import types

import importlib.util

def _load_module():
    # Ensure matplotlib uses non-interactive backend before module import
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "src" / "bnode_core" / "data_generation" / "utils" / "visualize_data.py"
    spec = importlib.util.spec_from_file_location("visualize_data_module", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_load_yaml(tmp_path):
    mod = _load_module()
    cfg = {"controls": {"c1": [0, 1]}, "parameters": {"p1": [-1, 1]}}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    loaded = mod.load_yaml(cfg_file)
    assert loaded == cfg

def test_plot_variable_1d_and_2d(tmp_path, monkeypatch):
    mod = _load_module()
    # prevent interactive show
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)

    outdir = tmp_path / "out"
    outdir.mkdir()

    # 1D data (no time)
    raw_1d = np.random.normal(size=100)
    mod.plot_variable("controls", "var1", raw_1d, outdir, n_bins=20, limits=[-3, 3], time=None, verbose=False)
    p1 = outdir / "controls_var1.png"
    assert p1.exists() and p1.stat().st_size > 0

    # 2D data (with time) -> density_overlay branch
    raw_2d = np.random.normal(size=(20, 50))
    time = np.linspace(0, 1, raw_2d.shape[1])
    mod.plot_variable("states", "var2", raw_2d, outdir, n_bins=20, limits=None, time=time, verbose=False)
    p2 = outdir / "states_var2.png"
    assert p2.exists() and p2.stat().st_size > 0

def test_main_creates_plots(tmp_path, monkeypatch):
    mod = _load_module()
    # prevent interactive show
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)

    h5_path = tmp_path / "test_raw_data.h5"
    _create_raw_h5(h5_path)

    # create config file matching expected structure for raw dataset
    cfg = {
        "controls": {"v1": [-1, 1], "v2": [-2, 2]},
        "outputs": {"v1": [-1, 1], "v2": [-2, 2]},
        "states": {"v1": [-1, 1], "v2": [-2, 2]},
        "parameters": {"v1": [-1, 1], "v2": [-2, 2]},
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    outdir = tmp_path / "visualization"
    args = argparse.Namespace(
        dataset=str(h5_path),
        config=str(cfg_file),
        figsize="6,3",
        output_dir=str(outdir),
        print_limits="controls,parameters",
        bins=10,
        verbose=False,
    )

    # run main
    mod.main(args)

    # check that pngs were created for each variable
    expected = []
    for cls in ["controls", "outputs", "states", "parameters"]:
        for name in ["v1", "v2"]:
            expected.append(outdir / f"{cls}_{name}.png")

    for p in expected:
        assert p.exists() and p.stat().st_size > 0