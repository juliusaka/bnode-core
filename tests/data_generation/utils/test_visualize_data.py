import os
import io
import argparse
from pathlib import Path
import yaml
import h5py
import numpy as np
import types

import importlib.util

import bnode_core.filepaths

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

    outdir = Path('./tests/_results/visualization/')
    
    dataset_path = Path(r"resources/data/surrogate-test-data/data/raw_data/StratifiedHeatFlowModel_v3_c-RROCS/StratifiedHeatFlowModel_v3_c-RROCS_raw_data.hdf5").absolute()
    config_path = Path(r"resources/data/surrogate-test-data/data/raw_data/StratifiedHeatFlowModel_v3_c-RROCS/StratifiedHeatFlowModel_v3_c-RROCS_RawData_config.yaml").absolute()

    # Mock sys.argv for argparse
    import sys
    monkeypatch.setattr(sys, "argv", [
        "visualize_data.py",
        "--dataset", str(dataset_path),
        "--config", str(config_path),
        # "--figsize", "6,3",
        "--output_dir", str(outdir),
        "--print_limits", "controls,parameters",
        "--bins", "10",
    ])

    # run main
    mod.main()

    # Check that some pngs were created (at least one per class that exists)
    # Don't hardcode variable names since they depend on the actual dataset
    created_files = list(outdir.glob("*.png"))
    assert len(created_files) > 0, f"No plots were created in {outdir}"