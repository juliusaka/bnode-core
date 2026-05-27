"""Load constant control values from an Excel file (strategy ``constantInput``)."""

import numpy as np
import pandas as pd
from bnode_core.config import data_gen_config


def constant_input_simulation_from_excel(cfg: data_gen_config) -> np.ndarray:
    """Load constant control values from an Excel file for steady-state simulations.
    
    Reads an Excel file with a sheet named 'Tabelle1' where each row defines one simulation 
    with constant control values. Control columns must be named to match config control names. 
    Each row's values are held constant for the entire sequence length.
    
    Useful for steady-state simulations or parameter sweeps with constant inputs.
    
    Args:
        cfg: Data generation configuration.
            cfg.pModel.RawData.controls_file_path: path to Excel file.
            cfg.pModel.RawData.controls: dict of control names (must match column names in Excel).
            cfg.pModel.RawData.Solver.sequence_length: length to replicate constant values.
    
    Returns:
        np.ndarray: Control values with shape (n_rows, n_controls, sequence_length).
            Each row from Excel becomes one sample with constant control values.
    
    Notes:
        Excel file structure:
        - Sheet name: 'Tabelle1'
        - First row: column headers matching control variable names
        - Each subsequent row: one set of constant control values for one simulation
    """
    file = pd.ExcelFile(cfg.pModel.RawData.controls_file_path)
    _df = file.parse(sheet_name='Tabelle1')
    _list = []
    for key in cfg.pModel.RawData.controls.keys():
        _list.append(_df[key].values)
    ctrl_values = np.array(_list).transpose()
    ctrl_values = np.expand_dims(ctrl_values, axis=2)
    ctrl_values = np.repeat(ctrl_values, (cfg.pModel.RawData.Solver.sequence_length), axis=2)
    return ctrl_values
