"""
Error histogram visualization for neural ODE datasets.

This module provides functions to analyze and visualize the distribution of errors
between different variables in a dataset, showing how errors are distributed
across samples.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import sys
import argparse
import filepaths

# Import useful functions from plot_dataset and plot_error_correlation
from plot_dataset import (
    calculate_scaling_dict,
    calculate_normalization_stats
)
from common import save_figure
import pandas as pd


def calculate_variable_errors(hdf5_file: h5py.File,
                           context_key: str,
                           normalization_stats: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, np.ndarray]:
    """
    Calculate normalized errors (including sign) for each variable in the specified context.

    All variables are assumed to have normalization statistics.
    Returns a dict mapping var_type -> errors array shaped:
      - time series: (n_samples, n_vars, n_timesteps)
      - parameters:  (n_samples, n_vars, 1)
    """
    variable_errors: Dict[str, np.ndarray] = {}
    variable_types = ['states', 'controls', 'outputs', 'parameters']

    for var_type in variable_types:
        if var_type in hdf5_file[context_key] and f'{var_type}_hat' in hdf5_file[context_key]:
            truth_data = hdf5_file[context_key][var_type][:]
            pred_data = hdf5_file[context_key][f'{var_type}_hat'][:]
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')

            # Build mean/std arrays in the same order as var_names
            means = np.array([normalization_stats[var_type][name][0] for name in var_names], dtype=float)
            stds = np.array([normalization_stats[var_type][name][1] for name in var_names], dtype=float)

            if var_type == 'parameters':
                # Ensure a time axis of length 1 for consistent handling
                truth_valid = truth_data[:, :, None]  # (n_samples, n_vars, 1)
                pred_valid = pred_data[:, :, None]    # (n_samples, n_vars, 1)
                means_b = means.reshape(1, -1, 1)
                stds_b = stds.reshape(1, -1, 1)
            else:
                # Time series 3D: (n_samples, n_vars, n_timesteps)
                truth_valid = truth_data
                pred_valid = pred_data
                means_b = means.reshape(1, -1, 1)
                stds_b = stds.reshape(1, -1, 1)

            # Vectorized calculation of normalized errors (with sign)
            normalized_errors = (truth_valid - pred_valid) / stds_b
            variable_errors[var_type] = normalized_errors

    return variable_errors


def get_variable_names(hdf5_file, normalization_stats, variable_names):
    for var_type in normalization_stats.keys():
        var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
        variable_names[var_type] = var_names.tolist()
    return variable_names


def collect_error_data(variable_errors: Dict[str, np.ndarray], variable_names: Dict[str, List[str]]):
    """
    Build lists of names and per-variable error arrays suitable for histogram plotting.

    Returns:
      - all_variable_names: ["type: name", ...]
      - all_error_vectors: list of arrays per variable with shape
            (n_samples,) or (n_samples, n_timesteps)
    """
    all_variable_names: List[str] = []
    all_error_vectors: List[np.ndarray] = []

    for var_type, errors in variable_errors.items():
        names = variable_names.get(var_type, [])
        if errors.ndim == 3:  # (n_samples, n_vars, n_timesteps)
            for i, var_name in enumerate(names):
                all_variable_names.append(f"{var_type}: {var_name}")
                all_error_vectors.append(errors[:, i, :])  # (n_samples, n_timesteps)
        elif errors.ndim == 2:  # (n_samples, n_vars)
            for i, var_name in enumerate(names):
                all_variable_names.append(f"{var_type}: {var_name}")
                all_error_vectors.append(errors[:, i])  # (n_samples,)
        else:
            logging.warning(f"Unexpected error array shape for {var_type}: {errors.shape}")

    return all_variable_names, all_error_vectors


def _draw_error_histogram(
    ax: plt.Axes,
    errors: np.ndarray,
    *,
    bins: int,
    error_range: float,
    title: Optional[str] = None,
    show_legend: bool = False,
    annotate_text: bool = False,
    xlabel: str = "Normalized Error",
    ylabel: str = "Density",
    title_fontsize: Optional[int] = None,
    label_fontsize: Optional[int] = None,
) -> None:
    """Draw a single histogram with consistent styling, lines and annotations.

    Args:
        ax: Matplotlib Axes to draw on
        errors: 1D array of errors (positive and negative values)
        bins: Number of bins for the histogram
        error_range: Range for x-axis and histogram [-value, +value]
        title: Optional title for this subplot
        show_legend: If True, adds a legend with stats
        annotate_text: If True, adds a compact text box with stats (mutually exclusive with legend)
        xlabel: X-axis label
        ylabel: Y-axis label
        title_fontsize: Optional fontsize for title
        label_fontsize: Optional fontsize for axis labels
    """
    # Fixed histogram range from -error_range to +error_range
    hist_range = (-error_range, error_range)

    # Pre-compute statistics
    min_val = float(np.min(errors))
    max_val = float(np.max(errors))
    mean_val = float(np.mean(errors))
    std_val = float(np.std(errors))
    p995_val = float(np.percentile(errors, 99.5))
    p005_val = float(np.percentile(errors, 0.5))

    # Histogram
    ax.hist(errors, bins=bins, alpha=0.7, color='blue', range=hist_range, density=True)

    # Vertical lines (no labels here; legend handles built separately for consistency)
    ax.axvline(x=mean_val, color='red', linestyle='--')
    ax.axvline(x=mean_val + std_val, color='green', linestyle=':')
    ax.axvline(x=mean_val - std_val, color='green', linestyle=':')
    ax.axvline(x=min_val, color='blue', linestyle='-')
    ax.axvline(x=max_val, color='blue', linestyle='-')
    ax.axvline(x=p995_val, color='purple', linestyle='--')
    ax.axvline(x=p005_val, color='purple', linestyle='--')

    # Zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    # Title and labels
    if title is not None:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    if label_fontsize is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Legend or annotation box
    if show_legend and not annotate_text:
        legend_handles = [
            plt.Line2D([0], [0], color='red', linestyle='--', label=f'Mean: {mean_val:.3f}'),
            plt.Line2D([0], [0], color='green', linestyle=':', label=f'Std: {std_val:.3f}'),
            plt.Line2D([0], [0], color='blue', linestyle='-', label=f'Min: {min_val:.3f}, Max: {max_val:.3f}'),
            plt.Line2D([0], [0], color='purple', linestyle='--', label=f'0.5%: {p005_val:.3f}, 99.5%: {p995_val:.3f}'),
        ]
        ax.legend(handles=legend_handles)
    elif annotate_text and not show_legend:
        # Compact text summary to save space in grids
        ax.text(
            0.95,
            0.95,
            f'μ: {mean_val:.3f}\nσ: {std_val:.3f}\nmin: {min_val:.3f}\nmax: {max_val:.3f}\n0.5%: {p005_val:.3f}\n99.5%: {p995_val:.3f}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
            fontsize=(label_fontsize if label_fontsize is not None else None)
        )

    # Axes range and grid
    ax.set_xlim(-error_range, error_range)
    ax.grid(True, alpha=0.3)


def plot_error_histogram(all_variable_names: List[str],
                        error_vectors: List[np.ndarray],
                        output_folder: Path,
                        output_filename: str = "error_histogram",
                        file_type: str = 'pdf',
                        bins: int = 50,
                        individual_plots: bool = False,
                        dpi: int = 300,
                        test_mode: bool = False,
                        error_range: float = 0.6,
                        num_timepoints: int = -1,
                        figsize: Tuple[int, int] = (5,3)
                        ) -> None:
    """
    Create and save histograms showing error distributions for all variables.
    
    Args:
        all_variable_names: List of variable names (with type prefix)
        error_vectors: List of error arrays for each variable. Each item can be:
            - 1D: (n_samples,)
            - 2D: (n_samples, n_timesteps) (time-series)
        output_folder: Path to save the output
        output_filename: Base filename for the output
        file_type: File type to save plots as (e.g., 'png', 'pdf')
        bins: Number of histogram bins
        individual_plots: If True, save each histogram as individual file
        dpi: Resolution in dots per inch for output files
        test_mode: If True, only plots the first few variables for quick testing
        error_range: Range for error histograms [-value, +value] (default: 0.6)
        num_timepoints: Number of time points to include per variable. If -1, use all (default: -1)
        figsize: Size of the figure for one axis.
    """
    # If test mode is enabled, only use the first few variables
    if test_mode:
        max_test_vars = min(5, len(all_variable_names))
        all_variable_names = all_variable_names[:max_test_vars]
        error_vectors = error_vectors[:max_test_vars]
        logging.info(f"Test mode enabled: plotting only first {max_test_vars} variables")

    def _prepare_errors_1d(err: np.ndarray, n_tp: int) -> np.ndarray:
        if len(err.shape) == 2:
            if n_tp > 0 and n_tp < err.shape[1]:
                return err[:, :n_tp].flatten()
            elif n_tp > err.shape[1]:
                raise ValueError(f"num_timepoints {n_tp} exceeds available {err.shape[1]}")
            else:
                return err.flatten()
        elif len(err.shape) == 1:
            return err
        else:
            raise ValueError(f"Unexpected error array shape: {err.shape}")

    n_variables = len(all_variable_names)
    
    # Calculate optimal grid layout for subplots
    n_cols = min(3, n_variables)
    n_rows = int(np.ceil(n_variables / n_cols))
    
    if individual_plots:
        # Individual plots - create a separate figure for each variable
        for i, (var_name, errors) in enumerate(zip(all_variable_names, error_vectors)):
            logging.info(f"Plotting histogram for variable {i+1}/{n_variables}: {var_name}")
            
            # Prepare 1D error vector (selecting timepoints if needed)
            errors_1d = _prepare_errors_1d(errors, num_timepoints)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            _draw_error_histogram(
                ax,
                errors_1d,
                bins=bins,
                error_range=error_range,
                title=f"Error Distribution - {var_name} - {num_timepoints if num_timepoints > 0 else 'all'} timepoints" ,
                show_legend=True,
                annotate_text=False,
            )
            
            # Save the figure
            var_filename = f"{output_filename}_{var_name.replace(' ', '_')}_tp_{num_timepoints if num_timepoints > 0 else 'all'}" 
            file_type = file_type if file_type is not None else 'png'
            save_figure(fig, output_folder, var_filename, file_type, dpi=dpi)
            plt.close(fig)
    else:
        # Combined plot - all variables in one figure
        logging.info(f"Creating combined histogram plot for {n_variables} variables")
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        gs = gridspec.GridSpec(n_rows, n_cols)
        gs.update(wspace=0.4, hspace=0.4)
        
        for i, (var_name, errors) in enumerate(zip(all_variable_names, error_vectors)):
            ax = plt.subplot(gs[i // n_cols, i % n_cols])

            # Prepare 1D error vector (selecting timepoints if needed)
            errors_1d = _prepare_errors_1d(errors, num_timepoints)

            _draw_error_histogram(
                ax,
                errors_1d,
                bins=bins,
                error_range=error_range,
                title=var_name,
                show_legend=False,
                annotate_text=True,
                title_fontsize=10,
                label_fontsize=8,
            )
        
        # Add overall title
        plt.suptitle(f"Error Distribution Histograms {num_timepoints if num_timepoints > 0 else 'all'} timepoints",
                      fontsize=16, y=0.995, va='top')
        
        # Save the figure
        output_filename = f"{output_filename}_tp_{num_timepoints if num_timepoints > 0 else 'all'}"
        file_type = file_type if file_type is not None else 'pdf'
        save_figure(fig, output_folder, output_filename, file_type, dpi=dpi)
        plt.close(fig)

def make_pandas_dataframe(variable_names, variable_errors) -> 'pd.DataFrame':
    """
    Create a pandas DataFrame summarizing error statistics for each variable.
    For each variable, report:
    - VRMSE (sqrt(mean(error^2) across all samples/timepoints))
    - Std of |error|
    - Max mean |error| per sample
    - Max |error| over all samples/timepoints
    - Per-timepoint VRMSE columns
    Args:
        variable_names: Dict[var_type -> List[var_name]]
        variable_errors: Dict[var_type -> np.ndarray]
            arrays shaped (n_samples, n_vars, n_timesteps) or (n_samples, n_vars)
    Returns:
        pd.DataFrame with one row per variable
    """
    import pandas as pd
    records = []
    for var_type, names in variable_names.items():
        if var_type not in variable_errors:
            continue
        errors_arr = variable_errors[var_type]
        if errors_arr.ndim == 2:  # (n_samples, n_vars)
            errors_arr = errors_arr[:, :, None]  # add time axis of length 1
        # Now (n_samples, n_vars, T)
        n_samples, n_vars, T = errors_arr.shape
        for i, var_name in enumerate(names):
            if i >= n_vars:
                # In case of name/array mismatch, skip
                continue
            e = errors_arr[:, i, :]  # (n_samples, T)
            e_abs = np.abs(e)
            vrmse_overall = float(np.sqrt(np.mean(e**2)))
            std_err = float(np.std(e_abs))
            max_err_sample = float(np.max(np.mean(e_abs, axis=1)))
            max_err_timepoint = float(np.max(e_abs))
            record = {
                'Variable type': var_type,
                'Variable': var_name,
                'VRMSE': vrmse_overall,
                'Std error': std_err,
                'Max error sample': max_err_sample,
                'Max error timepoint': max_err_timepoint,
            }
            for t in range(T):
                vrmse_t = float(np.sqrt(np.mean(e[:, t]**2)))
                record[f'VRMSE_t{t}'] = vrmse_t
            records.append(record)
    df = pd.DataFrame.from_records(records)
    return df  

def generate_error_histograms(hdf5_file: h5py.File,
                            context_key: str,
                            output_folder: Path,
                            file_type: str = 'pdf',
                            bins: int = 50,
                            individual_plots: bool = False,
                            dpi: int = 300,
                            test_mode: bool = False,
                            error_range: float = 0.6,
                            num_timepoints: int = -1) -> None:
    """
    Generate histograms showing error distributions for all variables in a dataset.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to analyze ('test', 'validation', etc.)
        output_folder: Path to save the plots
        file_type: File type to save plots as (e.g., 'png', 'pdf')
        bins: Number of histogram bins
        individual_plots: If True, save each histogram as an individual file
        dpi: Resolution in dots per inch for output files
        test_mode: If True, only plots the first few variables for quick testing
        error_range: Range for error histograms [-value, +value] (default: 0.6)
        num_timepoints: Number of time points to include per variable. If -1, use all (default: -1)
    """
    
    # Calculate normalization stats from training data
    normalization_stats = calculate_normalization_stats(hdf5_file, 'train')
    
    # Calculate errors for each variable
    variable_errors = calculate_variable_errors(hdf5_file, context_key, normalization_stats)

    # Get variable names
    variable_names = {}
    variable_names = get_variable_names(hdf5_file, normalization_stats, variable_names)

    # reshape error data for plotting
    all_variable_names, error_vectors = collect_error_data(variable_errors, variable_names)

    # make informative pandas dataframe
    df = make_pandas_dataframe(variable_names, variable_errors)
    
    # export as excel
    df.to_excel(output_folder / f'error_statistics_{context_key}.xlsx', index=False)

    # Generate error histograms
    output_filename = f"error_histogram_{context_key}"
    if test_mode:
        output_filename += "_test_mode"
    
    plot_error_histogram(
        all_variable_names, error_vectors, output_folder,
        output_filename=output_filename, file_type=file_type,
        bins=bins, individual_plots=individual_plots,
        dpi=dpi, test_mode=test_mode, error_range=error_range,
        num_timepoints=num_timepoints,
    )
    
    logging.info(f"Error histograms created for context {context_key}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot error histograms for neural ODE datasets')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to HDF5 dataset file (local or ML artifacts)')
    parser.add_argument('--context', type=str, default='common_test',
                      help='Context to analyze (default: test)')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Output folder for plots')
    parser.add_argument('--file_type', type=str, default=None,
                      choices=['pdf', 'png', 'jpg', 'svg'],
                      help='File type for saving plots')
    parser.add_argument('--bins', type=int, default=50,
                      help='Number of histogram bins (default: 50)')
    parser.add_argument('--individual_plots', action='store_true',
                      help='Save each histogram as an individual file (default: False)')
    parser.add_argument('--dpi', type=int, default=300,
                      help='Resolution in dots per inch for output files (default: 300)')
    parser.add_argument('--error-range', type=float, default=0.6,
                      help='Range for error histograms [-value, +value] (default: 0.6)')
    parser.add_argument('--num-timepoints', type=int, default=-1,
                      help='Number of time points to include per variable for histograms. -1 uses all time points (default: -1)')
    parser.add_argument('--test', action='store_true', dest='test_mode',
                      help='Enable test mode to plot only first few variables (default: False)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Open dataset and perform analysis
    try:
        with h5py.File(filepaths.filepath_from_local_or_ml_artifacts(args.dataset_path), 'r') as f:
            output_path = Path(args.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            generate_error_histograms(
                f, args.context, output_path,
                file_type=args.file_type,
                bins=args.bins,
                individual_plots=args.individual_plots,
                dpi=args.dpi,
                test_mode=args.test_mode,
                error_range=args.error_range,
                num_timepoints=args.num_timepoints,
            )
            
        print(f"Analysis complete. Histograms saved to {args.output_folder}")
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise