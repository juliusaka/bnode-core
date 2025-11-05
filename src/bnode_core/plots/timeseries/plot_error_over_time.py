"""
Plot absolute errors over time for each variable, with a histogram of all-timepoint
absolute errors shown as an inset in each subplot.

This script follows the data extraction flow used in plot_error_histogram.py:
  - normalization_stats = calculate_normalization_stats(hdf5_file, 'train')
  - variable_errors = calculate_variable_errors(hdf5_file, context_key, normalization_stats)
  - variable_names = get_variable_names(hdf5_file, normalization_stats, {})
  - all_variable_names, error_vectors = collect_error_data(variable_errors, variable_names)

It then plots |error| over time (aggregated across samples) for each variable and overlays
an inset histogram of |error| across all samples and timepoints.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import argparse
import logging
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import filepaths
from plot_dataset import calculate_normalization_stats
from plot_error_histogram import (
    calculate_variable_errors,
    get_variable_names,
    collect_error_data,
)
from common import save_figure


def _draw_error_time_series(
    ax: plt.Axes,
    errors: np.ndarray,
    *,
    title: Optional[str] = None,
    max_y: float = 0.8,
) -> None:
    """Draw mean absolute error over time.

    Args:
        ax: Matplotlib axis to draw on
        errors: Array of shape (n_samples, n_timesteps) or (n_samples,) for non-time-series
        title: Optional title for the subplot
        max_y: Upper y-limit for the line plot
    """
    e = np.asarray(errors)
    if e.ndim == 1:
        # Not a time series; replicate as a single timepoint for plotting, or skip handled by caller
        time_axis = np.arange(1)
        mean_abs = np.array([np.mean(np.abs(e))])
    elif e.ndim == 2:
        # Time series: average absolute error across samples per timepoint
        time_axis = np.arange(e.shape[1])
        mean_abs = np.mean(np.abs(e), axis=0)
    else:
        # Unexpected shape; flatten time dimension if present at the end
        time_axis = np.arange(e.shape[-1])
        mean_abs = np.mean(np.abs(e), axis=tuple(range(e.ndim - 1)))

    # Plot line of mean absolute error over time
    ax.plot(time_axis, mean_abs, color='C0', linewidth=1.8)
    ax.set_ylim(0, max_y)
    ax.set_xlim(0, len(time_axis) - 1 if len(time_axis) > 0 else 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('|error|')
    if title is not None:
        ax.set_title(title, fontsize=10)
    
    # fille area of 10%, 20%, 30%...90%, 100% percentiles with same color (blue), and alpha of 0.1
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for p in percentiles:
        if e.ndim == 1:
            perc_values_lower = np.percentile(np.abs(e), 100 - p/2)
            perc_values = np.percentile(np.abs(e), p/2)
            ax.fill_between(time_axis, perc_values_lower, perc_values, color='C0', alpha=0.1)
        else:
            perc_values_lower = np.percentile(np.abs(e), 100 - p/2, axis=0)
            perc_values = np.percentile(np.abs(e), p/2, axis=0)
            ax.fill_between(time_axis, perc_values_lower, perc_values, color='C0', alpha=0.1)
    
    # add legend
    ax.legend(['Mean |error|'], loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0)

def plot_error_over_time(
    all_variable_names: List[str],
    error_vectors: List[np.ndarray],
    output_folder: Path,
    *,
    output_filename: str = "error_over_time",
    file_type: str = 'pdf',
    dpi: int = 300,
    individual_plots: bool = False,
    max_y: float = 0.6,
    test_mode: bool = False,
) -> None:
    """Create line plots of absolute error over time with inset histograms per variable.

    Args:
        all_variable_names: Names like "type: var"
        error_vectors: Per-variable arrays: (n_samples, n_timesteps) or (n_samples,)
        output_folder: Destination folder
        output_filename: Base filename for combined figure
        file_type: Output file type
        dpi: Output DPI
        individual_plots: If True, save one file per variable; else one combined figure
        max_y: Upper y-limit for line plots (default 0.8)
        test_mode: If True, limit to first N variables for speed
    """
    # Filter to time-series variables (keep 2D with time > 1). Keep 1D optionally as single-point plots
    filtered = []
    for name, e in zip(all_variable_names, error_vectors):
        arr = np.asarray(e)
        if arr.ndim == 2 and arr.shape[1] >= 1:
            filtered.append((name, arr))
        elif arr.ndim == 1:
            # Include non-time-series (parameters) as a single bar across one timepoint
            filtered.append((name, arr))
        else:
            logging.warning(f"Skipping variable {name} with unsupported error shape {arr.shape}")

    if test_mode:
        filtered = filtered[: min(6, len(filtered))]
        logging.info(f"Test mode: plotting only first {len(filtered)} variables")

    if not filtered:
        logging.warning("No variables to plot.")
        return

    names = [n for n, _ in filtered]
    errors = [e for _, e in filtered]

    n_vars = len(names)

    if individual_plots:
        for i, (name, e) in enumerate(zip(names, errors)):
            logging.info(f"Plotting {i+1}/{n_vars}: {name}")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            _draw_error_time_series(ax, e, title=name, max_y=max_y)
            fname = name.replace('/', '_').replace(':', '_').replace(' ', '_')
            save_figure(fig, Path(output_folder), f"{output_filename}_{fname}", file_type, dpi=dpi)
            plt.close(fig)
    else:
        # Combined grid
        n_cols = min(3, n_vars)
        n_rows = int(np.ceil(n_vars / n_cols))
        fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 3.0))
        gs = gridspec.GridSpec(n_rows, n_cols)
        gs.update(wspace=0.5, hspace=0.6)

        for i, (name, e) in enumerate(zip(names, errors)):
            ax = plt.subplot(gs[i // n_cols, i % n_cols])
            _draw_error_time_series(ax, e, title=name, max_y=max_y)
            if (i // n_cols) == n_rows - 1:
                ax.set_xlabel('time step')

        plt.suptitle("Absolute Error Over Time", fontsize=14, y=0.995)
        save_figure(fig, Path(output_folder), output_filename, file_type, dpi=dpi)
        plt.close(fig)


def generate_error_over_time(
    hdf5_file: h5py.File,
    context_key: str,
    output_folder: Path,
    *,
    file_type: str = 'pdf',
    dpi: int = 300,
    individual_plots: bool = False,
    max_y: float = 0.8,
    test_mode: bool = False,
) -> None:
    """End-to-end generation of error-over-time plots from an HDF5 dataset."""
    # Calculate normalization stats from training data
    normalization_stats = calculate_normalization_stats(hdf5_file, 'train')

    # Calculate errors for each variable (normalized, signed)
    variable_errors = calculate_variable_errors(hdf5_file, context_key, normalization_stats)

    # Get variable names
    variable_names: Dict[str, List[str]] = {}
    variable_names = get_variable_names(hdf5_file, normalization_stats, variable_names)

    # Reshape error data for plotting
    all_variable_names, error_vectors = collect_error_data(variable_errors, variable_names)

    # Make plots
    output_filename = f"error_over_time_{context_key}"
    if test_mode:
        output_filename += "_test_mode"

    plot_error_over_time(
        all_variable_names,
        error_vectors,
        output_folder,
        output_filename=output_filename,
        file_type=file_type,
        dpi=dpi,
        individual_plots=individual_plots,
        max_y=max_y,
        test_mode=test_mode,
    )
    logging.info(f"Error-over-time plots created for context {context_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot absolute errors over time with inset histograms")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to HDF5 dataset file (local or ML artifacts)')
    parser.add_argument('--context', type=str, default='common_test', help='Context to analyze (default: common_test)')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for plots')
    parser.add_argument('--file_type', type=str, default='png', choices=['pdf', 'png', 'jpg', 'svg'], help='Output file type (default: pdf)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI (default: 300)')
    parser.add_argument('--individual_plots', action='store_true', help='Save one plot per variable instead of a combined grid')
    parser.add_argument('--max-y', type=float, default=0.8, dest='max_y', help='Maximum y-axis value for line plots (default: 0.8)')
    parser.add_argument('--test', action='store_true', dest='test_mode', help='Enable test mode to limit the number of variables')

    # For convenience during development
    sys.argv += ['--dataset_path', 'mlflow-artifacts:/689985610175568372/5b7ce6ec47694a5f9661e49ee1be98d0/artifacts/dataset.hdf5',
                '--output_folder', 'output/error_over_time',
                # '--test',
                '--individual_plots',
                # '--file_type', 'png',
                # '--num-timepoints', '1'
                ]

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        with h5py.File(filepaths.filepath_from_local_or_ml_artifacts(args.dataset_path), 'r') as f:
            out_dir = Path(args.output_folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            generate_error_over_time(
                f,
                args.context,
                out_dir,
                file_type=args.file_type,
                dpi=args.dpi,
                individual_plots=args.individual_plots,
                max_y=args.max_y,
                test_mode=args.test_mode,
            )
        print(f"Analysis complete. Plots saved to {args.output_folder}")
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise