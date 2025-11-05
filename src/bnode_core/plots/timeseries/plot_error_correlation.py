"""
Error correlation visualization for neural ODE datasets.

This module provides functions to analyze and visualize the error correlations
between different variables in a dataset, showing how errors in one variable 
may correlate with errors in others.
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
from matplotlib.colors import LinearSegmentedColormap

# Import useful functions from plot_dataset
from plot_dataset import (
    calculate_scaling_dict,
    calculate_normalization_stats,
)
from common import save_figure
from plot_error_histogram import (
    calculate_variable_errors,
    get_variable_names,
    collect_error_data,
)

def plot_error_correlation_scatter(all_variable_names: List[str], 
                                error_vectors: List[np.ndarray],
                                value_vectors: List[np.ndarray],
                                output_folder: Path,
                                output_filename: str = "error_correlation_scatter",
                                file_type: str = 'pdf',
                                individual_plots: bool = False,
                                rasterize_scatter: bool = True,
                                dpi: int = 300,
                                test_mode: bool = False) -> None:
    """
    Create and save scatter plots showing error correlations between variables.
    Diagonal plots show normalized variable values vs. normalized errors.
    
    Args:
        all_variable_names: List of variable names (with type prefix)
        error_vectors: List of error vectors for each variable
        value_vectors: List of normalized value vectors for each variable
        output_folder: Path to save the output
        output_filename: Base filename for the output
        file_type: File type to save plots as (e.g., 'png', 'pdf')
        individual_plots: If True, save each correlation subplot as individual file
        rasterize_scatter: If True, scatter plots will be rasterized when saving to vector formats (PDF, SVG)
                           to reduce file size and improve performance
        dpi: Resolution in dots per inch for rasterized components (default: 300)
        test_mode: If True, only plots the first two rows of variables for quick testing (default: False)
    """
    # If test mode is enabled, only use the first few variables
    if test_mode:
        max_test_vars = min(2, len(all_variable_names))  # At most 2 variables in test mode
        all_variable_names = all_variable_names[:max_test_vars]
        error_vectors = error_vectors[:max_test_vars]
        value_vectors = value_vectors[:max_test_vars]
    
    n_variables = len(all_variable_names)
    
    # Create figure
    if individual_plots:
        # For individual plots we'll create multiple figures
        pass
    else:
        # Create one large figure with all subplots
        # Make the figure larger
        fig_size = n_variables * 3.0
        fig = plt.figure(figsize=(fig_size, fig_size))
        gs = gridspec.GridSpec(n_variables, n_variables)
        gs.update(wspace=0.3, hspace=0.3)
    
    # Plot each scatter plot
    for i in range(n_variables):
        logging.info(f"Plotting variable {i+1}/{n_variables}: {all_variable_names[i]}")  # Log progress
        for j in range(n_variables):
            # Prepare flattened arrays for consistent scatter usage
            vi = np.asarray(value_vectors[i]).reshape(-1)
            ei = np.asarray(error_vectors[i]).reshape(-1)
            ej = np.asarray(error_vectors[j]).reshape(-1)
            
            if individual_plots:
                # Create individual figure for this pair
                plt.figure(figsize=(6, 6))
                ax = plt.gca()
            else:
                # Add subplot to the main figure
                ax = plt.subplot(gs[i, j])
            
            # For diagonal plots, show scatter of normalized values vs. normalized errors
            if i == j:
                # Show scatter plot of normalized variable values vs. normalized errors
                scatter = ax.scatter(vi, ei, s=10, alpha=0.5, color='blue', rasterized=rasterize_scatter)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)  # Add horizontal line at y=0 (no error)
                
                # Add trend line to see if there's systematic bias
                if vi.size > 1 and ei.size > 1:
                    z = np.polyfit(vi, ei, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(np.min(vi), np.max(vi), 100)
                    ax.plot(x_range, p(x_range), "r--", alpha=0.7)  # Trend lines remain vectorized
                
                # Calculate Pearson correlation coefficient
                corr_coef = np.corrcoef(vi, ei)[0, 1]
                ax.set_title(f"{all_variable_names[i]}\nr={corr_coef:.2f}", fontsize=8)
                
                # Set axis labels for diagonal plots
                ax.set_xlabel("Normalized Value", fontsize=6)
                ax.set_ylabel("Normalized Error", fontsize=6)
            else:
                # For off-diagonal, show scatter plot of errors between different variables
                scatter = ax.scatter(ej, ei, s=10, alpha=0.5, rasterized=rasterize_scatter)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)  # Add horizontal line at y=0
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)  # Add vertical line at x=0
                
                # Calculate Pearson correlation coefficient
                corr_coef = np.corrcoef(ej, ei)[0, 1]
                ax.set_title(f"r={corr_coef:.2f}", fontsize=8)
                
                # Add trend line if correlation is significant
                if abs(corr_coef) > 0.1 and ej.size > 1 and ei.size > 1:
                    z = np.polyfit(ej, ei, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(np.min(ej), np.max(ej), 100)
                    ax.plot(x_range, p(x_range), "r--", alpha=0.7)  # Trend lines remain vectorized
            
            # Set axis limits for better visibility
            if i == j:  # For diagonal plots (value vs. error)
                # For x-axis (value)
                max_val_x = max(abs(np.percentile(vi, 5)), abs(np.percentile(vi, 95)))
                # For y-axis (error)
                max_val_y = max(abs(np.percentile(ei, 5)), abs(np.percentile(ei, 95)))
                ax.set_xlim(-max_val_x*1.1 if max_val_x > 0 else -1, max_val_x*1.1 if max_val_x > 0 else 1)
                ax.set_ylim(-max_val_y*1.1 if max_val_y > 0 else -1, max_val_y*1.1 if max_val_y > 0 else 1)
            elif i != j:  # For off-diagonal plots (error vs. error)
                # Get max absolute value for symmetric limits
                max_val_x = max(abs(np.percentile(ej, 5)), abs(np.percentile(ej, 95)))
                max_val_y = max(abs(np.percentile(ei, 5)), abs(np.percentile(ei, 95)))
                max_val = max(max_val_x, max_val_y)
                if max_val == 0:
                    max_val = 1
                ax.set_xlim(-max_val*1.1, max_val*1.1)
                ax.set_ylim(-max_val*1.1, max_val*1.1)
            
            # Add variable names to left and bottom edges (only for non-diagonal plots)
            if i != j:
                if j == 0:
                    ax.set_ylabel(all_variable_names[i], fontsize=8, rotation=0, ha='right', va='center')
                if i == n_variables - 1:
                    ax.set_xlabel(all_variable_names[j], fontsize=8, rotation=45, ha='right')
            
            # Set smaller tick font size
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            if individual_plots:
                # Save individual plot
                filename_base = f"{output_filename}_{i}_{j}_{all_variable_names[i]}_{all_variable_names[j]}".replace("/", "_").replace(" ", "_")
                save_figure(plt.gcf(), output_folder, filename_base, file_type='png', dpi=300)
    
    if not individual_plots:
        logging.info("Saving combined correlation matrix figure")
        # Add title to the main figure
        plt.suptitle("Error Analysis: Normalized Values vs. Normalized Errors", fontsize=16, y=0.99)
        
        # Save the complete correlation matrix figure
        save_figure(fig, output_folder, output_filename, file_type, dpi=300)
    
    plt.close('all')


def _build_value_vectors(
    hdf5_file: h5py.File,
    context_key: str,
    normalization_stats: Dict[str, Dict[str, Tuple[float, float]]],
    all_variable_names: List[str],
    error_vectors: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute normalized value vectors aligned with the provided error vectors.
    For each "type: name" entry, fetch truth values, normalize, and flatten to 1D.
    """
    values: List[np.ndarray] = []
    cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for name, err in zip(all_variable_names, error_vectors):
        var_type, var_name = [s.strip() for s in name.split(':', 1)]
        # Cache datasets per var_type
        if var_type not in cache:
            truth = hdf5_file[context_key][var_type][:]
            var_names = list(map(str, hdf5_file[f"{var_type}_names"][:]))
            cache[var_type] = (truth, var_names)
        truth, var_names = cache[var_type]
        try:
            idx = var_names.index(var_name)
        except ValueError:
            logging.warning(f"Variable name {var_name} not found in {var_type}_names; filling zeros")
            values.append(np.zeros_like(np.asarray(err).reshape(-1)))
            continue
        mean, std = normalization_stats[var_type][var_name]
        if var_type == 'parameters':
            vals = truth[:, idx]
        else:
            vals = truth[:, idx, :]
        vals_norm = (vals - mean) / (std if std != 0 else 1.0)
        values.append(np.asarray(vals_norm).reshape(-1))
    return values


def plot_error_correlation(hdf5_file: h5py.File,
                          context_key: str,
                          output_folder: Path,
                          file_type: str = 'pdf',
                          individual_plots: bool = False,
                          rasterize_scatter: bool = True,
                          dpi: int = 300,
                          test_mode: bool = False) -> None:
    """
    Plot scatter plots showing relationships between normalized variable values and normalized errors,
    as well as error correlations between different variables.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to analyze ('test', 'validation', etc.)
        output_folder: Path to save the plots
        file_type: File type to save plots as (e.g., 'png', 'pdf')
        individual_plots: If True, save each correlation subplot as an individual file
        rasterize_scatter: If True, scatter plots will be rasterized when saving to vector formats (PDF, SVG)
                          to reduce file size and improve performance
        dpi: Resolution in dots per inch for rasterized components (default: 300)
        test_mode: If True, only plots the first two rows of variables for quick testing (default: False)
    """
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Calculate normalization stats from training data
    normalization_stats = calculate_normalization_stats(hdf5_file, 'train')
    
    # Add parameter normalization statistics if they exist
    # (since the default calculate_normalization_stats doesn't include parameters)
    if 'parameters' in hdf5_file['train']:
        if 'parameters' not in normalization_stats:
            normalization_stats['parameters'] = {}
            
        param_data = hdf5_file['train']['parameters'][:]
        param_names = np.array(hdf5_file['parameters_names'][:], dtype='str')
        
        for i, param_name in enumerate(param_names):
            # Parameters are 2D: (n_samples, n_params)
            param_values = param_data[:, i]
            mean_val = np.mean(param_values)
            std_val = np.std(param_values)
            # Avoid division by zero
            if std_val == 0:
                std_val = 1.0
                logging.warning(f"Standard deviation for parameters/{param_name} is zero. Setting std to 1.0.")
            normalization_stats['parameters'][param_name] = (mean_val, std_val)
    
    # Calculate errors and normalized values for each variable
    variable_errors = calculate_variable_errors(hdf5_file, context_key, normalization_stats)

    # Collect error and value data for scatter plots
    variable_names: Dict[str, List[str]] = {}
    variable_names = get_variable_names(hdf5_file, normalization_stats, variable_names)
    all_variable_names, error_vectors = collect_error_data(variable_errors, variable_names)
    value_vectors = _build_value_vectors(hdf5_file, context_key, normalization_stats, all_variable_names, error_vectors)

    # Plot scatter plots of error correlations and value-error relationships
    output_filename = f"error_correlation_scatter_{context_key}"
    if test_mode:
        output_filename += "_test_mode"
        logging.info("Test mode enabled: plotting only first two rows of variables")
    
    plot_error_correlation_scatter(
        all_variable_names, error_vectors, value_vectors, output_folder, 
        output_filename=output_filename, file_type=file_type,
        individual_plots=individual_plots, rasterize_scatter=rasterize_scatter,
        dpi=dpi, test_mode=test_mode
    )
    
    logging.info(f"Error correlation scatter plots created for context {context_key}")


if __name__ == "__main__":
    # Set up argument parser

    parser = argparse.ArgumentParser(description='Plot error correlation matrix')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to HDF5 dataset file (local or ML artifacts)')
    parser.add_argument('--context', type=str, default='common_test',
                      help='Context to analyze (default: test)')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Output folder for plots')
    parser.add_argument('--file_type', type=str, default='pdf',
                      choices=['pdf', 'png', 'jpg', 'svg'],
                      help='File type for saving plots (default: pdf)')
    parser.add_argument('--individual_plots', action='store_true',
                      help='Save each correlation subplot as an individual file (default: False)')
    parser.add_argument('--no-rasterize', action='store_false', dest='rasterize_scatter',
                      help='Disable rasterization of scatter plots in vector outputs (default: rasterize enabled)')
    parser.add_argument('--dpi', type=int, default=300,
                      help='Resolution in dots per inch for rasterized elements in vector formats (default: 300)')
    parser.add_argument('--test', action='store_true', dest='test_mode',
                      help='Enable test mode to plot only first two rows of variables (default: False)')
    parser.set_defaults(rasterize_scatter=True, test_mode=False)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Open dataset and perform analysis
    try:
        with h5py.File(filepaths.filepath_from_local_or_ml_artifacts(args.dataset_path), 'r') as f:
            output_path = Path(args.output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            plot_error_correlation(
                f, args.context, output_path, 
                file_type=args.file_type,
                individual_plots=args.individual_plots,
                rasterize_scatter=args.rasterize_scatter,
                dpi=args.dpi,
                test_mode=args.test_mode
            )
            
        print(f"Analysis complete. Plots saved to {args.output_folder}")
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise