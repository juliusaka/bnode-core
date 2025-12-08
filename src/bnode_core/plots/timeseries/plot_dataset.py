"""
Dataset plotting utilities for neural ODE analysis.

This module provides functions to plot and analyze predictions vs truth from HDF5 datasets,
with support for different variable types (states, controls, outputs, parameters) and 
error analysis capabilities.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import sys
import filepaths
from common import save_figure


def calculate_scaling_dict(hdf5_file: h5py.File, context_key: str = 'train') -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculate min/max scaling values for all variable types from the specified context.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to use for scaling calculation (default: 'train')
        
    Returns:
        Dictionary with structure: {variable_type: {variable_name: (min_val, max_val)}}
    """
    scaling_dict = {}
    variable_types = ['states', 'controls', 'outputs', 'parameters']
    
    for var_type in variable_types:
        if var_type in hdf5_file[context_key]:
            scaling_dict[var_type] = {}
            data = hdf5_file[context_key][var_type][:]
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
            
            for i, var_name in enumerate(var_names):
                if var_type == 'parameters':
                    # Parameters are 2D: (n_samples, n_params)
                    var_data = data[:, i]
                else:
                    # Time series data is 3D: (n_samples, n_variables, n_timesteps)
                    var_data = data[:, i, :]
                
                min_val = np.min(var_data)
                max_val = np.max(var_data)
                scaling_dict[var_type][var_name] = (min_val, max_val)
    
    return scaling_dict


def calculate_normalization_stats(hdf5_file: h5py.File, context_key: str = 'train') -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculate mean and standard deviation for normalization from the specified context.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to use for normalization calculation (default: 'train')
        
    Returns:
        Dictionary with structure: {variable_type: {variable_name: (mean, std)}}
    """
    normalization_stats = {}
    variable_types = ['states', 'controls', 'outputs', 'parameters']
    
    for var_type in variable_types:
        if var_type in hdf5_file[context_key]:
            normalization_stats[var_type] = {}
            data = hdf5_file[context_key][var_type][:]
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
            
            for i, var_name in enumerate(var_names):
                # Time series data is 3D: (n_samples, n_variables, n_timesteps)
                var_data = data[:, i, :].flatten() if var_type != 'parameters' else data[:, i].flatten()
                mean_val = np.mean(var_data)
                std_val = np.std(var_data)
                # Avoid division by zero
                if std_val == 0:
                    std_val = 1.0
                    logging.warning(f"Standard deviation for {var_type}/{var_name} is zero. Setting std to 1.0 to avoid division by zero.")
                normalization_stats[var_type][var_name] = (mean_val, std_val)
    
    return normalization_stats


def plot_single_axis(ax: plt.Axes, 
                    time: np.ndarray,
                    truth_data: Optional[np.ndarray],
                    prediction_data: Optional[np.ndarray],
                    variable_type: str,
                    variable_names: List[str],
                    scaling_dict: Dict[str, Dict[str, Tuple[float, float]]],
                    parameter_values: Optional[np.ndarray] = None,
                    normalization_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None) -> None:
    """
    Fill a single axis with appropriate plot based on variable type.
    
    Args:
        ax: Matplotlib axis object to plot on
        time: Time array for x-axis
        truth_data: Ground truth data (can be None for parameters)
        prediction_data: Predicted data (can be None for parameters or truth-only plots)
        variable_type: Type of variables ('states', 'controls', 'outputs', 'parameters')
        variable_names: List of variable names to plot
        scaling_dict: Scaling dictionary for y-axis limits
        parameter_values: Parameter values (only used for parameter plots)
    """
    
    if variable_type == 'parameters':
        # Bar plot for parameters
        if parameter_values is None:
            raise ValueError("parameter_values must be provided for parameter plots")
        
        # Scale parameters to [0,1] range
        scaled_values = np.zeros_like(parameter_values)
        original_values = parameter_values.copy()
        
        # Scale each parameter using its min/max from the scaling_dict
        if variable_type in scaling_dict and len(variable_names) > 0:
            for i, name in enumerate(variable_names):
                if name in scaling_dict[variable_type]:
                    min_val, max_val = scaling_dict[variable_type][name]
                    # Handle case when min equals max
                    if max_val == min_val:
                        scaled_values[i] = 0.5
                    else:
                        scaled_values[i] = (parameter_values[i] - min_val) / (max_val - min_val)
        
        x_pos = np.arange(len(variable_names))
        bars = ax.bar(x_pos, scaled_values, alpha=0.7, color='skyblue')
        ax.set_xticks(x_pos)
        
        # Set parameter names as x-tick labels with proper rotation
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        
        # Ensure enough bottom margin for labels
        plt.setp(ax.get_xticklabels(), va='top')
        
        ax.set_ylabel('Parameter Value (scaled [0,1])')
        ax.set_title('Parameters')
        
        # Add value labels on bars showing original values
        for bar, scaled_val, orig_val in zip(bars, scaled_values, original_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{orig_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Set y-limits to [0,1] with a small margin
        ax.set_ylim(-0.05, 1.05)
    
    elif variable_type == 'controls':
        # Stairs plot for controls
        for i, var_name in enumerate(variable_names):
            if truth_data is not None:
                linestyle = '--' if prediction_data is not None else '-'
                ax.step(time, truth_data[i], where='post', label=f'{var_name} (truth)', 
                       linestyle=linestyle, alpha=0.8)
            if prediction_data is not None:
                ax.step(time, prediction_data[i], where='post', label=f'{var_name} (pred)', 
                       alpha=0.8)
                
                # If there's only one variable with both truth and prediction data,
                # calculate and display the VRMSE
                if len(variable_names) == 1 and truth_data is not None:
                    var_name = variable_names[0]
                    
                    # Use normalized RMSE if normalization stats are available
                    if (normalization_stats is not None and 
                        variable_type in normalization_stats and 
                        var_name in normalization_stats[variable_type]):
                        
                        # Get mean and std for normalization
                        _, std_val = normalization_stats[variable_type][var_name]
                        
                        # Calculate normalized RMSE
                        normalized_error = (truth_data[i] - prediction_data[i]) / std_val
                        mse = np.mean(normalized_error**2)
                        rmse = np.sqrt(mse)
                        vrmse_text = f'Normalized VRMSE: {rmse:.4f}'
                    else:
                        # Calculate regular RMSE if normalization stats not available
                        mse = np.mean((truth_data[i] - prediction_data[i])**2)
                        rmse = np.sqrt(mse)
                        vrmse_text = f'VRMSE: {rmse:.4f}'
                    
                    # Add RMSE text to the plot in the top right corner
                    ax.text(0.98, 0.95, vrmse_text, 
                           transform=ax.transAxes,
                           horizontalalignment='right',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_ylabel('Control Values')
        ax.set_title('Controls')
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
        
    else:  # states or outputs
        # Line plots for states and outputs
        colors = plt.cm.tab10(np.linspace(0, 1, len(variable_names)))
        
        for i, (var_name, color) in enumerate(zip(variable_names, colors)):
            if truth_data is not None:
                ax.plot(time, truth_data[i], label=f'{var_name} (truth)', 
                       color=color, linestyle='--', alpha=0.8)
            if prediction_data is not None:
                ax.plot(time, prediction_data[i], label=f'{var_name} (pred)', 
                       color=color, alpha=0.8)
                
                # If there's only one variable with both truth and prediction data,
                # calculate and display the VRMSE
                if len(variable_names) == 1 and truth_data is not None:
                    # Use normalized RMSE if normalization stats are available
                    if (normalization_stats is not None and 
                        variable_type in normalization_stats and 
                        var_name in normalization_stats[variable_type]):
                        
                        # Get mean and std for normalization
                        _, std_val = normalization_stats[variable_type][var_name]
                        
                        # Calculate normalized RMSE
                        normalized_error = (truth_data[i] - prediction_data[i]) / std_val
                        mse = np.mean(normalized_error**2)
                        rmse = np.sqrt(mse)
                        vrmse_text = f'Normalized VRMSE: {rmse:.4f}'
                    else:
                        # Calculate regular RMSE if normalization stats not available
                        mse = np.mean((truth_data[i] - prediction_data[i])**2)
                        rmse = np.sqrt(mse)
                        vrmse_text = f'VRMSE: {rmse:.4f}'
                    
                    # Add RMSE text to the plot in the top right corner
                    ax.text(0.98, 0.95, vrmse_text, 
                           transform=ax.transAxes,
                           horizontalalignment='right',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_ylabel(f'{variable_type.capitalize()} Values') 
        # ax.set_title(f'{variable_type.capitalize()}')
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
    
    # Set y-limits based on scaling dict for time series data
    if variable_type != 'parameters' and variable_type in scaling_dict and len(variable_names) > 0:
        all_mins = [scaling_dict[variable_type][name][0] for name in variable_names 
                   if name in scaling_dict[variable_type]]
        all_maxs = [scaling_dict[variable_type][name][1] for name in variable_names 
                   if name in scaling_dict[variable_type]]
        if all_mins and all_maxs:
            y_margin = (max(all_maxs) - min(all_mins)) * 0.1
            ax.set_ylim(min(all_mins) - y_margin, max(all_maxs) + y_margin)


def create_multi_axis_figure(hdf5_file: h5py.File,
                            scaling_dict: Dict[str, Dict[str, Tuple[float, float]]],
                            context_key: str,
                            sample_idx: int,
                            plot_specifications: List[Dict[str, Union[str, List[str], List[int]]]],
                            description_text: str = "",
                            figsize: Tuple[int, int] = (8, 5),
                            normalization_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a multi-axis figure with synchronized x-axis for time series plots.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        scaling_dict: Scaling dictionary for y-axis limits
        context_key: Context to plot from ('train', 'test', etc.)
        sample_idx: Sample index to plot
        plot_specifications: List of dictionaries with keys:
            - 'variable_type': str ('states', 'controls', 'outputs', 'parameters')
            - 'variable_names': List[str] (names of variables to plot)
            - 'variable_indices': List[int] (indices of variables to plot, alternative to names)
        description_text: Additional text to display (e.g., "highest error", "median error")
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, list of axes)
    """
    
    # Get time array
    time = hdf5_file['time'][:]
    if np.max(time) > 10e3:
        time = time / 3600
        time_label = 'Time [h]'
    else:
        time_label = 'Time [s]'
    
    # Identify which plots are parameters (to not share x-axis)
    param_indices = [i for i, spec in enumerate(plot_specifications) if spec['variable_type'] == 'parameters']
    non_param_indices = [i for i, spec in enumerate(plot_specifications) if spec['variable_type'] != 'parameters']
    
    # Create figure and axes
    n_plots = len(plot_specifications)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=False)  # Don't share x by default
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    
    # Share x-axis only among time series plots (non-parameter plots)
    if len(non_param_indices) > 1:
        for i in non_param_indices[1:]:
            axes[i].sharex(axes[non_param_indices[0]])
    
    # Plot each specification
    for i, spec in enumerate(plot_specifications):
        variable_type = spec['variable_type']
        
        # Get variable names and indices
        if 'variable_indices' in spec and 'variable_names' in spec:
            # Both provided - use directly (most efficient)
            variable_indices = spec['variable_indices']
            variable_names = spec['variable_names']
        elif 'variable_names' in spec:
            # Names provided - get indices from names
            variable_names = spec['variable_names']
            all_var_names = np.array(hdf5_file[f'{variable_type}_names'][:], dtype='str')
            variable_indices = [np.where(all_var_names == name)[0][0] for name in variable_names]
        else:
            # Only indices provided - get names from indices
            variable_indices = spec['variable_indices']
            all_var_names = np.array(hdf5_file[f'{variable_type}_names'][:], dtype='str')
            variable_names = [all_var_names[idx] for idx in variable_indices]
        
        # Get data
        truth_data = None
        prediction_data = None
        parameter_values = None
        
        if variable_type == 'parameters':
            if f'{context_key}' in hdf5_file and variable_type in hdf5_file[context_key]:
                parameter_values = hdf5_file[context_key][variable_type][sample_idx, variable_indices]
        else:
            if f'{context_key}' in hdf5_file and variable_type in hdf5_file[context_key]:
                truth_data = hdf5_file[context_key][variable_type][sample_idx, variable_indices, :]
            
            if f'{context_key}' in hdf5_file and f'{variable_type}_hat' in hdf5_file[context_key]:
                prediction_data = hdf5_file[context_key][f'{variable_type}_hat'][sample_idx, variable_indices, :]
        
        # Plot on axis
        plot_single_axis(axes[i], time, truth_data, prediction_data, variable_type, 
                        variable_names, scaling_dict, parameter_values, normalization_stats)
                        
        # For parameter plots, we need to make sure they're not affected by the time axis settings
        if variable_type == 'parameters':
            # Remove x-label for parameter plots since they don't have a time dimension
            axes[i].set_xlabel('')
    
    # Set x-label only on bottom plot (except for parameters)
    axes[-1].set_xlabel(time_label)

    # Set x-limits based on time data for non-parameter plots
    for i, spec in enumerate(plot_specifications):
        if spec['variable_type'] != 'parameters':
            # Apply time scaling only for non-parameter plots
            axes[i].set_xlim(np.min(time), np.max(time))
    
    # Add title with context and sample information
    title_text = f'Context: {context_key}, Sample: {sample_idx}'
    if description_text:
        title_text += f' - {description_text}'
    
    fig.suptitle(title_text, fontsize=12, y=0.9999, va='top')
    
    # Adjust layout
    plt.tight_layout()
    # plt.subplots_adjust(top=0.92)
    
    return fig, axes


def calculate_variable_errors(hdf5_file: h5py.File,
                            context_key: str,
                            normalization_stats: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, Tuple[List[str], List[int], np.ndarray]]:
    """
    Calculate normalized errors for each variable in the specified context.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to calculate errors for
        normalization_stats: Normalization statistics (mean, std) for each variable
        
    Returns:
        Dictionary with structure: {variable_type: (variable_names, variable_indices, error_array)}
        where error_array has shape (n_variables, n_samples) containing the normalized RMSE for each variable and sample
    """
    variable_errors = {}
    variable_types = ['states', 'controls', 'outputs']
    
    for var_type in variable_types:
        if (var_type in hdf5_file[context_key] and 
            f'{var_type}_hat' in hdf5_file[context_key] and
            var_type in normalization_stats):
            
            truth_data = hdf5_file[context_key][var_type][:]  # (n_samples, n_variables, n_timesteps)
            pred_data = hdf5_file[context_key][f'{var_type}_hat'][:]  # (n_samples, n_variables, n_timesteps)
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
            
            # Collect valid variables (those with normalization stats)
            valid_indices = []
            valid_names = []
            valid_stds = []
            
            for i, var_name in enumerate(var_names):
                if var_name in normalization_stats[var_type]:
                    valid_indices.append(i)
                    valid_names.append(var_name)
                    _, std_val = normalization_stats[var_type][var_name]
                    valid_stds.append(std_val)
            
            if valid_indices:
                # Extract data for valid variables only
                truth_valid = truth_data[:, valid_indices, :]  # (n_samples, n_valid_vars, n_timesteps)
                pred_valid = pred_data[:, valid_indices, :]    # (n_samples, n_valid_vars, n_timesteps)
                valid_stds = np.array(valid_stds).reshape(1, -1, 1)  # (1, n_valid_vars, 1) for broadcasting
                
                # Vectorized calculation of normalized errors
                normalized_errors = (truth_valid - pred_valid) / valid_stds  # (n_samples, n_valid_vars, n_timesteps)
                
                # Calculate RMSE across time for each variable and sample
                rmse_per_variable_sample = np.sqrt(np.mean(normalized_errors**2, axis=2))  # (n_samples, n_valid_vars)
                
                # Transpose to get (n_valid_vars, n_samples) for easier indexing by variable
                rmse_per_variable_sample = rmse_per_variable_sample.T  # (n_valid_vars, n_samples)
                
                variable_errors[var_type] = (valid_names, valid_indices, rmse_per_variable_sample)
    
    return variable_errors


def plot_variable_wise_analysis(hdf5_file: h5py.File,
                            context_key: str,
                            output_folder: Path,
                            variable_errors: Dict[str, Tuple[List[str], List[int], np.ndarray]],
                            scaling_dict: Dict[str, Dict[str, Tuple[float, float]]],
                            normalization_stats: Dict[str, Dict[str, Tuple[float, float]]],
                            figsize: Tuple[int, int] = (8, 5),
                            n_worst: int = 1,
                            file_type: Optional[str] = None,
                            one_file_per_variable: bool = True) -> None:
    """
    Plot worst and median predictions for each variable individually.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to analyze ('test', 'validation', etc.)
        output_folder: Path to save the plots
        variable_errors: Dictionary with variable errors from calculate_variable_errors
        scaling_dict: Scaling dictionary for y-axis limits
        normalization_stats: Normalization statistics for each variable
        figsize: Figure size tuple (width, height)
        n_worst: Number of worst trajectories to plot (default: 1)
        file_type: File type to save plots as (e.g., 'png', 'pdf', None for default)
        one_file_per_variable: If True, save each variable to its own file (default: True)
    """
    # Default file type for variable_wise mode is PNG
    if file_type is None:
        file_type = 'png'
    # Analyze each variable separately
    for var_type in variable_errors:
        var_names, var_indices, error_matrix = variable_errors[var_type]
        
        for i, (var_name, var_idx) in enumerate(zip(var_names, var_indices)):
            errors = error_matrix[i, :]  # Extract errors for this variable across all samples
            
            # Find worst and median samples for this variable
            worst_indices = np.argsort(errors)[-n_worst:][::-1]  # Get n_worst indices in descending order
            median_idx = np.argsort(errors)[len(errors)//2]
            
            # Create plot specifications using variable index directly
            plot_specs = [{
                'variable_type': var_type,
                'variable_indices': [var_idx],
                'variable_names': [var_name]  # Add variable name for reference
            }]
            
            # Plot each of the worst predictions
            for rank, worst_idx in enumerate(worst_indices):
                fig, _ = create_multi_axis_figure(
                    hdf5_file, scaling_dict, context_key, worst_idx, 
                    plot_specs, f"{'Worst' if n_worst == 1 else f'#{rank+1} Worst'} prediction (VRMSE: {errors[worst_idx]:.4f})", figsize,
                    normalization_stats
                )
                
                filename_base = f"{var_type}_{var_name.replace('/', '_')}_{context_key}_worst"
                if n_worst > 1:
                    filename_base += f"_{rank+1}"
                
                save_figure(fig, output_folder, filename_base, file_type)
                
                logging.info(f"Plotted {var_type}/{var_name}: {'worst' if n_worst == 1 else f'#{rank+1} worst'} (sample {worst_idx})")
            
            # Plot median prediction
            fig, _ = create_multi_axis_figure(
                hdf5_file, scaling_dict, context_key, median_idx, 
                plot_specs, f"Median prediction (VRMSE: {errors[median_idx]:.4f})", figsize,
                normalization_stats
            )
            
            filename_base = f"{var_type}_{var_name.replace('/', '_')}_{context_key}_median"
            save_figure(fig, output_folder, filename_base, file_type)
            
            logging.info(f"Plotted {var_type}/{var_name}: median (sample {median_idx})")


def plot_sample_wise_analysis(hdf5_file: h5py.File,
                             context_key: str,
                             output_folder: Path,
                             variable_errors: Dict[str, Tuple[List[str], List[int], np.ndarray]],
                             scaling_dict: Dict[str, Dict[str, Tuple[float, float]]],
                             normalization_stats: Dict[str, Dict[str, Tuple[float, float]]],
                             figsize: Tuple[int, int] = (8, 5),
                             n_worst: int = 1,
                             file_type: Optional[str] = None) -> None:
    """
    Plot worst and median predictions across all variables for entire samples.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to analyze ('test', 'validation', etc.)
        output_folder: Path to save the plots
        variable_errors: Dictionary with variable errors from calculate_variable_errors
        scaling_dict: Scaling dictionary for y-axis limits
        normalization_stats: Normalization statistics for each variable
        figsize: Figure size tuple (width, height)
        n_worst: Number of worst trajectories to plot (default: 1)
        file_type: File type to save plots as (e.g., 'png', 'pdf', None for default)
    """
    # Default file type for sample_wise mode is PDF
    if file_type is None:
        file_type = 'pdf'
    # Calculate combined error for each sample across all variables
    n_samples = None
    combined_errors = None
    
    # Collect all variables and their specifications (including all available variable types)
    all_plot_specs = []
    
    # Collect all individual variable errors to treat them equally
    all_error_matrices = []
    
    # First add variables with error calculations
    for var_type in variable_errors:
        var_names, var_indices, error_matrix = variable_errors[var_type]
        
        if var_names:  # Check if we have any valid variables
            # Add each variable as a separate plot specification
            for i, (var_name, var_idx) in enumerate(zip(var_names, var_indices)):
                all_plot_specs.append({
                    'variable_type': var_type,
                    'variable_indices': [var_idx],  # Single variable per plot
                    'variable_names': [var_name]    # Single variable name
                })
            
            # Collect this variable type's error matrix
            all_error_matrices.append(error_matrix)  # error_matrix shape: (n_variables, n_samples)
    
    # Add all other available variable types (controls, parameters) even if they don't have error calculations
    all_variable_types = ['states', 'controls', 'outputs', 'parameters']
    for var_type in all_variable_types:
        if (var_type in hdf5_file[context_key] and 
            var_type not in variable_errors):  # Only add if not already included above
            
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
            
            if var_type == 'parameters':
                # Add all parameters as a single plot specification
                all_indices = list(range(len(var_names)))
                all_plot_specs.append({
                    'variable_type': var_type,
                    'variable_indices': all_indices,    # All parameter indices
                    'variable_names': var_names.tolist() # All parameter names
                })
            else:
                # Add each variable as a separate plot specification for other types
                for i, var_name in enumerate(var_names):
                    all_plot_specs.append({
                        'variable_type': var_type,
                        'variable_indices': [i],     # Use actual index
                        'variable_names': [var_name] # Single variable name
                    })
    
    if all_error_matrices:
        # Concatenate all error matrices along the variable axis to treat all variables equally
        combined_error_matrix = np.concatenate(all_error_matrices, axis=0)  # Shape: (total_n_variables, n_samples)
        # Take mean across all individual variables (not by variable type)
        combined_errors = np.mean(combined_error_matrix, axis=0)  # Shape: (n_samples,)
        n_samples = len(combined_errors)
        
        # Find worst and median samples overall based on errors
        worst_indices = np.argsort(combined_errors)[-n_worst:][::-1]  # Get n_worst indices in descending order
        median_idx = np.argsort(combined_errors)[len(combined_errors)//2]
        
    else:
        logging.warning("No variables available for sample-wise analysis")
        return
    
    if all_plot_specs:
        figsize = (10, 2 * len(all_plot_specs))  # Adjust height based on number of plots
        
        # Plot worst samples
        for rank, worst_idx in enumerate(worst_indices):
            error_description = f"{'Worst' if n_worst == 1 else f'#{rank+1} Worst'} overall prediction (combined VRMSE: {combined_errors[worst_idx]:.4f})"
            
            fig, _ = create_multi_axis_figure(
                hdf5_file, scaling_dict, context_key, worst_idx, 
                all_plot_specs, error_description, figsize,
                normalization_stats
            )
            
            filename_base = f"all_variables_{context_key}_worst"
            if n_worst > 1:
                filename_base += f"_{rank+1}"
            
            save_figure(fig, output_folder, filename_base, file_type)
            
            logging.info(f"Plotted sample-wise analysis: {'worst' if n_worst == 1 else f'#{rank+1} worst'} (sample {worst_idx})")
        
        # Plot median sample
        error_description_median = f"Median overall prediction (combined VRMSE: {combined_errors[median_idx]:.4f})"
        
        fig, _ = create_multi_axis_figure(
            hdf5_file, scaling_dict, context_key, median_idx, 
            all_plot_specs, error_description_median, figsize,
            normalization_stats
        )
        
        filename_base = f"all_variables_{context_key}_median"
        save_figure(fig, output_folder, filename_base, file_type)
        
        logging.info(f"Plotted sample-wise analysis: median (sample {median_idx})")


def plot_worst_and_median_predictions(hdf5_file: h5py.File,
                                    context_key: str,
                                    output_folder: Path,
                                    analysis_type: str = 'variable_wise',
                                    figsize: Tuple[int, int] = (8, 5),
                                    n_worst: int = 1,
                                    file_type: Optional[str] = None) -> None:
    """
    Plot worst and median predictions for all variables in the specified context.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to analyze ('test', 'validation', etc.)
        output_folder: Path to save the plots
        analysis_type: 'variable_wise' for per-variable analysis or 'sample_wise' for per-sample analysis
        figsize: Figure size tuple (width, height)
        n_worst: Number of worst trajectories to plot (default: 1)
        file_type: File type to save plots as (e.g., 'png', 'pdf', None for default)
    """
    
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Calculate scaling and normalization
    scaling_dict = calculate_scaling_dict(hdf5_file, 'train')
    normalization_stats = calculate_normalization_stats(hdf5_file, 'train')
    
    # Calculate errors
    variable_errors = calculate_variable_errors(hdf5_file, context_key, normalization_stats)
    
    if analysis_type == 'variable_wise':
        plot_variable_wise_analysis(
            hdf5_file, context_key, output_folder, 
            variable_errors, scaling_dict, normalization_stats, figsize,
            n_worst=n_worst, file_type=file_type
        )
    elif analysis_type == 'sample_wise':
        plot_sample_wise_analysis(
            hdf5_file, context_key, output_folder,
            variable_errors, scaling_dict, normalization_stats, figsize,
            n_worst=n_worst, file_type=file_type
        )
    else:
        raise ValueError("analysis_type must be 'variable_wise' or 'sample_wise'")

# Example usage functions
def plot_single_sample_all_variables(hdf5_file: h5py.File,
                                   context_key: str,
                                   sample_idx: int,
                                   output_folder: Optional[Path] = None,
                                   description: str = "",
                                   figsize: Tuple[int, int] = None,
                                   file_type: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot all available variables for a single sample.
    
    Args:
        hdf5_file: Open HDF5 file containing the dataset
        context_key: Context to plot from
        sample_idx: Sample index to plot
        output_folder: Optional path to save the plot
        description: Description text for the plot
        figsize: Figure size tuple (width, height), if None automatically determined
        file_type: File type to save plots as (e.g., 'png', 'pdf', None for default)
        
    Returns:
        Tuple of (figure, list of axes)
    """
    scaling_dict = calculate_scaling_dict(hdf5_file, 'train')
    normalization_stats = calculate_normalization_stats(hdf5_file, 'train')
    
    # Create plot specifications for all available variable types
    plot_specs = []
    variable_types = ['states', 'controls', 'outputs', 'parameters']
    
    for var_type in variable_types:
        if var_type in hdf5_file[context_key]:
            var_names = np.array(hdf5_file[f'{var_type}_names'][:], dtype='str')
            if var_type == 'parameters':
                # Add all parameters in a single plot
                plot_specs.append({
                    'variable_type': var_type,
                    'variable_names': var_names.tolist()
                })
            else:
                # For other types, add one plot per variable
                for var_name in var_names:
                    plot_specs.append({
                        'variable_type': var_type,
                        'variable_names': [var_name]
                    })
    
    # Calculate appropriate figsize if not provided
    if figsize is None:
        figsize = (10, 2 * len(plot_specs))  # Adjust height based on number of plots
    
    fig, axes = create_multi_axis_figure(
        hdf5_file, scaling_dict, context_key, sample_idx, plot_specs, description,
        figsize=figsize,
        normalization_stats=normalization_stats
    )
    
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        filename_base = f"{context_key}_sample_{sample_idx}_all_variables"
        
        # If file_type is not provided, save both PDF and PNG
        if file_type is None:
            file_type = 'pdf'
        
        save_figure(fig, output_folder, filename_base, file_type)
        
        logging.info(f"Plotted sample {sample_idx} from {context_key}")
    
    return fig, axes


def progress_string(progress: float, length: int = 20) -> str:
    """
    Returns a visual progress string of the form '|||||.....' for a given progress value in [0, 1].
    Args:
        progress (float): Progress value between 0 and 1.
        length (int): Total length of the progress string.
    Returns:
        str: Progress bar string.
    """
    progress = max(0, min(1, progress))  # Clamp to [0, 1]
    n_complete = int(round(progress * length))
    n_remaining = length - n_complete
    return '|' * n_complete + '.' * n_remaining


if __name__ == "__main__":
    # Example usage
    import argparse

    # sys.argv += ['--dataset_path', 'mlflow-artifacts:/689985610175568372/5b7ce6ec47694a5f9661e49ee1be98d0/artifacts/dataset.hdf5',
    #              '--context', 'common_test',
    #              '--output_folder', 'output/plots',
    #              '--mode', 'plot_worst',
    #              '--analysis_type', 'variable_wise',
    #              '--n_worst', '3']
    
    parser = argparse.ArgumentParser(description='Plot dataset analysis')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to HDF5 dataset file (local or ML artifacts)')
    parser.add_argument('--context', type=str, default='common_test',
                       help='Context to analyze (default: common_test)')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder for plots')
    parser.add_argument('--mode', type=str, default='plot_worst',
                       choices=['plot_worst', 'plot_sample_all'],
                       help='Working mode: plot_worst (worst/median), plot_sample_all (specific sample)')
    parser.add_argument('--analysis_type', type=str, default='variable_wise',
                       choices=['variable_wise', 'sample_wise'],
                       help='Type of analysis to perform for plot_worst mode')
    parser.add_argument('--n_worst', type=int, default=1,
                       help='Number of worst trajectories to plot (default: 1)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to plot for plot_sample_all mode')
    parser.add_argument('--file_type', type=str, default=None,
                        help='File type to save plots (png, pdf, etc.). If not specified, saves a default version specific to each function.')
    
    args = parser.parse_args()
    
    # Open dataset and perform analysis
    with h5py.File(filepaths.filepath_from_local_or_ml_artifacts(args.dataset_path), 'r') as f:
        output_path = Path(args.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.mode == 'plot_worst':
            plot_worst_and_median_predictions(
                f, args.context, output_path, args.analysis_type,
                n_worst=args.n_worst, file_type=args.file_type
            )
        elif args.mode == 'plot_sample_all':
            description = f"Sample {args.sample_idx} from {args.context}"
            plot_single_sample_all_variables(
                f, args.context, args.sample_idx, output_path, description,
                file_type=args.file_type
            )
        else:
            print(f"Unknown mode: {args.mode}")
    
    print(f"Analysis complete. Plots saved to {args.output_folder}")