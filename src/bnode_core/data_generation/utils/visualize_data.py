import argparse
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import shutil
from pathlib import Path
from matplotlib import gridspec

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def plot_variable(class_name, var_name, raw_data, output_dir, n_bins, limits=None, time=None, figsize=(12, 6), verbose=False):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    sns.set(style="whitegrid")
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 3, 8], wspace=0.1, left=0.05, right=0.95, figure=fig)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    raw_data_flat = raw_data.flatten()

    # statistics
    mean = np.mean(raw_data_flat)
    std = np.std(raw_data_flat)
    min_data = np.min(raw_data_flat)
    max_data = np.max(raw_data_flat)
    perc005 = np.percentile(raw_data_flat, 0.005)
    perc995 = np.percentile(raw_data_flat, 99.5)
    data_range = max_data - min_data
    rvr = data_range / std if std != 0 else np.inf  # Range Variance Ratio

    if limits is not None:
        _limits_range = limits[1] - limits[0]
        ymin = min(min_data, limits[0] - 0.05 * _limits_range)
        ymax = max(max_data, limits[1] + 0.05 * _limits_range)
    else:
        ymin = min_data
        ymax = max_data


    # add statistics to the first subplot
    stats_text = (
        f"min: {min_data:.2e}\n"
        f"max: {max_data:.2e}\n"
        f"99%: {perc005:.2e} \n   - {perc995:.2e}\n"
        f"$\\mu$:   {mean:.2e}\n"
        f"$\\sigma$:   {std:.2e}\n"
        f"\nRVR: {rvr:.2f}\n"
    )
    # add limits if provided
    if limits is not None:
        stats_text += f"\n\nlower lim.: {limits[0]:.2e}\nupper lim.: {limits[1]:.2e}"
    stats_text += f"\n\n# samples (flat): \n    {raw_data_flat.size:.3e}"
    stats_text += f"\n# samples (raw):  \n    {raw_data.shape}"
    axs[0].text(
        0.05, 0.95,
        stats_text,
        transform=axs[0].transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        family='monospace'
    )
    note = (
        # "y-limits are the 99%-\npercentile or, if\ngreater, the provided\nlimits.\n"
        "y-limits are the \nmin/max of the data.\n" \
        "RVR: Range Variance Ratio\n" \
        "      = (max - min) / std\n" \
    )
    axs[0].text(
        0.05, 0.05,
        note,
        transform=axs[0].transAxes,
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=8,
        family='monospace',
        color='gray'
    )

    # hide the first subplot
    axs[0].axis('off')
    axs[0].set_title("Statistics")

    # histogram
    axs[2].sharey(axs[1])

    bins = np.linspace(ymin, ymax, n_bins + 1)
    axs[1].hist(raw_data_flat, bins=bins, orientation='horizontal', color='steelblue', edgecolor='black', alpha=0.7)
    axs[1].set_title('Histogram')
    axs[1].invert_xaxis()
    if limits is not None:
        axs[1].axhline(limits[0], color='tomato', linestyle='--')
        axs[1].axhline(limits[1], color='tomato', linestyle='--')
    # add mean, std, min, max lines
    line_mean = axs[1].axhline(mean, color='green', linestyle='solid', label='$\\mu$')
    line_stdplus = axs[1].axhline(mean + std, color='green', linestyle='dotted', label='$\\mu \\pm \\sigma$')
    line_stdminus = axs[1].axhline(mean - std, color='green', linestyle='dotted', label='Mean - 1 Std')
    handles=[line_mean, line_stdplus]
    # add min, max if they are not the same as limits
    if limits is not None:
        line_lim = axs[1].axhline(limits[0], color='red', linestyle='solid', label='Limits')
        handles.append(line_lim)
        axs[1].axhline(limits[1], color='red', linestyle='solid', label='Upper Limit')
    axs[1].legend(handles=handles, loc='best', fontsize='8')
    
    axs[1].set_xlabel('Count')
    axs[1].set_ylim([ymin, ymax])
    
    # Trajectories
    opt = 'density_overlay' if raw_data.ndim == 2 else 'lines'
    if time is not None:
        if opt == 'lines':
            for i in range(raw_data.shape[0]):
                axs[2].plot(time, raw_data[i,:], label=f'Trajectory {i+1}', alpha=1.0, color='steelblue', linewidth=0.5)
            axs[2].set_title('Trajectories: plotted as lines')
        elif opt == 'density_overlay':
            # Density overlay: plot a pseudo-heatmap of trajectory density
            n_bins = 1024

            ymin, ymax
            hist = np.zeros((len(time), n_bins))
            for t in range(len(time)):
                hist[t, :], _ = np.histogram(raw_data[:, t], bins=n_bins, range=(ymin, ymax))
            
            hist = hist.T  # Transpose to have time on x-axis and variable on y-axis

            cmap = matplotlib.colormaps['inferno']
            hist = np.sqrt(hist)  # Square root for better visibility
            v_min = np.min(hist)
            v_max = np.percentile(hist, 99.9)

            axs[2].imshow(hist, aspect='auto', origin='lower',
                          extent=[np.min(time), np.max(time), ymin, ymax],
                          cmap=cmap, vmin=v_min, vmax=v_max)
            axs[2].set_title('Trajectories: pseudo-heatmap of density, sqrt-transformed')

        axs[2].set_xlabel('time')
        axs[2].set_xlim([np.min(time), np.max(time)])
    else:
        axs[2].axis('off')
    

    fig.suptitle(f"{class_name.capitalize()} - {var_name}")
    # fig.tight_layout()
    if verbose:
        plt.show()
    path = output_dir / f"{class_name}_{var_name}.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"\t\tSaved plot to {path}")
    plt.close()

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Plot HDF5 variables with histograms and trajectories.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset file.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file. If not provided, will look for 'config.yaml' in the dataset directory.")
    parser.add_argument("--figsize", type=str, default="15,6", help="Figure size, e.g., '12,6'")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots. Defaults to the dataset directory.")
    parser.add_argument("--print_limits", type=str, default="controls,parameters", help="Comma-separated list of classes to print limits for. Defaults to 'controls,parameters'.")
    parser.add_argument("--bins", type=int, default=80, help="Number of bins for histograms. Defaults to 20.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()
    
    # Parse arguments
    dataset_path = args.dataset
    dataset_path = Path(dataset_path).resolve()
    config_path = args.config
    figsize = args.figsize
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_path.parent / "visualization"
    print_limits = args.print_limits.split(',')
    assert "outputs" not in print_limits, "Outputs are not supported for limits."
    n_bins = args.bins
    verbose = args.verbose
    
    # modify arguments
    figsize = tuple(map(int, figsize.split(',')))
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Removing it.")
        shutil.rmtree(output_dir)
    print(f"Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load dataset
    f = h5py.File(dataset_path, "r")
    time_axis = f["time"][:]

    # determine if this is a raw dataset or a processed one
    raw_dataset = False if "train" in f.keys() else True
    if raw_dataset:
        print("Detected raw dataset.")
        data = f
        completed = f["logs"]["completed"][:]
    else:
        data = f["train"]

    # load config
    if config_path is None:
        dataset_name = dataset_path.stem
        if raw_dataset:
            config_filename = dataset_name.replace('raw_data', 'RawData_config.yaml')
            pass
        else:
            config_filename = dataset_name.replace('_dataset', '_pModel_config.yaml')
        config_path = dataset_path.parent / config_filename
    else:
        config_path = Path(config_path).resolve()
    config = load_yaml(config_path)

    for class_name in ["controls", "outputs", "states", "parameters"]:
        if class_name in data.keys():
            print(f"Processing {class_name}...")
            var_data = data[class_name]
            if raw_dataset:
                var_data = var_data[completed==1]
            var_names = [name.decode() for name in f[f"{class_name}_names"]]

            for i, name in enumerate(var_names):
                print(f"  Plotting {name}...")
                raw_data = var_data[:, i]

                # Limits from config
                if class_name in print_limits:
                    if raw_dataset:
                        limits = config[class_name][name]
                    else:
                        limits = config['RawData'][class_name][name]
                    limits = [float(limits[0]), float(limits[1])]
                else:
                    limits = None

                # Time plot only for non-parameters
                if class_name == "parameters":
                    time = None
                else:
                    time = time_axis

                plot_variable(class_name, name, raw_data, output_dir, n_bins, limits=limits, time=time, figsize=figsize, verbose=verbose)

if __name__ == "__main__":
    main()
