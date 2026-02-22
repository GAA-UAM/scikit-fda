
import matplotlib.pyplot as plt
import numpy as np

from .diffusion_process import VarianceExplodingDiffusionProcess

from ...representation.grid import FDataGrid
import torch
from .synthetic_data import generate_sin_function, generate_lines_function

from .metrics import get_wasserstein_distance

from .diffusion_model import FDataGenerator
from . import metrics

from matplotlib.gridspec import GridSpec


def plot_combined_metrics_comparison(
        real_data,
        syn_data_list:  list,
        ylim,
        n_plots ,
        metrics_real : dict,
        metrics_list: list[dict],
        char_names: list[str],
        path,
        name: str,
        figsize=(20, 8)):
    '''
    Plots combined metrics comparison in two rows, 
    the first one shows real data and all synthetic data models,
    and the second one is equivalent to plot_metrics_comparison.

    Args:
        real_data:  The real functional data (skfda.FDataGrid).
        syn_data_list: List of synthetic functional data (list of skfda.FDataGrid).
        ylim:  Y-axis limits for the comparison plot.
        n_plots: Number of functions to plot in the first row.
        metrics_real: Dictionary containing metrics for training real data.
        char_names: List of characteristic names for labeling.
        path: Path to save the generated plots.
        figsize: Figure size tuple (width, height).
    '''
    n_models = len(syn_data_list)
    n_plots = min([n_plots, len(real_data)] + [len(syn_data) for syn_data in syn_data_list])
    if n_plots > 100:
        print("Warning:  Plotting more than 100 functions "
              "may not show any conclusions.")
    n_batches = len(metrics_list[0]['wasserstein_distance_batched'])
    n_chars = len(metrics_list[0]['wasserstein_distance_batched'][0])
    
    # Number of plots in each row
    # Row 1: n_models + 1 plots (Real Data + each Synthetic Data model)
    # Row 2: 1 + n_chars plots (Noise Metric + Wasserstein for each characteristic)
    n_plots_row1 = n_models + 1
    n_plots_row2 = 1 + n_chars
    
    # LCM-friendly:  total columns divisible by both n_plots_row1 and n_plots_row2
    total_cols = n_plots_row1 * n_plots_row2
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, total_cols, figure=fig)
    
    # ==================== ROW 1: Real Data + All Synthetic Data ====================
    cols_per_plot_row1 = total_cols // n_plots_row1
    
    # First plot: Real Data
    ax_real = fig.add_subplot(gs[0, :cols_per_plot_row1])
    if ylim is not None:
        ax_real.set_ylim(ylim)
    index_real = np.random.choice(len(real_data), n_plots, replace=False)
    real_data_subset = real_data[index_real]
    real_data_subset.plot(axes=ax_real, color='green')
    ax_real.set_title("Real Data", fontsize=12)
    
    # Remaining plots: Synthetic Data (one per model)
    axs_syn = []
    for i, syn_data in enumerate(syn_data_list):
        start_col = (i + 1) * cols_per_plot_row1
        end_col = (i + 2) * cols_per_plot_row1
        ax_syn = fig.add_subplot(gs[0, start_col: end_col], sharey=ax_real)
        
        if ylim is not None:
            ax_syn.set_ylim(ylim)
        syn_index = np.random.choice(len(syn_data), n_plots, replace=False)
        syn_data_subset = syn_data[syn_index]
        syn_data_subset.plot(axes=ax_syn, color='red')
        
        # Use model name from metrics_list if available
        model_name = metrics_list[i]['model_name'] if i < len(metrics_list) else f"Synthetic {i+1}"
        ax_syn.set_title(f"Synthetic:  {model_name}", fontsize=12)
        axs_syn.append(ax_syn)
    
    # ==================== ROW 2: Metrics Comparison ====================
    cols_per_plot_row2 = total_cols // n_plots_row2
    
    # Create axes for second row
    axs_row2 = []
    for i in range(n_plots_row2):
        ax = fig.add_subplot(gs[1, i * cols_per_plot_row2:(i + 1) * cols_per_plot_row2])
        axs_row2.append(ax)
    
    # Plot Noise Metric (first plot in row 2)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'olive', 'magenta', 'navy', 'teal']

    for i, metrics in enumerate(metrics_list): 
        axs_row2[0].bar(metrics['model_name'], metrics['nosie'], color=colors[i % len(colors)])
    axs_row2[0].set_title('Noise Metric Comparison', fontsize=14)
    axs_row2[0].set_ylabel('Noise Metric', fontsize=12)
    axs_row2[0].set_xlabel('Model', fontsize=12)
    axs_row2[0].grid(axis='y', alpha=0.3)
    
    # Plot Wasserstein Distance histograms for each characteristic
    
    for char_idx in range(n_chars):
        ax = axs_row2[1 + char_idx]
        
        # Calculate bins based on min/max across all models
        max_val = max([max(metrics['wasserstein_distance_batched'][:, char_idx]) for metrics in metrics_list])
        max_val = max(max_val, 1.0)  # Ensure max_val is at least 1 for better visualization
        bins = np.linspace(0, max_val, 50).reshape(-1)
        for i, metrics in enumerate(metrics_list):
            ax.hist(
                metrics['wasserstein_distance_batched'][:, char_idx],
                bins=bins,
                alpha=0.6,
                label=metrics['model_name'],
                color=colors[i % len(colors)],
                density=False
            )
        ax.hist(
            metrics_real['wasserstein_distance_batched'][:, char_idx],
            bins=bins,
            alpha=0.6,
            label='Real Data',
            color='black',
            density=False
        )
        
        ax.set_title(f'Wasserstein Distance - {char_names[char_idx]}', fontsize=14)
        ax.set_ylabel('Number of Batches', fontsize=12)
        ax.set_xlabel('Wasserstein Distance', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{path}/{name}.png")
    plt.show()



if __name__ == "__main__":
    if __package__ is None:
        print("This module must be run as a package. From project root run:")
        print("  python -m skfda.datasets.diffusion.example")
        raise SystemExit(1)

    n_threads = torch.get_num_threads()
    print("Number of threads: {:d}".format(n_threads))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    data = generate_sin_function(
        n_samples=1000,
        amplitude_range=(0.5, 2),
        frequency_range=(3.5*np.pi, 4.5*np.pi),
        phase_range=(0,np.pi/2),
        noise=False,
    )
    intercept = (-1,1)
    slope = (0,0)
    data = generate_lines_function(n_samples=1000, n_points=128, intercept_range=intercept, slope_range=slope)
    a = np.array([intercept[0], slope[0]])
    b = np.array([intercept[1], slope[1]])
    diffusion_model = FDataGenerator(n_jobs=n_threads, device=device)
    diffusion_model.fit(data)
    generated_data = diffusion_model.generate(n_samples=100)
    w_dist = get_wasserstein_distance(generated_data, "sin", a, b, {})
    w_dist_real = get_wasserstein_distance(data, "sin", a, b, {})
    noise_metric = metrics.get_noise_metric(generated_data, "sin",{})
    # Aggregate metrics for the current model
    models_metrics = []
    aggregated_metrics = {
        'model_name': "vp",
        'nosie': noise_metric,
        'wasserstein_distance_batched': w_dist[None,],  # Add batch dimension for consistency
    }
    models_metrics.append(aggregated_metrics)
    g_data_list = [generated_data]
    diffusion_model = FDataGenerator(diff_process=VarianceExplodingDiffusionProcess(), scale_by_inv_sigma=False, n_jobs=n_threads, device=device)
    diffusion_model.fit(data)
    generated_data = diffusion_model.generate(n_samples=100)
    g_data_list.append(generated_data)
    w_dist = get_wasserstein_distance(generated_data, "lines", a, b, {'has_slope': True})
    noise_metric = metrics.get_noise_metric(generated_data, "lines",{'has_slope': True})
    print("Wasserstein distance for generated data: ", w_dist)
    # Aggregate metrics for the current model
    aggregated_metrics = {
        'model_name': "ve",
        'nosie': noise_metric,
        'wasserstein_distance_batched': w_dist[None,],  # Add batch dimension for consistency
    }
    models_metrics.append(aggregated_metrics)
    metrics_real = {
        'model_name': "real",
        'nosie': 0.0,  # No noise in real data
        'wasserstein_distance_batched': w_dist_real[None,],  # Add batch dimension for consistency
    }
    y_min = data.data_matrix.min() - 0.5
    y_max = data.data_matrix.max() + 0.5
    plot_combined_metrics_comparison(
        data,
        g_data_list,
        (y_min, y_max),
        20,
        metrics_real,
        models_metrics, ["intercept", "slope"],
        "./skfda/datasets/diffusion/Experiments",
        f"Big_intercept_({a[0]},{b[0]})_slope_({a[1]:.1f}pi,{b[1]:.1f}pi)_comparison",
        figsize=(20,8))


