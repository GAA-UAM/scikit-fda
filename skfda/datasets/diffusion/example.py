
import matplotlib.pyplot as plt
import numpy as np
import os

from .diffusion_process import (
    VariancePreservingDiffusionProcess,
    VarianceExplodingDiffusionProcess,
    VarianceExplodingDiffusionProcessAA3,
)

from ...representation.grid import FDataGrid
import torch
from .synthetic_data import generate_data

from .metrics import get_characteristics, wasserstein_to_uniform, get_noise_metric

from .diffusion_model import FDataGenerator

from matplotlib.gridspec import GridSpec


# ==================== PLOTTING FUNCTIONS ====================

def plot_combined_metrics_comparison(
        real_data,
        syn_data_list: list,
        ylim,
        n_plots,
        metrics_real: dict,
        metrics_list: list[dict],
        char_names: list[str],
        path: str,
        name: str,
        figsize=(20, 8)):
    '''
    Plots combined metrics comparison in two rows.
    Row 1: real data and all synthetic data models.
    Row 2: noise metric bar chart + Wasserstein distance histograms per characteristic.

    Args:
        real_data: The real functional data (FDataGrid).
        syn_data_list: List of generated FDataGrid, one per diffusion model.
        ylim: Y-axis limits for the data plots.
        n_plots: Number of functions to plot in the first row.
        metrics_real: Dict with 'wasserstein_distance_batched' for real data.
        metrics_list: List of dicts per model with 'model_name', 'noise',
                      'wasserstein_distance_batched'.
        char_names: List of characteristic names for labeling.
        path: Directory path to save the plot.
        name: File name (without extension).
        figsize: Figure size tuple (width, height).
    '''
    n_models = len(syn_data_list)
    n_plots = min(
        [n_plots, len(real_data)] +
        [len(syn_data) for syn_data in syn_data_list]
    )
    if n_plots > 100:
        print("Warning: Plotting more than 100 functions "
              "may not show clear conclusions.")

    n_chars = len(char_names)

    # Row 1: Real Data + n_models synthetic plots
    # Row 2: 1 noise bar + n_chars wasserstein histograms
    n_plots_row1 = n_models + 1
    n_plots_row2 = 1 + n_chars

    # Use LCM-friendly total columns
    total_cols = n_plots_row1 * n_plots_row2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, total_cols, figure=fig)

    # ==================== ROW 1: Data Plots ====================
    cols_per_plot_row1 = total_cols // n_plots_row1

    # Real data plot
    ax_real = fig.add_subplot(gs[0, :cols_per_plot_row1])
    if ylim is not None:
        ax_real.set_ylim(ylim)
    index_real = np.random.choice(len(real_data), n_plots, replace=False)
    real_data[index_real].plot(axes=ax_real, color='green')
    ax_real.set_title("Real Data", fontsize=12)

    # Synthetic data plots
    for i, syn_data in enumerate(syn_data_list):
        start_col = (i + 1) * cols_per_plot_row1
        end_col = (i + 2) * cols_per_plot_row1
        ax_syn = fig.add_subplot(gs[0, start_col:end_col], sharey=ax_real)
        if ylim is not None:
            ax_syn.set_ylim(ylim)
        syn_index = np.random.choice(len(syn_data), n_plots, replace=False)
        syn_data[syn_index].plot(axes=ax_syn, color='red')
        model_name = metrics_list[i]['model_name']
        ax_syn.set_title(f"Generated: {model_name}", fontsize=12)

    # ==================== ROW 2: Metrics ====================
    cols_per_plot_row2 = total_cols // n_plots_row2

    axs_row2 = []
    for i in range(n_plots_row2):
        ax = fig.add_subplot(
            gs[1, i * cols_per_plot_row2:(i + 1) * cols_per_plot_row2]
        )
        axs_row2.append(ax)

    colors = [
        'blue', 'orange', 'red', 'purple', 'brown',
        'olive', 'magenta', 'navy', 'teal', 'cyan',
    ]

    # Noise metric bar chart
    for i, m in enumerate(metrics_list):
        axs_row2[0].bar(
            m['model_name'], m['noise'],
            color=colors[i % len(colors)],
        )
    axs_row2[0].set_title('Noise Metric', fontsize=14)
    axs_row2[0].set_ylabel('MSE', fontsize=12)
    axs_row2[0].set_xlabel('Model', fontsize=12)
    axs_row2[0].grid(axis='y', alpha=0.3)

    # Wasserstein distance histograms per characteristic
    for char_idx in range(n_chars):
        ax = axs_row2[1 + char_idx]

        all_vals = np.concatenate(
            [m['wasserstein_distance_batched'][:, char_idx]
             for m in metrics_list] +
            [metrics_real['wasserstein_distance_batched'][:, char_idx]]
        )
        max_val = max(all_vals.max(), 1.0)
        bins = np.linspace(0, max_val, 50)

        for i, m in enumerate(metrics_list):
            ax.hist(
                m['wasserstein_distance_batched'][:, char_idx],
                bins=bins,
                alpha=0.6,
                label=m['model_name'],
                color=colors[i % len(colors)],
                density=False,
            )
        ax.hist(
            metrics_real['wasserstein_distance_batched'][:, char_idx],
            bins=bins,
            alpha=0.6,
            label='Real Data',
            color='black',
            density=False,
        )

        ax.set_title(
            f'Wasserstein Distance - {char_names[char_idx]}', fontsize=14
        )
        ax.set_ylabel('Number of Batches', fontsize=12)
        ax.set_xlabel('Wasserstein Distance', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_characteristics_distributions(
    characteristics_dict: dict,
    char_names: list[str],
    path: str = None,
    name: str = None,
    figsize=None,
):
    '''
    Plot a table of histograms where rows are models and columns are
    characteristics. Each cell shows the distribution (histogram) of that
    characteristic for that model.

    Real data is included as a row, allowing comparison between all models
    and the ground truth.

    Args:
        characteristics_dict: Ordered dict of
            {model_name: ndarray (n_samples, n_chars)}.
            The first entry is typically "Real Data".
        char_names: List of characteristic names for column headers.
        path: Directory path to save the plot. None to skip saving.
        name: File name (without extension). None to skip saving.
        figsize: Figure size tuple (width, height). Auto-computed if None.
    '''
    model_names = list(characteristics_dict.keys())
    n_models = len(model_names)
    n_chars = len(char_names)

    if figsize is None:
        figsize = (4 * n_chars, 3 * n_models)

    fig, axes = plt.subplots(n_models, n_chars, figsize=figsize, squeeze=False)

    colors = [
        'green', 'blue', 'orange', 'red', 'purple',
        'brown', 'olive', 'magenta', 'navy', 'teal',
    ]

    # Compute global min/max per characteristic for consistent x-axes
    global_min = np.full(n_chars, np.inf)
    global_max = np.full(n_chars, -np.inf)
    for chars in characteristics_dict.values():
        for j in range(n_chars):
            global_min[j] = min(global_min[j], chars[:, j].min())
            global_max[j] = max(global_max[j], chars[:, j].max())

    for i, model_name in enumerate(model_names):
        chars = characteristics_dict[model_name]
        for j in range(n_chars):
            ax = axes[i, j]
            bins = np.linspace(global_min[j], global_max[j], 50)
            ax.hist(
                chars[:, j], bins=bins,
                color=colors[i % len(colors)],
                alpha=0.7, density=True,
            )

            # Column headers on the top row
            if i == 0:
                ax.set_title(char_names[j], fontsize=14, fontweight='bold')
            # Row labels on the left column
            if j == 0:
                ax.set_ylabel(model_name, fontsize=12, fontweight='bold')

            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Characteristics Distributions by Model',
        fontsize=16, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    if path and name:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/{name}.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==================== EXPERIMENT HELPERS ====================

def compute_batched_wasserstein(characteristics, a, b, n_batches):
    """
    Compute Wasserstein distance in batches to obtain a distribution.

    Splits the characteristics array into n_batches equal parts and computes
    the Wasserstein distance to the uniform distribution for each batch.

    Args:
        characteristics: ndarray (n_samples, n_chars) of extracted features.
        a: Lower bounds of the uniform distribution, shape (n_chars,).
        b: Upper bounds of the uniform distribution, shape (n_chars,).
        n_batches: Number of batches to split into.

    Returns:
        ndarray (n_batches, n_chars) of Wasserstein distances per batch.
    """
    n_samples = characteristics.shape[0]
    batch_size = n_samples // n_batches
    w_batched = np.zeros((n_batches, len(a)))
    for i in range(n_batches):
        batch_chars = characteristics[i * batch_size:(i + 1) * batch_size]
        w_batched[i] = wasserstein_to_uniform(batch_chars, a, b)
    return w_batched


# ==================== EXPERIMENT RUNNER ====================

def run_experiment(
    dataset_config: dict,
    diffusion_configs: list[dict],
    n_samples_train: int = 1000,
    n_samples_gen: int = 1000,
    max_iter: int = 200,
    n_batches: int = 10,
    experiments_dir: str = "./Experiments",
    device=None,
    n_jobs: int = 0,
    n_plots: int = 20,
):
    """
    Run a full experiment for a given dataset with multiple diffusion models.

    For each experiment, two images are generated and saved:
    1. Combined comparison plot: data visualizations + Wasserstein distance
       histograms per characteristic + noise metric bar chart.
    2. Characteristics distribution table: rows = models (including real data),
       columns = characteristics, cells = distribution histograms.

    The pipeline is:
        extract characteristics -> compute Wasserstein distance
                                -> plot distributions

    Args:
        dataset_config: Configuration dict for the dataset. Must contain:
            - 'dataset': str (e.g., 'sin', 'lines', 'step')
            - 'char_names': list[str] of characteristic names
            - 'a': ndarray lower bounds for uniform distribution
            - 'b': ndarray upper bounds for uniform distribution
            Optional:
            - 'name': str experiment name (defaults to dataset name)
            - 'n_points': int grid points (default 100)
            - 'data_kwargs': dict kwargs for generate_data
            - 'char_kwargs': dict kwargs for get_characteristics
        diffusion_configs: List of dicts, each containing:
            - 'name': str model display name
            - 'process_class': class of ForwardDiffusionProcess
            - 'process_kwargs': dict kwargs for process constructor (optional)
            - 'scale_by_inv_sigma': bool (optional, default True)
        n_samples_train: Number of training samples.
        n_samples_gen: Number of generated samples per model.
        max_iter: Number of training epochs.
        n_batches: Number of batches for Wasserstein computation.
        experiments_dir: Directory to save result images.
        device: PyTorch device. Auto-detected if None.
        n_jobs: Number of data loader workers.
        n_plots: Number of functions to show in data visualization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(experiments_dir, exist_ok=True)

    # Unpack dataset configuration
    dataset_name = dataset_config['dataset']
    n_points = dataset_config.get('n_points', 100)
    data_kwargs = dataset_config.get('data_kwargs', {})
    char_kwargs = dataset_config.get('char_kwargs', {})
    char_names = dataset_config['char_names']
    a = dataset_config['a']
    b = dataset_config['b']
    experiment_name = dataset_config.get('name', dataset_name)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {dataset_name}, n_train={n_samples_train}, "
          f"n_gen={n_samples_gen}, max_iter={max_iter}")
    print(f"{'=' * 60}")

    # 1. Generate training data
    _, train_data = generate_data(
        n_samples_train, n_points, dataset_name, **data_kwargs,
    )

    # 2. Extract characteristics for real data (step 1 of the pipeline)
    real_chars = get_characteristics(train_data, dataset_name, char_kwargs)

    # 3. Compute batched Wasserstein for real data (step 2 of the pipeline)
    real_w_batched = compute_batched_wasserstein(real_chars, a, b, n_batches)
    metrics_real = {
        'model_name': 'Real Data',
        'wasserstein_distance_batched': real_w_batched,
    }

    # Store characteristics for all models (real data = first row)
    all_characteristics = {'Real Data': real_chars}
    syn_data_list = []
    metrics_list = []

    # 4. Train and evaluate each diffusion model
    for diff_config in diffusion_configs:
        model_name = diff_config['name']
        print(f"\nTraining: {model_name}")

        process = diff_config['process_class'](
            **diff_config.get('process_kwargs', {}),
        )

        generator = FDataGenerator(
            diff_process=process,
            scale_by_inv_sigma=diff_config.get('scale_by_inv_sigma', True),
            max_iter=max_iter,
            n_jobs=n_jobs,
            device=device,
        )
        generator.fit(train_data)
        gen_data = generator.generate(n_samples=n_samples_gen)
        syn_data_list.append(gen_data)

        # Pipeline step 1: Extract characteristics
        gen_chars = get_characteristics(gen_data, dataset_name, char_kwargs)
        all_characteristics[model_name] = gen_chars

        # Pipeline step 2: Compute Wasserstein distance from characteristics
        gen_w_batched = compute_batched_wasserstein(
            gen_chars, a, b, n_batches,
        )

        # Compute noise metric (reuses pre-computed characteristics)
        noise = get_noise_metric(
            gen_data, dataset_name, char_kwargs,
            characteristics=gen_chars,
        )
        noise_mean = float(np.mean(noise))

        metrics_list.append({
            'model_name': model_name,
            'noise': noise_mean,
            'wasserstein_distance_batched': gen_w_batched,
        })

        print(f"  Noise MSE: {noise_mean:.6f}")
        print(f"  Wasserstein (mean per char): {gen_w_batched.mean(axis=0)}")

    # 5. Compute y-axis limits for data plots
    y_min = train_data.data_matrix.min() - 0.5
    y_max = train_data.data_matrix.max() + 0.5

    # 6. Image 1: Combined comparison plot (data + Wasserstein + noise)
    plot_combined_metrics_comparison(
        real_data=train_data,
        syn_data_list=syn_data_list,
        ylim=(y_min, y_max),
        n_plots=n_plots,
        metrics_real=metrics_real,
        metrics_list=metrics_list,
        char_names=char_names,
        path=experiments_dir,
        name=f"{experiment_name}_comparison",
    )
    print(f"\nSaved: {experiments_dir}/{experiment_name}_comparison.png")

    # 7. Image 2: Characteristics distributions table
    plot_characteristics_distributions(
        characteristics_dict=all_characteristics,
        char_names=char_names,
        path=experiments_dir,
        name=f"{experiment_name}_distributions",
    )
    print(f"Saved: {experiments_dir}/{experiment_name}_distributions.png")


# ==================== DEFAULT CONFIGURATIONS ====================
#
# To customize experiments, modify these lists or create new ones.
# Each dataset config defines the data generation parameters and
# the expected characteristics to extract.
# Each diffusion config defines a model to train.

DIFFUSION_CONFIGS = [
    {
        'name': 'VP (cosine)',
        'process_class': VariancePreservingDiffusionProcess,
        'process_kwargs': {'beta_schedule': 'cosine'},
        'scale_by_inv_sigma': True,
    },
    {
        'name': 'VE (exponential)',
        'process_class': VarianceExplodingDiffusionProcess,
        'process_kwargs': {},
        'scale_by_inv_sigma': False,
    },
    {
        'name': 'VE (scaled)',
        'process_class': VarianceExplodingDiffusionProcess,
        'process_kwargs': {},
        'scale_by_inv_sigma': True,
    },
    {
        'name': 'VE (AA3)',
        'process_class': VarianceExplodingDiffusionProcessAA3,
        'process_kwargs': {},
        'scale_by_inv_sigma': True,
    }
]

DATASET_CONFIGS = [
    {
        'name': 'sin_default',
        'dataset': 'sin',
        'n_points': 100,
        'data_kwargs': {
            'amplitude_range': (0.5, 2.0),
            'frequency_range': (3.5 * np.pi, 4.5 * np.pi),
            'phase_range': (0, 0.3 * np.pi),
        },
        'char_kwargs': {},
        'char_names': ['Amplitude', 'Frequency', 'Phase'],
        'a': np.array([0.5, 3.5 * np.pi, 0]),
        'b': np.array([2.0, 4.5 * np.pi, 0.3 * np.pi]),
    },
    {
        'name': 'lines_with_slope',
        'dataset': 'lines',
        'n_points': 100,
        'data_kwargs': {
            'slope_range': (-2.0, 2.0),
            'intercept_range': (-1.0, 1.0),
        },
        'char_kwargs': {'has_slope': True},
        'char_names': ['Intercept', 'Slope'],
        'a': np.array([-1.0, -2.0]),
        'b': np.array([1.0, 2.0]),
    },
    {
        'name': 'lines_constant',
        'dataset': 'lines',
        'n_points': 100,
        'data_kwargs': {
            'slope_range': (0, 0),
            'intercept_range': (-1, 1),
        },
        'char_kwargs': {'has_slope': True},
        'char_names': ['Intercept', 'Slope'],
        'a': np.array([-1.0, 0]),
        'b': np.array([1.0, 0]),
    },
]


if __name__ == "__main__":
    if __package__ is None:
        print("This module must be run as a package. From project root run:")
        print("  python -m skfda.datasets.diffusion.example")
        raise SystemExit(1)

    n_threads = torch.get_num_threads()
    print(f"Threads: {n_threads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==================== EXPERIMENT PARAMETERS ====================
    # Modify these to control all experiments at once.
    N_SAMPLES_TRAIN = 1000
    N_SAMPLES_GEN = 1000
    MAX_ITER = 200
    N_BATCHES = 1
    N_PLOTS = 20
    EXPERIMENTS_DIR = "./skfda/datasets/diffusion/Experiments"

    # Select which datasets and diffusion models to run.
    # Comment/uncomment entries or add new ones to customize.
    dataset_configs = DATASET_CONFIGS
    diffusion_configs = DIFFUSION_CONFIGS

    # ==================== RUN EXPERIMENTS ====================
    for ds_config in dataset_configs:
        run_experiment(
            dataset_config=ds_config,
            diffusion_configs=diffusion_configs,
            n_samples_train=N_SAMPLES_TRAIN,
            n_samples_gen=N_SAMPLES_GEN,
            max_iter=MAX_ITER,
            n_batches=N_BATCHES,
            experiments_dir=EXPERIMENTS_DIR,
            device=device,
            n_jobs=n_threads,
            n_plots=N_PLOTS,
        )

    print("\nAll experiments completed!")
