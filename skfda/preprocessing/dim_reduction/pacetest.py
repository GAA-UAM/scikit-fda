from skfda.preprocessing.dim_reduction._pace import PACE
from skfda.datasets._real_datasets import fetch_cd4
import numpy as np
from skfda.representation import FDataIrregular
import matplotlib.pyplot as plt




n_samples = 10  # Number of samples
n_points = 2    # Points per observation

# Generate random time points and values for each sample
time_points = [
    [0,0],
    [2,2],
    [4,4],
    [6,6],
    [8,8],
    [1,1],
    [3,3],
    [5,5],
    [7,7],
    [9,9],
]
values = [np.random.rand() for _ in range(n_samples)]

fd = FDataIrregular(
    start_indices=[0, 5],
    points=time_points,
    values=values,
    argument_names=["x", "y"],
)
# print(fd)

pace = PACE(
    n_components=2,
    n_grid_points=51,
    bandwidth_mean=4.061,
    bandwidth_cov=7.2998,
    assume_noisy=False,
    # boundary_effect_interval=(0.1, 0.9),
    variance_error_interval=(0.25, 0.75),
)

# print(cd4.data)

# pace.fit(fd)
# exit()
cd4 = fetch_cd4()
pace.fit(cd4.data)

t_matlab = np.array([
    -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
    -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42
])
mu_matlab = np.array([
    968.5930, 966.1337, 965.1959, 965.9797, 968.4143, 972.2001, 976.8664,
    981.8387, 986.5110, 990.3067, 992.6978, 993.1532, 991.0188, 985.4300,
    975.4770, 960.7085, 941.5415, 919.0714, 868.9633, 843.1778, 817.8532,
    793.5688, 770.7743, 749.7596, 730.6933, 713.6864, 698.8022, 686.0022,
    675.0835, 665.6710, 657.2884, 649.4767, 641.9093, 634.4678, 627.2623,
    620.5915, 614.8536, 610.4287, 607.5573, 606.2393, 606.1827, 606.8244,
    607.4195, 607.1811, 605.4469, 601.8386, 596.3571, 589.3693, 581.4937,
    573.4517, 565.9380, 559.5352, 554.6674, 551.5887, 550.3976, 551.0617,
    553.4378, 557.2745, 562.2011, 567.7152
])

# plt.figure(figsize=(10, 6))
# plt.plot(t_matlab, mu_matlab, label='PACE MATLAB', color='blue', linewidth=2)
# plt.plot(t_matlab, pace.mean_, label='PACE Python', color='red', linestyle='--', linewidth=2)
# plt.xlabel('Time (months)')
# plt.ylabel('Estimated Mean CD4 Count')
# plt.title('Comparison of Estimated Mean Function (MATLAB vs Python)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

# Interpolate Python PACE mean (assumes domain is 1D)
interp_python_mean = interp1d(
    t_matlab,              # x values
    pace.mean_[:, 0],                     # y values
    kind='linear',
    bounds_error=False,
    fill_value='extrapolate',
)

# Evaluate interpolated mean at MATLAB time points
mu_python_interp = interp_python_mean(t_matlab)

# Compute MSE between MATLAB and Python mean
mse = mean_squared_error(mu_matlab, mu_python_interp)
print(f"📐 Mean Squared Error (Python vs MATLAB): {mse:.4f}")



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid points used during the fit (used in _cov_lls)
grid = np.linspace(cd4.data.domain_range[0][0], cd4.data.domain_range[0][1], pace.n_grid_points)

# Extract the smoothed covariance matrix
cov = pace.covariance_.squeeze()  # shape: (n_grid_points, n_grid_points)

# Create meshgrid for r and s
R_grid, S_grid = np.meshgrid(grid, grid, indexing='ij')

# Optional: Simulate missing or noisy data to scatter on top
# In this case we just visualize the smoothed covariance

# Plotting the covariance surface
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(R_grid, S_grid, cov, cmap='viridis', alpha=0.7)

# If you had observed (r,s,G) points and wanted to scatter them:
# for i in range(...):
#     ax.scatter(r, s, G_val, color='red', label="Observed Covariance")

ax.set_xlabel('r (Time)')
ax.set_ylabel('s (Time)')
ax.set_zlabel('Covariance G(r, s)')
ax.set_title('Smoothed Covariance Surface via PACE')
plt.tight_layout()
plt.show()
