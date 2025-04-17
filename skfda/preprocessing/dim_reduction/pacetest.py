import matplotlib.pyplot as plt
import numpy as np

from skfda.datasets._real_datasets import fetch_cd4
from skfda.preprocessing.dim_reduction._pace import PACE
from skfda.representation import FDataIrregular

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
    n_components=0.99,
    n_grid_points=51,
    bandwidth_mean=8.25,
    bandwidth_cov=6.269,
    # bandwidth_cov=np.array([5.0, 30.0]),
    # assume_noisy=False,
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
    39, 40, 41, 42,
])
mu_matlab = np.array([
    986.1189, 989.1560, 991.6712, 993.5323, 994.6030, 994.7466,
    993.8285, 991.7216, 988.3114, 983.5016, 977.2211, 969.4295,
    960.1220, 949.3330, 937.1374, 923.6488, 909.0164, 893.4185,
    860.1407, 842.8924, 825.5250, 808.2426, 791.2328, 774.6622,
    758.6738, 743.3849, 728.8873, 715.2470, 702.5060, 690.6839,
    679.7799, 669.7756, 660.6366, 652.3158, 644.7553, 637.8886,
    631.6432, 625.9426, 620.7085, 615.8628, 611.3299, 607.0380,
    602.9214, 598.9214, 594.9876, 591.0794, 587.1662, 583.2283,
    579.2575, 575.2572, 571.2423, 567.2387, 563.2831, 559.4211,
    555.7061, 552.1970, 548.9562, 546.0469, 543.5310, 541.4660,
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

from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

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


import matplotlib.pyplot as plt
import numpy as np
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
# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(R_grid, S_grid, cov, cmap='viridis', alpha=0.7)
# ax.set_xlabel('r (Time)')
# ax.set_ylabel('s (Time)')
# ax.set_zlabel('Covariance G(r, s)')
# ax.set_title('Smoothed Covariance Surface via PACE')
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Load expected covariance matrix from MATLAB (comma-separated)
cov_matlab = np.loadtxt("skfda/preprocessing/dim_reduction/cd4_cov_from_matlab.csv", delimiter=",")
# print(cov_matlab.shape)
# print(cov_matlab)

# Get Python PACE covariance
cov_python = pace.covariance_
cov_python = cov_python.squeeze()  # shape: (n_grid_points, n_grid_points)
# print(cov_python.shape)
# print(cov_python)

# Check dimensions match
assert cov_matlab.shape == cov_python.shape, "Mismatch in matrix dimensions!"

# Compute MSE
cov_mse = mean_squared_error(cov_matlab.flatten(), cov_python.flatten())
print(f"📊 Covariance MSE (Python vs MATLAB): {cov_mse:.6f}")

# Plot for visual comparison
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(cov_matlab, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
# plt.title("MATLAB Covariance")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(cov_python, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
# plt.title("Python Covariance")
# plt.colorbar()

# plt.suptitle("Covariance Surface Comparison")
# plt.tight_layout()
# plt.show()

