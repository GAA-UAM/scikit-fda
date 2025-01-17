import numpy as np
from binning import DataBinner

from skfda.representation import FDataGrid

# Test Case 1: Randomized Data with NaNs
print("--- Test Case 1: Randomized Data with NaNs ---")
grid_points = np.linspace(0, 20, 20)
data_matrix = np.random.uniform(1, 10, (3, 20))
nan_indices = np.random.choice(data_matrix.size, 58, replace=False)
data_matrix.ravel()[nan_indices] = np.nan
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

binner = DataBinner(n_bins=7, non_empty=True, mode="mean")
binned_data = binner.fit_transform(fd)

print("Original Data Matrix:")
print(fd.data_matrix)
print("\nBinned Data Matrix (Mean):")
print(binned_data.data_matrix)
print("\nOriginal Grid Points:")
print(fd.grid_points)
print("\nBinned Grid Points:")
print(binned_data.grid_points)


# Test Case 2: Weighted Mean Mode
print("\n--- Test Case 2: Weighted Mean Mode ---")
grid_points = np.linspace(0, 1, 10)
data_matrix = np.array(
    [
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
        [2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

binner = DataBinner(n_bins=5, mode="weighted_mean")
binned_data = binner.fit_transform(fd)

print("Original Data Matrix:")
print(fd.data_matrix)
print("\nBinned Data Matrix (Weighted Mean):")
print(binned_data.data_matrix)
print("\nOriginal Grid Points:")
print(fd.grid_points)
print("\nBinned Grid Points:")
print(binned_data.grid_points)


# Test Case 3: Median Mode
print("\n--- Test Case 3: Median Mode ---")
grid_points = np.linspace(0, 20, 10)
data_matrix = np.array(
    [
        [2, 2, 3, 3, 4, 4, 5, 5, np.nan, np.nan],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [np.nan, np.nan, np.nan, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

binner = DataBinner(n_bins=5, mode="median", non_empty=False)
binned_data = binner.fit_transform(fd)

print("Original Data Matrix:")
print(fd.data_matrix)
print("\nBinned Data Matrix (Median):")
print(binned_data.data_matrix)
print("\nOriginal Grid Points:")
print(fd.grid_points)
print("\nBinned Grid Points:")
print(binned_data.grid_points)


# Test Case 4: Bin edges as argument
print("\n--- Test Case 4: Bin edges as argument ---")
grid_points = np.linspace(0, 10, 10)
data_matrix = np.array(
    [
        [2, 2, 3, 3, 4, 4, 5, 5, np.nan, np.nan],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [np.nan, np.nan, np.nan, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

bin_edges = np.array([0, 1, 4, 9, 10])
binner = DataBinner(bin_edges=bin_edges, mode="median", non_empty=False)
binned_data = binner.fit_transform(fd)

print("Original Data Matrix:")
print(fd.data_matrix)
print("\nBinned Data Matrix (Median):")
print(binned_data.data_matrix)
print("\nOriginal Grid Points:")
print(fd.grid_points)
print("\nBinned Grid Points:")
print(binned_data.grid_points)


# Test Case 5: Bin edges out of domain
print("\n--- Test Case 5: Bin edges out of domain ---")
grid_points = np.linspace(0, 10, 10)
data_matrix = np.array(
    [
        [2, 2, 3, 3, 4, 4, 5, 5, np.nan, np.nan],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [np.nan, np.nan, np.nan, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

bin_edges = np.array([-1, 1, 4])
binner = DataBinner(bin_edges=bin_edges, mode="median", non_empty=False)
binned_data = binner.fit_transform(fd)

print("Original Data Matrix:")
print(fd.data_matrix)
print("\nBinned Data Matrix (Median):")
print(binned_data.data_matrix)
print("\nOriginal Grid Points:")
print(fd.grid_points)
print("\nBinned Grid Points:")
print(binned_data.grid_points)
