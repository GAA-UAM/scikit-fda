import numpy as np
from binning import DataBinner

from skfda.representation import FDataGrid

grid_points = np.linspace(0, 5, 6)
data_matrix = np.array(
    [
        [np.nan, 2.0, 3.0, 4.0, 5.0, np.nan],
        [np.nan, 2.0, 2.0, 3.0, 5.0, 6.0],
        [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points, domain_range=(-2, 7))

print("Original Domain, Data and Grid Points:")
print(fd.data_matrix)
print(fd.grid_points)

print("Test Case 1: bins is an integer, range is None, non_empty is False, bin aggregation mode is median")

binner = DataBinner(bins=3, non_empty=False, bin_aggregation="median")
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)

print("Test Case 2: bins is an integer, range is None, non_empty is True, bin aggregation mode is mean")

binner = DataBinner(bins=3, non_empty=True, bin_aggregation="mean")
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)

print("Test Case 3: bins is an array, range is None")

bins = np.array([-1, 1, 2, 10])

binner = DataBinner(bins=bins, bin_aggregation="median")
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)

print("Test Case 4: bins is an integer, range is smaller than domain")

binner = DataBinner(bins=3, bin_aggregation="mean", range=(1, 4))
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)

print("Test Case 5: bins is an integer, range is bigger than domain")

binner = DataBinner(bins=3, bin_aggregation="mean", range=(-5, 9))
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)

print("Test Case 6: bins is an array, range is not None")

binner = DataBinner(bins=bins, bin_aggregation="median", range=(1, 4))
binned_data = binner.fit_transform(fd)

print(binned_data.data_matrix)
print(binned_data.grid_points)








grid_points = [np.linspace(0, 3, 4), np.linspace(0, 2, 3)]
data_matrix = np.array([
    [
        [np.nan, 2.0, 3.0],
        [np.nan, 2.0, 2.0],
        [np.nan, 2.0, 3.0],
        [np.nan, 1.5, 2.5]
    ],  
    [
        [1.0, 2.1, 3.1],
        [1.2, 2.3, 3.3],
        [1.4, 2.5, 3.5],
        [1.6, 2.7, 3.7]
    ],
    [
        [0.5, 1.5, 2.5],
        [0.7, 1.7, 2.7],
        [0.9, 1.9, 2.9],
        [1.1, 2.1, 3.1]
    ]
])
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points, domain_range=((-2, 7),(0,5)))
print(fd)
print(fd.domain_range)