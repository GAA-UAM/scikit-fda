import numpy as np
from binning import DataBinner

from skfda.representation import FDataGrid

def check_params(binner: DataBinner):
    print("Dim: ", binner.dim)
    print("Edges: ", binner.bin_edges)
    print("N Bins: ", binner.n_bins)
    print("Range: ", binner.range)
    print("Representative: ", binner.bin_representative)
    print("Output Grid: ", binner.output_grid)
    print("Aggregation method: ", binner.bin_aggregation)
    print("Non empty: ", binner.non_empty)
    print("Min Domain: ", binner.min_domain)
    print("Max Domain: ", binner.max_domain)
    print("\n")


# -- One Dimensional --

grid_points = np.linspace(0, 5, 6)
data_matrix = np.array(
    [
        [np.nan, 2.0, 3.0, 4.0, 5.0, np.nan],
        [np.nan, 2.0, 2.0, 3.0, 6.0, 6.0],
        [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
)
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points, domain_range=(-2, 7))

binner = DataBinner(
    bins=np.array([1,2,3,6]),
    bin_aggregation="median", 
    range=((1, 4)),
    output_grid="left",
)
print(binner.fit_transform(fd))
check_params(binner)

binner = DataBinner(
    bins=3, 
    bin_aggregation="median", 
    range=(1, 4),
    output_grid="left"
)
print(binner.fit_transform(fd))
check_params(binner)

binner = DataBinner(
    bins=3, 
    bin_aggregation="median", 
    output_grid="right",
    non_empty=True
)
print(binner.fit_transform(fd))
check_params(binner)

binner = DataBinner(
    bins=3, 
    bin_aggregation="median", 
    range=(1, 4),
    output_grid=np.array([2, 3, 4])
)
binner.fit_transform(fd)
check_params(binner)

# -- Multidimensional --
grid_points = [np.linspace(0, 3, 4), np.linspace(0, 2, 3)]
data_matrix = np.array([
    [
        [np.nan, 2.0, 3.0],
        [np.nan, 2.0, 2.5],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan]
    ],  
    [
        [1.0, 2.1, 3.1],
        [1.2, 1.0, 3.3],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan]
    ],
    [
        [0.5, 1.5, 2.5],
        [0.7, 1.7, 2.7],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan]
    ]
])
fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points, domain_range=((-2, 7),(0,5)))
print(fd)

# print("Data matrix: ", fd.data_matrix[:, [False, True, True, False], [False, True, True]])

binner = DataBinner(
    bins=(2,3), 
    bin_aggregation="median", 
    range=((0, 4), (0, 4)),
    output_grid=(np.array([2, 3]), np.array([1, 2, 4])),
    non_empty=True,
)
print(binner.fit_transform(fd))
check_params(binner)

binner = DataBinner(
    bins=(2,3), 
    bin_aggregation="median", 
    range=((0, 4), (0, 4)),
    output_grid=(np.array([2, 3]), np.array([1, 2, 4])),
    non_empty=False,
)
print(binner.fit_transform(fd))
check_params(binner)

data_matrix = np.array([
    [
        [np.nan, 2.0, 3.0],
        [np.nan, 2.0, 2.5],
        [np.nan, 7.0, 3.0],
        [np.nan, 1.5, 2.5]
    ],  
    [
        [1.0, 2.1, 3.1],
        [1.2, 1.0, 3.3],
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

binner = DataBinner(
    bins=(2,3), 
    bin_aggregation="mean", 
    range=((0, 4), (0, 2)),
    output_grid=(np.array([2, 3]), np.array([0, 1, 2])),
    non_empty=True,
)
print(binner.fit_transform(fd))
check_params(binner)

binner = DataBinner(
    bins=(2,3), 
    bin_aggregation="median", 
    output_grid="left"
)
binner.fit_transform(fd)
check_params(binner)

binner = DataBinner(
    bins=(np.array([1,2,3]), np.array([1,2,3,4])), 
    bin_aggregation="median", 
    range=((1, 4), (1, 4))
)
binner.fit_transform(fd)
check_params(binner)