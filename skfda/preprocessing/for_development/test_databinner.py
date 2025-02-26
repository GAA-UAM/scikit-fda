import numpy as np

from skfda.preprocessing.binning import DataBinner
from skfda.representation import FDataGrid


def check_binning_result(binner, fd, msg: str):
    try:
        output_fd = binner.fit_transform(fd)
        print(output_fd)
        print(f"\n^^^Parameter to check:^^^\n{msg}\n")

    except ValueError as e:
        print(f"\nValueError:\n\tExpected output: {msg}\n\tError: {e}")


def check_params(
    msg,
    bins=None,
    range=None,
    output_grid=None,
    bin_aggregation=None,
):
    try:
        DataBinner(
            bins=bins,
            range=range,
            output_grid=output_grid,
            bin_aggregation=bin_aggregation,
        )

    except ValueError as e:
        print(f"\nValueError:\n\tExpected output: {msg}\n\tError: {e}")


# -- FDataGrids to use in the tests -------------------------------------------

grid_points = np.linspace(0, 5, 6)
data_matrix = np.array(
    [
        [np.nan, 2.0, 3.0, 4.0, 5.0, np.nan],
        [np.nan, 2.0, 2.0, 3.0, 5.0, 6.0],
        [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
)
fd_1 = FDataGrid(
    data_matrix=data_matrix, grid_points=grid_points, domain_range=(-2, 7)
)

grid_points = [np.linspace(0, 3, 4), np.linspace(0, 2, 3)]
data_matrix = np.array(
    [
        [
            [np.nan, 2.0, 3.0],
            [np.nan, 2.0, 2.0],
            [np.nan, 2.0, 3.0],
            [np.nan, 1.5, 2.5],
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
        ],
    ]
)
fd_2 = FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
    domain_range=((-2, 7), (0, 5)),
)

data_matrix = np.array(
    [
        [
            [np.nan, 2.0, 3.0],
            [np.nan, 2.0, 2.5],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        [
            [1.0, 2.1, 3.1],
            [1.2, 1.0, 3.3],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        [
            [0.5, 1.5, 2.5],
            [0.7, 1.7, 2.7],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
    ]
)
fd_nan = FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
    domain_range=((-2, 7), (0, 5)),
)

grid_points = [
    np.linspace(0, 3, 4),
    np.linspace(0, 2, 3),
    np.linspace(0, 1, 2),
]
data_matrix = np.array(
    [
        [
            [
                [np.nan, 2.0],
                [np.nan, 2.5],
                [np.nan, 3.0]
            ],
            [
                [1.0, 2.1],
                [1.2, 2.3],
                [1.4, 2.5]
            ],
            [
                [0.5, 1.5],
                [0.7, 1.7],
                [0.9, 1.9]
            ],
            [
                [1.1, 2.1],
                [1.3, 2.3],
                [1.5, 2.5]
            ]
        ],
        [
            [
                [1.2, 2.2],
                [1.4, 2.4],
                [1.6, 2.6]
            ],
            [
                [0.8, 1.8],
                [1.0, 2.0],
                [1.2, 2.2]
            ],
            [
                [1.5, 2.5],
                [1.7, 2.7],
                [1.9, 2.9]
            ],
            [
                [2.0, 3.0],
                [2.2, 3.2],
                [2.4, 3.4]
            ]
        ]
    ]
)
fd_3 = FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
    # domain_range=((-2, 7), (0, 5), (-1, 2)),
)

# -- Tests --------------------------------------------------------------------

binner = DataBinner(bins=3, bin_aggregation="median")

check_binning_result(
    binner,
    fd_1,
    "Test Case 1: bins is an integer, range is None. Domain range must be "
    "inherited from the input FDataGrid.",
)

binner = DataBinner(
    bins=(2, 3),
)

check_binning_result(
    binner,
    fd_2,
    "Test Case 2: bins is an integer, range is None in the bidimensional "
    "case. Domain range must be inherited from the input FDataGrid.",
)

binner = DataBinner(bins=np.array([-1, 1, 2, 10]), bin_aggregation="median")

check_binning_result(
    binner,
    fd_1,
    "Test Case 3: bins is an array, range is None. Domain range must be "
    "defined by the bin edges.",
)

binner = DataBinner(
    bins=(np.array([-1, 1, 2, 4]), np.array([0, 1, 2, 3])),
    bin_aggregation="median",
)

check_binning_result(
    binner,
    fd_2,
    "Test Case 4: bins is an array, range is None in the bidimensional "
    "case. Domain range must be defined by the bin edges.",
)

binner = DataBinner(
    bins=3,
    bin_aggregation="median",
    range=(0, 2),
)

check_binning_result(
    binner,
    fd_1,
    "Test Case 5: bins is an integer, range is not None. Domain range must "
    "be defined by the parameter.",
)

binner = DataBinner(
    bins=(2, 3),
    range=((0, 7), (0, 7)),
)

check_binning_result(
    binner,
    fd_2,
    "Test Case 6: bins is an integer, range is not None in the bidimensional "
    "case. Domain range must be defined by the parameter.",
)

binner = DataBinner(
    bins=np.array([-1, 1, 2, 10]),
    bin_aggregation="median",
    range=(0, 2),
)

check_binning_result(
    binner,
    fd_1,
    "Test Case 7: bins is an array, range is not None. Range must be ignored "
    "and domain range must be defined by the bin edges.",
)

binner = DataBinner(
    bins=(np.array([-1, 1, 2, 4]), np.array([0, 1, 2, 3])),
    bin_aggregation="median",
    range=((0, 7), (0, 7)),
)

check_binning_result(
    binner,
    fd_2,
    "Test Case 8: bins is an array, range is not None in the bidimensional "
    "case. Range must be ignored and domain range must be defined by the "
    "bin edges.",
)

check_binning_result(
    binner,
    fd_1,
    "Test Case 9: binner is prepared for two-dimensional case, as dimension "
    "is defined by the bins parameter, but input is one-dimensional.",
)

check_params(
    "Test Case 10: range and bins don't share dimensionality.",
    bins=(3, 2),
    range=(0, 2),
    output_grid=(np.array([0, 1, 2]), np.array([0, 1, 2])),
    bin_aggregation="median",
)

check_params(
    "Test Case 11: output_grid and bins don't share dimensionality.",
    bins=(3, 2),
    range=((0, 2), (0, 2)),
    output_grid=np.array([0, 1, 2]),
    bin_aggregation="median",
)

check_params(
    "Test Case 12: output_grid elements don't fit in number of bins.",
    bins=(np.array([-1, 1, 2, 4]), np.array([0, 1, 2])),
    range=((0, 2), (0, 2)),
    output_grid=(np.array([0, 1, 2]), np.array([0, 1, 2])),
    bin_aggregation="median",
)

check_params(
    "Test Case 13: output_grid elements are not increasing.",
    bins=(np.array([-1, 1, 2, 4]), np.array([0, 1, 2])),
    range=((0, 2), (0, 2)),
    output_grid=(np.array([0, 1, 1]), np.array([0, 1])),
    bin_aggregation="median",
)

binner = DataBinner(
    bins=(np.array([-1, 1, 2, 4]), np.array([0, 1, 2])),
    range=((0, 2), (0, 2)),
    output_grid=(np.array([0, 1, 1.5]), np.array([0, 1])),
    bin_aggregation="median",
)

check_binning_result(
    binner,
    fd_2,
    "Test Case 14: output_grid points don't fit in the bin division. Each "
    "bin must contain one grid point.",
)

binner = DataBinner(
    bins=(2,3,2), 
    bin_aggregation="median", 
    # range=((0, 3), (0, 8), (-1, 10)),
    output_grid="right",
)

# Adds 4 upper elements, and then the two lower for the first two points in
# dim 0, and then the other two.
check_binning_result(
    binner,
    fd_3,
    "Test Case 15: check three-dimensional case.",
)