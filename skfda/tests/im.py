import numpy as np
from skfda.representation import FDataGrid
from skfda.preprocessing import DataBinner
grid_points=[
    [1., 2.],
    [1., 2.],
]
values=np.array(
    [
        [
            [[1.0], [1.15]],
            [[1.45], [1.6]],
        ],
        [
            [[2.0], [2.15]],
            [[2.45], [2.6]],
        ],
    ]
)
fd = FDataGrid(
    grid_points=grid_points,
    data_matrix=values,
),
binner = DataBinner(bins=(2, 2))
binned_fd = binner.fit_transform(fd)