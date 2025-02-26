import numpy as np
from skfda.preprocessing.binning import DataBinner
from skfda.representation import FDataIrregular, FDataGrid

# Irregular ------------------------------------------------------------------

# Example FDataIrregular with two observations
indices = [0, 2]
arguments = [[1.], [4.], [3.], [4.], [5.]]
values = [[1., 1., 1], [2., 2., 1], [3., 3., 1], [4., 4., 1], [5., 5., 1]]
fd = FDataIrregular(indices, arguments, values)

print(fd.dim_domain, fd.dim_codomain)
print("Domain range: ", fd.domain_range[0])
binner = DataBinner(bins=2, output_grid=np.array([1, 3]))
print(binner.fit_transform(fd))
# exit()

indices = [0, 2]  # Two samples
arguments = np.array([
    [1., 1., 5], [2., 2., 3], [3., 3., 4],  # First sample
    [4., 4., 4], [5., 6., 2]   # Second sample
])
values = np.array([
    10., 20.,
    30., 40., 50.        # Second sample
])
fd = FDataIrregular(indices, arguments, values)
print(fd.dim_domain, fd.dim_codomain)
print("Domain range: ", fd.domain_range)
binner = DataBinner(bins=(1,2, 2), 
                    bin_aggregation="median", 
                    range=((1, 5), (1, 6), (2, 5)),
                    output_grid="left")
print(binner.fit_transform(fd))


exit()

# Grid -----------------------------------------------------------------------
print("\n\n\n")

data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
grid_points = [2, 4]
fd = FDataGrid(data_matrix, grid_points,
               coordinate_names=["Component 1", "Component 2"])
print("Dims: ", fd.dim_domain, fd.dim_codomain)
print(fd.data_matrix.shape)
print("Domain range: ", fd.domain_range)

binner = DataBinner(bins=1)
# print(binner.fit_transform(fd))

data_matrix = [
    [
        [1, 0.3], 
        [2, 0.4]
    ], 
    [
        [2, 0.5], 
        [3, 0.6]
    ]
]
grid_points = [[2, 4], [3,6]]
fd = FDataGrid(data_matrix, grid_points)
# print(fd)

print(fd.dim_domain, fd.dim_codomain)
print(fd.data_matrix.shape)
print("Domain range: ", fd.domain_range)


binner = DataBinner(bins=(1,2), bin_aggregation="median")
# binned_fd = binner.fit_transform(fd)
# print(binned_fd)
