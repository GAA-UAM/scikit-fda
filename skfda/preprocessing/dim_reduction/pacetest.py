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
    centering=True,
    n_grid_points=61,
    # boundary_effect_interval=(0.1, 0.9),
    variance_error_interval=(0.25, 0.75),
)

# print(cd4.data)

pace.fit(fd)
# cd4 = fetch_cd4()
# pace.fit(cd4.data)

# t_matlab = np.array([
#     -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
#     -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
#     39, 40, 41, 42
# ])
# mu_matlab = np.array([
#     968.5930, 966.1337, 965.1959, 965.9797, 968.4143, 972.2001, 976.8664,
#     981.8387, 986.5110, 990.3067, 992.6978, 993.1532, 991.0188, 985.4300,
#     975.4770, 960.7085, 941.5415, 919.0714, 868.9633, 843.1778, 817.8532,
#     793.5688, 770.7743, 749.7596, 730.6933, 713.6864, 698.8022, 686.0022,
#     675.0835, 665.6710, 657.2884, 649.4767, 641.9093, 634.4678, 627.2623,
#     620.5915, 614.8536, 610.4287, 607.5573, 606.2393, 606.1827, 606.8244,
#     607.4195, 607.1811, 605.4469, 601.8386, 596.3571, 589.3693, 581.4937,
#     573.4517, 565.9380, 559.5352, 554.6674, 551.5887, 550.3976, 551.0617,
#     553.4378, 557.2745, 562.2011, 567.7152
# ])

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