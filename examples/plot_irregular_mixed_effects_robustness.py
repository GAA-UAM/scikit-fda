"""
Mixed-effects model for irregular data when removing measurement points
=======================================================================

This example converts irregular data to a basis representation using a mixed
effects model and checks the robustness of the method by fitting
the model with decreasing number of measurement points per curve. 
"""
# Author: Pablo Cuesta Sierra
# License: MIT

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skfda import FDataIrregular
from skfda.datasets import fetch_weather, irregular_sample
from skfda.representation.basis import FourierBasis
from skfda.representation.conversion import EMMixedEffectsConverter
from skfda.misc.scoring import r2_score, mean_squared_error


# %%
# For this example, we are going to check the robustness of
# the mixed effects method for converting irregular data to basis
# representation by removing some measurement points from the test and train
# sets and comparing the results. The temperatures from the Canadian weather
# dataset are used to generate the irregular data.
fd_temperatures = fetch_weather().data.coordinates[0]
basis = FourierBasis(n_basis=5, domain_range=fd_temperatures.domain_range)

fd_temperatures.plot()
plt.show()
basis.plot()
plt.title("Basis functions")
plt.show()

# %%
# We split the data into train and test sets:
random_state = np.random.RandomState(seed=4934792)
train_original, test_original = train_test_split(
    fd_temperatures,
    test_size=0.3,
    random_state=random_state,
)

# %%
# Then, we create datasets with decreasing number of measurement points per
# curve, by removing measurement points from the previous dataset iteratively.
train_irregular_list = [train_original]
test_irregular_list = [test_original]
n_points_list = [40, 10, 7, 5, 4, 3]
for n_points in n_points_list:
    train_irregular_list.append(
        irregular_sample(
            train_irregular_list[-1],
            n_points_per_curve=n_points,
            random_state=random_state,
        ),
    )
    test_irregular_list.append(
        irregular_sample(
            test_irregular_list[-1],
            n_points_per_curve=n_points,
            random_state=random_state,
        ),
    )

train_irregular_datasets = {
    n_points: train_irregular
    for n_points, train_irregular in zip(
        n_points_list, train_irregular_list[1:],
    )
}
test_irregular_datasets = {
    n_points: test_irregular
    for n_points, test_irregular in zip(
        n_points_list, test_irregular_list[1:],
    )
}

# %%
# We convert the irregular data to basis representation and compute the scores.
# To do so, we fit the converter once per train set. After fitting the
# the converter with a train set that has :math:`k` points per curve, we
# use it to transform that train set, the test set with :math:`k` points per
# curve and the original test set with 365 points per curve.
score_functions = {
    "R^2": r2_score,
    "MSE": mean_squared_error,
}
converted_data = {
    "Train-sparse": {},
    "Test-sparse": {},
    "Test-original": {},
}
scores = {
    score_name: {
        "n_points_per_curve": n_points_list,
        **{data_name: [] for data_name in converted_data.keys()},
    }
    for score_name in score_functions.keys()
}
converter = EMMixedEffectsConverter(basis)
for n_points, train_irregular, test_irregular in zip(
    n_points_list,
    train_irregular_datasets.values(),
    test_irregular_datasets.values(),
):
    converter = converter.fit(train_irregular)
    train_sparse_converted = converter.transform(train_irregular)
    test_sparse_converted = converter.transform(test_irregular)
    test_original_converted = converter.transform(
        FDataIrregular.from_fdatagrid(test_original),
    )
    converted_data["Train-sparse"][n_points] = train_sparse_converted
    converted_data["Test-sparse"][n_points] = test_sparse_converted
    converted_data["Test-original"][n_points] = test_original_converted

    for score_name, score_fun in score_functions.items():
        scores[score_name]["Train-sparse"].append(score_fun(
            train_original,
            train_sparse_converted.to_grid(train_original.grid_points),
        ))
        scores[score_name]["Test-sparse"].append(score_fun(
            test_original,
            test_sparse_converted.to_grid(test_original.grid_points),
        ))
        scores[score_name]["Test-original"].append(score_fun(
            test_original,
            test_original_converted.to_grid(test_original.grid_points),
        ))

# %%
# Finally, we have the scores for the train and test sets with decreasing
# number of measurement points per curve.
for score_name in scores.keys():
    print("-" * 62)
    print(f"{score_name} scores:")
    print("-" * 62)
    print((
        pd.DataFrame(scores[score_name])
        .set_index("n_points_per_curve").sort_index().to_string()
    ), end="\n\n\n")

# %%
# The following plots show the original curves along with the converted
# test curves for the conversions with 5, 4 and 3 points per curve.


def plot_converted_test_curves(n_points_per_curve):
    plt.figure(figsize=(10, 23))
    for k in range(7):
        axes = plt.subplot(7, 1, k + 1)

        test_irregular_datasets[n_points_per_curve][k].scatter(
            axes=axes, color=f"C{k}",
        )
        test_original[k].plot(
            axes=axes, color=f"C{k}", linewidth=0.65,
            label="Original test curve",
        )
        converted_data["Test-sparse"][n_points_per_curve][k].plot(
            axes=axes, color=f"C{k}", linestyle="--",
            label=f"Test curve transformed from {n_points_per_curve} points",
        )
        converted_data["Test-original"][n_points_per_curve][k].plot(
            axes=axes, color=f"C{k}", alpha=0.5,
            label="Test curve transformed from original 365 points",
        )
        axes.legend(bbox_to_anchor=(1., 1.))
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.suptitle(f"Fitted model with {n_points_per_curve=}")

    plt.show()


# %%
plot_converted_test_curves(n_points_per_curve=5)

# %%
plot_converted_test_curves(n_points_per_curve=4)

# %%
plot_converted_test_curves(n_points_per_curve=3)

# %%
# References
# ----------
#
# .. footbibliography::
