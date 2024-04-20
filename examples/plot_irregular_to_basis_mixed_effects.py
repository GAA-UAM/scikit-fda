"""
Mixed effects model to convert irregular data to basis representation
=======================================================================

This example converts irregular data to a basis representation using a mixed
effects model.
"""
# Author: Pablo Cuesta Sierra
# License: MIT

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skfda import FDataBasis
from skfda.datasets import fetch_weather, irregular_sample
from skfda.representation.basis import BSplineBasis, FourierBasis
from skfda.representation.conversion import EMMixedEffectsConverter
from skfda.misc.scoring import r2_score, mean_squared_error


# %%
# Converting irregular data to basis representation
# #################################################
# For the first part of this example, we are going to simulate the irregular
# sampling of a dataset following the mixed effects model, to later attempt to
# reconstruct the original data.
#
# First, we generate the original basis representation of the data following
# the mixed effects model for irregular data as presented by
# :footcite:t:`james_2018_sparsenessfda`. This just means that
# the coefficients of the basis representation are generated from a Gaussian
# distribution.
n_curves = 50
n_basis = 4
domain_range = (0, 12)
basis = BSplineBasis(n_basis=n_basis, domain_range=domain_range, order=4)

basis.plot()
plt.title("Basis functions")

coeff_mean = np.array([-10, 20, -24, 4])
cov_sqrt = np.array([
    [3.2, 0.0, 0.0, 0.0],
    [0.4, 6.0, 0.0, 0.0],
    [0.3, 1.5, 2.0, 0.0],
    [1.2, 0.3, 2.5, 1.8],
])
random_state = np.random.RandomState(seed=4934755)
coefficients = (
    coeff_mean + random_state.normal(size=(n_curves, n_basis)) @ cov_sqrt
)
fdatabasis_original = FDataBasis(basis, coefficients)

# Plot the first 10 curves
fdatabasis_original[:10].plot()
plt.title("Original curves")
plt.show()


# %%
# Sencondly, we simulate the irregular sampling of the original data.
fd_irregular = irregular_sample(
    fdatabasis_original,
    n_points_per_curve=3,  # Number of points per curve in the irregular data
    random_state=random_state,
)

# Plot the last 6 curves of the newly created irregular data
fig = plt.figure()
fdatabasis_original[-6:].plot(fig=fig, alpha=0.3)
fd_irregular[-6:].scatter(fig=fig)
plt.show()

# %%
# We split our irregular data into two groups, the train curves
# and the test curves.
train_original, test_original, train_irregular, test_irregular = (
    train_test_split(
        fdatabasis_original,
        fd_irregular,
        test_size=0.5,
        random_state=random_state,
    )
)

# %%
# Now, we create and train the mixed effects converter using the train curves,
converter = EMMixedEffectsConverter(basis)
converter = converter.fit(train_irregular)

# %%
# and we convert the irregular data to basis representation.
train_converted = converter.transform(train_irregular)
test_converted = converter.transform(test_irregular)

# For comparison, we also convert to basis representation using the separate
# basis representation for each curve.
train_separate_basis = train_irregular.to_basis(basis)
test_separate_basis = test_irregular.to_basis(basis)

# %%
# To visualize the conversion results, we plot the first 8 original and
# converted curves of the test set. On the background, we plot the train set.
fig = plt.figure(figsize=(10, 15))
for k in range(8):
    axes = plt.subplot(4, 2, k + 1)

    train_original.plot(axes=axes, color=(0, 0, 0, 0.05))
    train_irregular.scatter(axes=axes, color=(0, 0, 0, 0.05), marker=".")

    test_converted[k].plot(
        axes=axes, color=f"C{k}", linestyle="--", label="Converted",
    )
    test_original[k].plot(
        axes=axes, color=f"C{k}", linewidth=0.65, label="Original",
    )
    test_irregular[k].scatter(
        axes=axes, color=f"C{k}", label="Irregular"
    )
    test_separate_basis[k].plot(
        axes=axes, color=f"C{k}", linestyle=":",
        label="Separate basis representation",
    )
    plt.legend()
plt.show()

# %%
# Finally, we make use of the :math:`R^2` score and the :math:`MSE` to compare
# the converted basis representations with the original data, both for the
# train and test sets.
train_r2_score = r2_score(train_original, train_converted)
test_r2_score = r2_score(test_original, test_converted)
train_mse = mean_squared_error(train_original, train_converted)
test_mse = mean_squared_error(test_original, test_converted)
train_r2_score_separate = r2_score(train_original, train_separate_basis)
test_r2_score_separate = r2_score(test_original, test_separate_basis)
train_mse_separate = mean_squared_error(train_original, train_separate_basis)
test_mse_separate = mean_squared_error(test_original, test_separate_basis)
print(f"R2 score (mixed effects - train): {train_r2_score:.2f}")
print(f"R2 score (mixed effects - test): {test_r2_score:.2f}")
print(f"R2 score (separate basis - train): {train_r2_score_separate:.2f}")
print(f"R2 score (separate basis - test): {test_r2_score_separate:.2f}")
print(f"Mean Squared Error (mixed effects - train): {train_mse:.2f}")
print(f"Mean Squared Error (mixed effects - test): {test_mse:.2f}")
print(f"Mean Squared Error (separate basis - train): {train_mse_separate:.2f}")
print(f"Mean Squared Error (separate basis - test): {test_mse_separate:.2f}")

# %%
# Check robustness of the method by removing measurement points
# #############################################################
# For the second part of the example, we are going to check the robustness of
# the method by removing some measurement points from the test and train sets
# and comparing the results. The temperatures from the Canadian weather
# dataset are used to generate the irregular data.
fd_temperatures = fetch_weather().data.coordinates[0]
fd_irregular = irregular_sample(
    fdata=fd_temperatures, n_points_per_curve=40, random_state=random_state,
)
basis = FourierBasis(n_basis=5, domain_range=fd_temperatures.domain_range)

# %%
# Split the data into train and test sets
train_original, test_original, train_irregular, test_irregular = (
    train_test_split(
        fd_temperatures,
        fd_irregular,
        test_size=0.2,
        random_state=random_state,
    )
)

# %%
# Create the different datasets by removing some measurement points
train_irregular_list = []
test_irregular_list = []
n_points_list = [40, 10, 5, 4, 3]
for n_points in n_points_list:
    train_irregular_list.append(
        irregular_sample(
            train_original,
            n_points_per_curve=n_points,
            random_state=random_state,
        )
    )
    test_irregular_list.append(
        irregular_sample(
            test_original,
            n_points_per_curve=n_points,
            random_state=random_state,
        )
    )

# %%
# We convert the irregular data to basis representation and compute the scores:
scores = {
    "n_points_per_curve": n_points_list,
    "Train R2 score": [],
    "Test R2 score": [],
    "Train MSE": [],
    "Test MSE": [],
}
converter = EMMixedEffectsConverter(basis)
for train_irregular, test_irregular in zip(
    train_irregular_list,
    test_irregular_list,
):
    converter = converter.fit(train_irregular)
    train_converted = converter.transform(train_irregular)
    test_converted = converter.transform(test_irregular)

    scores["Train R2 score"].append(r2_score(
        train_original, train_converted.to_grid(train_original.grid_points),
    ))
    scores["Test R2 score"].append(r2_score(
        test_original, test_converted.to_grid(test_original.grid_points),
    ))
    scores["Train MSE"].append(mean_squared_error(
        train_original, train_converted.to_grid(train_original.grid_points),
    ))
    scores["Test MSE"].append(mean_squared_error(
        test_original, test_converted.to_grid(test_original.grid_points),
    ))

# %%
# Finally, we can see the results in a table:
df = pd.DataFrame(scores)
df = df.set_index("n_points_per_curve")
print(df)

# %%
# References
# ----------
#
# .. footbibliography::
