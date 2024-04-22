"""
Mixed effects model for irregular data
==============================================================================

This example converts irregular data to a basis representation using a mixed
effects model.
"""
# %%
# Author: Pablo Cuesta Sierra
# License: MIT

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skfda import FDataBasis, FDataIrregular
from skfda.datasets import irregular_sample
from skfda.representation.basis import BSplineBasis
from skfda.representation.conversion import EMMixedEffectsConverter
from skfda.misc.scoring import r2_score, mean_squared_error


# %%
# For this example, we are going to simulate the irregular
# sampling of a dataset following the mixed effects model, to later attempt to
# reconstruct said original dataset.
#
# First, we generate the original basis representation of the data following
# the mixed effects model for irregular data as presented by
# :footcite:t:`james_2018_sparsenessfda`. This just means that
# the coefficients of the basis representation are generated from a Gaussian
# distribution.
n_curves = 50
n_basis = 4
domain_range = (0, 12)
basis = BSplineBasis(n_basis=n_basis, domain_range=domain_range, order=3)

plt.figure(figsize=(10, 5))
basis.plot()
plt.title("Basis functions")

coeff_mean = np.array([-10, 20, -24, 4])
coeff_cov_sqrt = np.array([
    [3.2, 0.0, 0.0, 0.0],
    [0.4, 6.0, 0.0, 0.0],
    [0.3, 1.5, 2.0, 0.0],
    [1.2, 0.3, 2.5, 1.8],
])
random_state = np.random.RandomState(seed=4934755)
coefficients = (
    coeff_mean + random_state.normal(size=(n_curves, n_basis)) @ coeff_cov_sqrt
)
fdatabasis_original = FDataBasis(basis, coefficients)

# Plot the first 6 curves
plt.figure(figsize=(10, 5))
fdatabasis_original[:6].plot()
plt.title("Original curves")
plt.show()


# %%
# Sencondly, we subsample of the original data by measuring a random number of
# points per curve generating an irregular dataset.
# Moreover, we add some noise to the data.
fd_irregular_without_noise = irregular_sample(
    fdatabasis_original,
    n_points_per_curve=random_state.randint(3, 5, n_curves),
    random_state=random_state,
)
noise_std = 0.1
fd_irregular = FDataIrregular(
    points=fd_irregular_without_noise.points,
    start_indices=fd_irregular_without_noise.start_indices,
    values=fd_irregular_without_noise.values + random_state.normal(
        0, noise_std, fd_irregular_without_noise.values.shape,
    ),
)

# Plot 9 curves of the newly created irregular data
fig = plt.figure(figsize=(10, 10))
for k in range(9):
    axes = plt.subplot(3, 3, k + 1)
    fdatabasis_original[k].plot(axes=axes, alpha=0.3, color=f"C{k}")
    fd_irregular[k].plot(axes=axes, marker=".", color=f"C{k}")
plt.show()

# %%
# We split our irregular data into two groups, the train curves
# and the test curves.
train_original, test_original, train_irregular, test_irregular = (
    train_test_split(
        fdatabasis_original,
        fd_irregular,
        test_size=0.3,
        random_state=random_state,
    )
)

# %%
# Now, we create and train the mixed effects converter using the train curves,
# and we convert the irregular data to basis representation.
# For comparison, we also convert to basis representation using the default
# basis representation for each curve, which is done curve-wise instead of
# taking into account the whole dataset.
converter = EMMixedEffectsConverter(basis)
converter.fit(train_irregular)

train_converted = converter.transform(train_irregular)
test_converted = converter.transform(test_irregular)

train_curvewise_to_basis = train_irregular.to_basis(basis)
test_curvewise_to_basis = test_irregular.to_basis(basis)

# %%
# To visualize the conversion results, we plot the first 8 original and
# converted curves of the test set. On the background, we plot the train set.
fig = plt.figure(figsize=(10, 25))
for k in range(8):
    axes = plt.subplot(8, 1, k + 1)

    # train_original.plot(axes=axes, color=(0, 0, 0, 0.05))
    # train_irregular.scatter(axes=axes, color=(0, 0, 0, 0.05), marker=".")

    test_irregular[k].scatter(
        axes=axes, color=f"C{k}", label="Irregular"
    )
    test_curvewise_to_basis[k].plot(
        axes=axes, color=f"C{k}", linestyle=":",
        label="Curve-wise conversion",
    )
    test_converted[k].plot(
        axes=axes, color=f"C{k}", linestyle="--",
        label="Mixed-effects conversion",
    )
    test_original[k].plot(
        axes=axes, color=f"C{k}", alpha=0.5,
        label="Original basis representation",
    )
    axes.legend(bbox_to_anchor=(1., 1.))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# %%
# Finally, we make use of the :math:`R^2` score and the :math:`MSE` to compare
# the converted basis representations with the original data, both for the
# train and test sets.
score_functions = {"R^2": r2_score, "MSE": mean_squared_error}
scores = {
    score_name: pd.DataFrame({
        "Mixed-effects": {
            "Train": score_fun(train_original, train_converted),
            "Test": score_fun(test_original, test_converted),
        },
        "Curve-wise": {
            "Train": score_fun(train_original, train_curvewise_to_basis),
            "Test": score_fun(test_original, test_curvewise_to_basis),
        },
    })
    for score_name, score_fun in score_functions.items()
}
for score_name, score_df in scores.items():
    print("-" * 35)
    print(f"{score_name} scores:")
    print("-" * 35)
    print(score_df, end=f"\n\n\n")

# %%
# References
# ----------
#
# .. footbibliography::
