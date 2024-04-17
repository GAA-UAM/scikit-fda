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
from sklearn.model_selection import train_test_split

from skfda import FDataBasis
from skfda.datasets import irregular_sample
from skfda.representation.basis import BSplineBasis
from skfda.representation.conversion import EMMixedEffectsConverter
from skfda.misc.scoring import r2_score, mean_squared_error


# %%
# For this example, we are going to simulate the irregular sampling of a
# dataset following the mixed effects model, to later attempt to reconstruct
# the original data.
#
# First, we generate the original basis representation of the data following
# the mixed effects model for irregular data as presented by
# :footcite:t:`james_2018_sparsenessfda`. This just means that
# the coefficients of the basis representation are generated from a Gaussian
# distribution.
n_curves = 50
n_basis = 4
domain_range = (0, 10)
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

# %%
# To visualize the conversion results, we plot the first 8 original and
# converted curves of the test set. On the background, we plot the train set.
fig = plt.figure(figsize=(10, 15))
for k in range(8):
    axes = plt.subplot(4, 2, k + 1)

    train_original.plot(axes=axes, color=(0, 0, 0, 0.05))
    train_irregular.scatter(axes=axes, color=(0, 0, 0, 0.05), marker=".")

    test_converted[k].plot(
        axes=axes, color=f"C{k}", label="Converted",
    )
    test_original[k].plot(
        axes=axes, color=f"C{k}", linestyle="--", label="Original",
    )
    test_irregular[k].scatter(
        axes=axes, color=f"C{k}", label="Irregular"
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
print(f"R2 score (train): {train_r2_score:.2f}")
print(f"R2 score (test): {test_r2_score:.2f}")
print(f"Mean Squared Error (train): {train_mse:.2f}")
print(f"Mean Squared Error (test): {test_mse:.2f}")

# %%
# References
# ----------
#
# .. footbibliography::
