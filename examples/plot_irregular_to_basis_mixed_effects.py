"""
Mixed effects model to convert irregular data to basis representation
=======================================================================

Convert irregular data to a basis representation using the mixed effects models
implemented in :class:`skfda.representation.irregular.FDataIrregular` class.
"""
# Author: Pablo Cuesta Sierra
# License: MIT

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from skfda import FDataBasis
from skfda.datasets import irregular_sample
from skfda.representation.basis import BSplineBasis
from skfda.representation.conversion import EMMixedEffectsConverter
from skfda.misc.scoring import r2_score, mean_squared_error

np.random.seed(4934755)  # set the seed for reproducibility


# %%
# For this example, we are going to simulate the irregular sampling of a
# dataset following the mixed effects model, to later attempt to reconstruct
# the original data with an
# :class:`skfda.representation.conversion.MixedEffectsConverter`.
#
# First, we create the original :class:`skfda.representation.basis.FDataBasis`
# object, whose coefficients follow the mixed effects model for irregular data
# as presented in :cite:p:`james_2018_sparsenessfda`. This just means that
# the coefficients are generated from a Gaussian distribution. Our dataset
# will contain 40 curves.
n_basis = 4
domain_range = (0, 10)
basis = BSplineBasis(n_basis=n_basis, domain_range=domain_range, order=4)
basis.plot()
plt.title("Basis functions")

coeff_mean = np.array([-10, 20, -24, 4])
coeff_cov_sqrt = np.random.rand(n_basis, n_basis) * 5
coeff_cov = coeff_cov_sqrt @ coeff_cov_sqrt.T  # ensure positive semidefinite
coefficients = np.random.multivariate_normal(
    mean=coeff_mean, cov=coeff_cov, size=40,
)

fdatabasis_original = FDataBasis(basis, coefficients)
# Plot the first 10 curves
fdatabasis_original[:10].plot()
plt.title("Original curves")
plt.show()


# %%
# Sencondly, we will simulate the irregular sampling of the original data
# with random noise. For each curve, we will sample 4 points from the domain.
fd_irregular = irregular_sample(
    fdatabasis_original, n_points_per_curve=4, noise_stddev=0.2,
)
fig = plt.figure()
fdatabasis_original[-6:].plot(fig=fig)
fd_irregular[-6:].scatter(fig=fig, alpha=0.1)
plt.show()

# %%
# Moreover, we will split our irregular data into two groups, the train curves
# and the test curves. We will use the train curves to fit the mixed effects
# model and the test curves to evaluate the quality of the conversion.
test_original = fdatabasis_original[::2]
train_original = fdatabasis_original[1::2]
test_irregular = fd_irregular[::2]
train_irregular = fd_irregular[1::2]

# %%
# Now, we create and train the mixed effects converter.
converter = EMMixedEffectsConverter(basis)
converter = converter.fit(train_irregular)

# %%
# And convert the irregular data to basis representation.
train_converted = converter.transform(train_irregular)
test_converted = converter.transform(test_irregular)

# %%
# Let's plot the first 8 original and converted curves of the test set.
# On the background, we plot the train set.
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
# Finally, we will use the :math:`R^2` score and the :math:`MSE` to compare
# the converted basis representations with the original data, both for the
# train and test sets.
train_r2_score = r2_score(train_original, train_converted)
test_r2_score = r2_score(test_original, test_converted)
train_mse = mean_squared_error(train_original, train_converted)
test_mse = mean_squared_error(test_original, test_converted)
print(f"Train R2 score: {train_r2_score:.2f}")
print(f"Test R2 score: {test_r2_score:.2f}")
print(f"Train Mean Squared Error: {train_mse:.2f}")
print(f"Test Mean Squared Error: {test_mse:.2f}")

# %%
# References
# ----------
#
# .. footbibliography::
