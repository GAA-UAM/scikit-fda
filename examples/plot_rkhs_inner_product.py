"""
Reproducing Kernel Hilbert Space Inner Product for the Brownian Bridge
=======================================================================

This example shows how to compute the inner product of two functions in the
reproducing kernel Hilbert space (RKHS) of the Brownian Bridge.
"""

# Author: Martín Sánchez Signorini
# License: MIT

import numpy as np
import pandas as pd

from skfda.misc.rkhs_product import rkhs_inner_product
from skfda.representation import FDataGrid
from skfda.typing._numpy import NDArrayFloat

###############################################################################
# The kernel corresponding to a Brownian Bridge process
# :footcite:p:`gutierrez++_1992_numerical` in the interval :math:`[0, 1]` is
#
# .. math::
#     k(s, t) = \min(s, t) - st
#
# The RKHS inner product method
# :func:`~skfda.misc.rkhs_product.rkhs_inner_product` requires the kernel to
# be defined as a function of two vector arguments that returns the matrix of
# values of the kernel in the corresponding grid.
# The following function defines the kernel as such a function.


def brownian_bridge_covariance(
    t: NDArrayFloat,
    s: NDArrayFloat,
) -> NDArrayFloat:
    """
    Covariance function of the Brownian Bridge process.

    This function must receive two vectors of points while returning the
    matrix of values of the covariance function in the corresponding grid.
    """
    t_col = t[:, None]
    s_row = s[None, :]
    return np.minimum(t_col, s_row) - t_col * s_row


###############################################################################
# The RKHS of this kernel :footcite:p:`berlinet+thomas-agnan_2011_reproducing`
# is the set of functions
#
# .. math::
#     f: [0, 1] \to \mathbb{R} \quad \text{ such that }
#     f \text{ is absolutely continuous, }
#     f(0) = f(1) = 0 \text{ and }
#     f' \in L^2([0, 1]).
#
# For this example we will be using the following functions in this RKHS:
#
# .. math::
#     \begin{align}
#     f(t) &= 1 - (2t - 1)^2 \\
#     g(t) &= \sin(\pi t)
#     \end{align}
#
# The following code defines a method to compute the inner product of these
# two functions in the RKHS of the Brownian Bridge, using a variable number of
# points of discretization of the functions.

def brownian_bridge_rkhs_inner_product(
    num_points: int,
) -> float:
    """Inner product of two functions in the RKHS of the Brownian Bridge."""
    # Define the functions
    # Remove first and last points to avoid a singular covariance matrix
    grid_points = np.linspace(0, 1, num_points + 2)[1:-1]
    f = FDataGrid(
        [1 - (2 * grid_points - 1)**2],
        grid_points,
    )
    g = FDataGrid(
        [np.sin(np.pi * grid_points)],
        grid_points,
    )

    # Compute the inner product
    return rkhs_inner_product(  # type: ignore[no-any-return]
        fdata1=f,
        fdata2=g,
        cov_function=brownian_bridge_covariance,
    )[0]


# Plot the functions f and g in the same figure
plt = FDataGrid(
    np.concatenate(
        [
            1 - (2 * np.linspace(0, 1, 100) - 1)**2,
            np.sin(np.pi * np.linspace(0, 1, 100)),
        ],
        axis=0,
    ).reshape(2, 100),
    np.linspace(0, 1, 100),
).plot()

plt.show()

###############################################################################
# The inner product of two functions :math:`f, g` in this RKHS
# :footcite:p:`berlinet+thomas-agnan_2011_reproducing` is
#
# .. math::
#     \langle f, g \rangle = \int_0^1 f'(t) g'(t) dt.
#
# Therefore, the exact value of the product of these two functions in the RKHS
# of the Brownian Bridge can be explicitly calculated.
# First, we have that their derivatives are
#
# .. math::
#     \begin{align}
#     f'(t) &= 4(1 - 2t) \\
#     g'(t) &= \pi \cos(\pi t)
#     \end{align}
#
# Then, the inner product in :math:`L^2([0, 1])` of these derivatives is
#
# .. math::
#     \begin{align}
#     \langle f', g' \rangle &= \int_0^1 f'(t) g'(t) dt \\
#                            &= \int_0^1 4(1 - 2t) \pi \cos(\pi t) dt \\
#                            &= \frac{16}{\pi}.
#     \end{align}
#
# Which is the exact value of their inner product in the RKHS of the Brownian
# Bridge.
# Thus, we measure the difference between the exact value and the value
# computed by the method :func:`~skfda.misc.rkhs_product.rkhs_inner_product`
# for increasing numbers of discretization points.
# In particular, we will be using from 500 to 10000 points with a step of 500.
#
# The following code computes the inner product for each number of points and
# plots the difference between the exact value and the computed value.

num_points_list = np.arange(
    start=500,
    stop=10001,
    step=500,
)
expected_value = 16 / np.pi

errors_df = pd.DataFrame(
    columns=["Number of points of discretization", "Absolute error"],
)

for num_points in num_points_list:
    computed_value = brownian_bridge_rkhs_inner_product(num_points)
    error = np.abs(computed_value - expected_value)

    # Add new row to the dataframe
    errors_df.loc[len(errors_df)] = [num_points, error]

# Plot the errors
errors_df.plot(
    x="Number of points of discretization",
    y="Absolute error",
    title="Absolute error of the inner product",
    xlabel="Number of points of discretization",
    ylabel="Absolute error",
)


###############################################################################
# The following code plots the errors using a logarithmic scale in the y-axis.

errors_df.plot(
    x="Number of points of discretization",
    y="Absolute error",
    title="Absolute error of the inner product",
    xlabel="Number of points of discretization",
    ylabel="Absolute error",
    logy=True,
)

###############################################################################
# This example shows the convergence of the method
# :func:`~skfda.misc.rkhs_product.rkhs_inner_product` for the Brownian Bridge
# kernel, while also showing how to apply this method using a custom covariance
# function.

###############################################################################
# **References:**
#     .. footbibliography::
