"""
Pairwise alignment
==================

Shows the usage of the elastic registration to perform a pairwise alignment.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 5


import fda
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


###############################################################################
# Given any two functions :math:`f` and :math:`g`, we define their
# pairwise alignment or  registration to be the problem of finding a warping
# function :math:`\gamma` such that a certain energy term
# :math:`E[f, g \circ \gamma]` is minimized. That is, we solve for:
#
# .. math::
#   \gamma^*= *{argmin}_{\gamma \in \Gamma}` E[f \circ \gamma, g]
#
# In the case of elastic registration it is taken as energy function the
# squared Fisher-Rao distance with a penalisation term, due to the property of
# invariance to reparameterizations of warpings functions.[1]
#
# Firstly, we will create two unimodal samples, :math:`f` and :math:`g`,
# defined in [-1, 1] wich will be used to show the elastic registration.
#

# Samples with modes in 1/3 and 2/3
fd = fda.datasets.make_multimodal_samples(n_samples=2, modes_location=[1/3,2/3],
                                          random_state=1, start=0, mode_std=.01)

fd.plot()
plt.legend(['$f$', '$g$'])

###############################################################################
#
# In this case :math:`g` it is used as template and :math:`f` is aligned to
# :math:`g`. In the following figure it is shown the result of register
# :math:`f` to :math:`g`, wich can be computed using :func:`elastic_registration
# <fda.registration.elastic_registration>`.
#


f, g = fd[0], fd[1]

# Aligns f to g
fd_align = fda.registration.elastic_registration(f, g)


plt.figure()

fd.plot()
fd_align.plot(color='C0', linestyle='--')


plt.legend(['$f$', '$g$', '$f \\circ h $'])

###############################################################################
# TODO
#

# Warping to align f to g
warping = fda.registration.elastic_registration_warping(f, g)


plt.figure()

t = np.linspace(0,1)
plt.plot(t,t)

warping.plot(color='C0', linestyle='--')

plt.legend(['$h_{id}$', '$h$'])

###############################################################################
# TODO
#


warping_inverse = fda.registration.invert_warping(warping)


plt.figure()

fd.plot(label='$f$')
g.compose(warping_inverse).plot(color='C1', linestyle='--')


plt.legend(['$f$', '$g$', '$g \\circ h^{-1} $'])


###############################################################################
# TODO
#



# Values of lambda
lambdas = np.linspace(0, .2, 20)

# Creation of a color gradient
cmap = clr.LinearSegmentedColormap.from_list('custom cmap', ['C1','C0'])

color = cmap(.2 + 3*lambdas)

plt.figure()

for lam, c in zip(lambdas, color):
    # Plots result of alignment
    fda.registration.elastic_registration(f, g, lam=lam).plot(color=c)


l1, = f.plot(color='C0', linewidth=2., label='$f$')
l2, = g.plot(color='C1', linewidth=2., label='$g$')

plt.legend(handles=[l1, l2])



###############################################################################
# TODO
#

plt.figure()

for lam, c in zip(lambdas, color):
    fda.registration.elastic_registration_warping(f, g, lam=lam).plot(color=c)

# Plots identity
plt.plot(t,t,  color='C0', linestyle="--")


###############################################################################
# TODO
#

state = np.random.RandomState(0)

location1 = state.normal(loc=-.3, scale=.1, size=3)
fd = fda.datasets.make_multimodal_samples(n_samples=3, modes_location=location1,
                                          noise=.001 ,random_state=1)

location2 = state.normal(loc=.3, scale=.1, size=3)
g = fda.datasets.make_multimodal_samples(n_samples=3, modes_location=location2,
                                           random_state=2)





plt.figure()

l1, *_ = fd.plot(color="C0", label="$f_i$")
l2, *_ = g.plot(color="C1", label="$g_i$")

plt.legend(handles=[l1, l2])

###############################################################################
# TODO
#


plt.figure()

fd_registered = fda.registration.elastic_registration(fd, g)

l1, *_ = fd.plot(color="C0", label="$f_i$")
l2, *_ = g.plot(color="C1", label="$g_i$")
l3, *_ = fd_registered.plot(color="C0", linestyle="--", label="$f_i\\circ h_i$")


plt.legend(handles=[l1, l2, l3])
plt.show()


###############################################################################
# TODO
#
