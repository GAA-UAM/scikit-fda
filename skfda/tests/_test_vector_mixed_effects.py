# %%
import numpy as np
from skfda.representation.basis import MonomialBasis
from skfda.representation.basis import VectorValuedBasis
import matplotlib.pyplot as plt
# %%
m2 = MonomialBasis(n_basis=2, domain_range=(0, 10))
m3 = MonomialBasis(n_basis=3, domain_range=(0, 10))
m3.plot()
plt.show()

# %%
# m2.plot()
vbasis = VectorValuedBasis([m2, m3])
vbasis.plot()
plt.show()

# %%
vbasis(2)