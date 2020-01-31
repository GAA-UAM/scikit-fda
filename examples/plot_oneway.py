"""
One-way functional ANOVA
========================

This example shows how to perform a functional one-way ANOVA test.
"""

# Author: David García Fernández
# License: MIT

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import skfda
from skfda.inference.anova import func_oneway

################################################################################
# TODO
#

dataset = skfda.datasets.fetch_aemet()

y = dataset['meta']
fd = dataset['data'][0]
meta_names = dataset['meta_names']

province = y[:, np.asarray(meta_names) == 'province'].ravel()

fig = fd.plot(group=province)

################################################################################
# TODO

sel_prov = ['A CORUÑA', 'BALEARES', 'LAS PALMAS']

filt = np.logical_or.reduce([np.asarray(province) == p for p in sel_prov])

province = province[filt]
fd = fd[filt]

fig = fd.plot(group=province, legend=True)

##############################################################################
# TODO


fd_groups = [fd.copy(data_matrix=fd.data_matrix[province == label],
                     dataset_label=fd.dataset_label + ' in ' + label)
             for label in sel_prov]

###############################################################################
# ANOVA
#

func_oneway(*fd_groups)
