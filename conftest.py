#https://github.com/scikit-learn/scikit-learn/issues/8959
import numpy as np
try:
    np.set_printoptions(sign=' ')
except TypeError:
    pass

collect_ignore = ['setup.py']
