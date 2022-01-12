"""
cython implementation of soft-DTW recursion.
Refs:
    - Blondel, M., Mensch, A., & Vert, J.-P. (2021).
    Differentiable Divergences Between Time Series. AISTATS. http://arxiv.org/abs/2010.08354
    - Mensch, A.,  Blondel, M. (2018).
    Differentiable Dynamic Programming for Structured Prediction and Attention. ICML. https://proceedings.mlr.press/v80/mensch18a.html

@author: Clément Lejeune <clementlej@gmail.com>
"""
cimport numpy as np
import numpy as np
from cython cimport boundscheck, wraparound, cdivision
from libc.math cimport exp, log
from libc.float cimport DBL_MAX

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cdef inline DTYPE_t _soft_min_argmin(
      DTYPE_t x, DTYPE_t y, DTYPE_t z) nogil:

    cdef DTYPE_t min_xyz = min(x, min(y, z))
    # cdef double e_x
    # cdef double e_y
    # cdef double e_z
    cdef DTYPE_t nn
    # cdef DTYPE_t soft_min

    # e_x = exp(min_xyz - x)
    # e_y = exp(min_xyz - y)
    # e_z = exp(min_xyz - z)
    # nn = e_x + e_y + e_z # normalizing constant

    nn = exp(min_xyz - x)
    nn += exp(min_xyz - y)
    nn += exp(min_xyz - z)
    # soft_min = min_xyz - log(nn) # smoothed_min operator value

    return min_xyz - log(nn)# soft_min#, e_x/nn, e_y/nn, e_z/nn

@cdivision(True)
@boundscheck(False)
@wraparound(False) 
def _sdtw_C_cy(
    np.ndarray[DTYPE_t, ndim=2] C,
    DTYPE_t gamma):

    dtype = DTYPE

    cdef Py_ssize_t i, j
    cdef Py_ssize_t len_X = C.shape[0]
    cdef Py_ssize_t len_Y = C.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] V = np.zeros((len_X + 1, len_Y + 1), dtype=dtype)

    # this is useless:
    # cdef double[:, :, :] P = np.zeros((len_X + 2, len_Y + 2, 3), dtype=dtype)
    
    cdef DTYPE_t soft_min = 0.0

    with nogil:

        # initialize firt row to a large value
        for j in range(1, len_Y + 1):
            V[0, j] = DBL_MAX
        
        # initilize first column to a large value
        for i in range(1, len_X + 1):
            V[i, 0] = DBL_MAX

        for i in range(1, len_X + 1):
            for j in range(1, len_Y + 1):
                # P_ij are useless (unless need for gradient computation)
    ##                soft_min, P[i, j, 0], P[i, j, 1], P[i, j, 2] = _soft_min_argmin(
    ##                                                                    V[i, j-1],
    ##                                                                    V[i-1, j-1],
    ##                                                                    V[i-1, j])
                soft_min = _soft_min_argmin(V[i, j-1], V[i-1, j-1], V[i-1, j])
                if gamma != 1.0:
                    V[i, j] = (C[i-1, j-1] / gamma) + soft_min
                else:
                    V[i, j] = C[i-1, j-1] + soft_min
                
    return gamma * V[len_X, len_Y]
