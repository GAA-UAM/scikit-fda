# cython: language_level=3
cimport cDP
import numpy as np
from numpy.linalg import norm

cimport numpy as np
from cpython cimport array

def coptimum_reparam_fN(np.ndarray[double, ndim=1, mode="c"] mf, np.ndarray[double, ndim=1, mode="c"] time,
                      np.ndarray[double, ndim=2, mode="c"] f, lam1=0.0):
    """
    cython interface calculates the warping to align a set of functions f to a single function mf

    :param mf: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f: numpy ndarray of shape (M,N) of N functions with M samples
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of f with mf

    """
    cdef int M, N, n1, disp
    cdef double lam
    mf = mf / norm(mf)
    M, N = f.shape[0], f.shape[1]
    n1 = 1
    disp = 0
    lam = lam1
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] fi = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        fi = f[:, k] / norm(f[:, k])
        fi = np.ascontiguousarray(fi)

        cDP.DP(&fi[0], &mf[0], &n1, &M, &lam, &disp, &gami[0])
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_fN2(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                       np.ndarray[double, ndim=2, mode="c"] f2, lam1=0.0):
    """
    cython interface calculates the warping to align a set of functions f1 to another set of functions f2

    :param f1: numpy ndarray of shape (M,N) of M srsfs with N samples
    :param time: vector of size N describing the sample points
    :param f2: numpy ndarray of shape (M,N) of M srsfs with N samples
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype numpy ndarray
    :return gam: describing the warping functions used to align columns of f with mf

    """
    cdef int M, N, n1, disp
    cdef double lam

    M, N = f1.shape[0], f1.shape[1]
    n1 = 1
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        f1i = f1[:, k] / norm(f1[:, k])
        f2i = f2[:, k] / norm(f2[:, k])
        f1i = np.ascontiguousarray(f1i)
        f2i = np.ascontiguousarray(f2i)

        cDP.DP(&f2i[0], &f1i[0], &n1, &M, &lam, &disp, &gami[0])
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_f(np.ndarray[double, ndim=1, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=1, mode="c"] f2, lam1=0.0):
    """
    cython interface for calculates the warping to align functions f2 tof1

    :param f1: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f2: vector of size N samples of second function
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, n1, disp
    cdef double lam
    M = f1.shape[0]
    n1 = 1
    lam = lam1
    disp = 0
    f1 = f1 / norm(f1)
    f2 = f2 / norm(f2)
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)

    cDP.DP(&f2[0], &f1[0], &n1, &M, &lam, &disp, &gami[0])
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_fN2_pair(np.ndarray[double, ndim=2, mode="c"] f, np.ndarray[double, ndim=1, mode="c"] time,
                            np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=2, mode="c"] f2, lam1=0.0):
    """
    cython interface for calculates the warping to align paired function f1 and f2 to f

    :param f: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f1: vector of size N samples of second function
    :param f2: vector of size N samples of second function
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, N, n1, disp
    n1 = 2
    cdef double lam
    M, N = f1.shape[0], f1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * n1)

    gam = np.zeros((M, N))
    for k in xrange(0, N):
        f1i = f.reshape(M*n1)
        f2tmp = np.column_stack((f1[:, k], f2[:, k]))
        f2i = f2tmp.reshape(M*n1)

        f1i = np.ascontiguousarray(f1i)
        f2i = np.ascontiguousarray(f2i)

        cDP.DP(&f2i[0], &f1i[0], &n1, &M, &lam, &disp, &gami[0])
        gam[:, k] = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_pair_f(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                          np.ndarray[double, ndim=2, mode="c"] f2, lam1=0.0):
    """
    cython interface for calculates the warping to align paired srsf q2 to q1

    :param f1: vector of size N samples of first function
    :param time: vector of size N describing the sample points
    :param f2: vector of size N samples of second function
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, N, disp
    cdef double lam
    M, N = f1.shape[0], f1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * N)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * N)

    sizes = np.zeros(1, dtype=np.int32)
    f1i = f1.reshape(M*N)
    f2i = f2.reshape(M*N)

    f1i = np.ascontiguousarray(f1i)
    f2i = np.ascontiguousarray(f2i)

    cDP.DP(&f2i[0], &f1i[0], &N, &M, &lam, &disp, &gami[0])
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam

def coptimum_reparam_curve_f(np.ndarray[double, ndim=2, mode="c"] f1, np.ndarray[double, ndim=1, mode="c"] time,
                     np.ndarray[double, ndim=2, mode="c"] f2, lam1=0.0):
    """
    cython interface for calculates the warping to align curve f2 to f1

    :param f1: matrix of size nxN samples of first SRVF
    :param time: vector of size N describing the sample points
    :param f2: matrix of size nxN samples of second SRVF
    :param lam1: controls the amount of elasticity (default = 0.0)

    :rtype vector
    :return gam: describing the warping function used to align f2 with f1
    """
    cdef int M, n1, disp
    cdef double lam
    n1 = f1.shape[0]
    M = f1.shape[1]
    lam = lam1
    disp = 0
    cdef np.ndarray[double, ndim=1, mode="c"] gami = np.zeros(M)
    cdef np.ndarray[double, ndim=1, mode="c"] f1i = np.zeros(M * n1)
    cdef np.ndarray[double, ndim=1, mode="c"] f2i = np.zeros(M * n1)

    f1i = f1.reshape(M*n1, order='F')
    f2i = f2.reshape(M*n1, order='F')

    f1i = np.ascontiguousarray(f1i)
    f2i = np.ascontiguousarray(f2i)

    cDP.DP(&f2i[0], &f1i[0], &n1, &M, &lam, &disp, &gami[0])
    gam = (gami - gami[0]) / (gami[-1] - gami[0])

    return gam
