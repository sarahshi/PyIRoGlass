# %%  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 9/28/19

"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_LA
import math
import scipy.linalg as LA
from scipy.spatial import ConvexHull
from scipy.linalg import solveh_banded
import pandas as pd


# def baseline_als(y, lam, p, niter=10):
#     """
#     Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005 implemented on stackoverflow by user: sparrowcide
#     https://stackoverflow.com/questions/29156532/python-baseline-correction-library
#     """
#     L = len(y)
#     D = sparse.csc_matrix(np.diff(np.eye(L), 2))
#     w = np.ones(L)
#     for i in range(niter):
#         W = sparse.spdiags(w, 0, L, L)
#         Z = W + lam * D.dot(D.transpose())
#         # Z = W + lam * np.dot(D,D.T)
#         z = sp_LA.spsolve(Z, w * y)
#         w = p * (y > z) + (1 - p) * (y < z)
#     return z


# def baseline_polynomial(y, deg=3, max_it=100, tol=1e-3):
#     # Baseline function from PeakUtils 1.1.0 by Lucas Hermann Negri
#     """
#     Baseline function from PeakUtils 1.1.0 by Lucas Hermann Negri
#     Computes the baseline of a given data.
#     Iteratively performs a polynomial fitting in the data to detect its
#     baseline. At every iteration, the fitting weights on the regions with
#     peaks are reduced to identify the baseline only.
#     Parameters
#     ----------
#     y : ndarray
#         Data to detect the baseline.
#     deg : int
#         Degree of the polynomial that will estimate the data baseline. A low
#         degree may fail to detect all the baseline present, while a high
#         degree may make the data too oscillatory, especially at the edges.
#     max_it : int
#         Maximum number of iterations to perform.
#     tol : float
#         Tolerance to use when comparing the difference between the current
#         fit coefficient and the ones from the last iteration. The iteration
#         procedure will stop when the difference between them is lower than
#         *tol*.
#     Returns
#     -------
#     ndarray
#         Array with the baseline amplitude for every original point in *y*
#     """
#     order = deg + 1
#     coeffs = np.ones(order)

#     # try to avoid numerical issues
#     # cond = math.pow(y.max(), 1. / order)
#     cond = np.power(y.max(), 1.0 / order)
#     x = np.linspace(0.0, cond, y.size)
#     base = y.copy()

#     vander = np.vander(x, order)
#     vander_pinv = LA.pinv2(vander)

#     for _ in range(max_it):
#         coeffs_new = np.dot(vander_pinv, y)

#         if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
#             break

#         coeffs = coeffs_new
#         base = np.dot(vander, coeffs)
#         y = np.minimum(y, base)

#     return base


# def select_wn(wn, abs, wn_low, wn_high):

#     spectrum = prof.spectra[15]
#     wn_low, wn_high = 2800, 4000
#     idx_high = (np.abs(wn - wn_high)).argmin()
#     idx_low = (np.abs(wn - wn_low)).argmin()

#     abs_range = abs[idx_low:idx_high]
#     wn_range = wn[idx_low:idx_high]
#     return wn_range, abs_range

#     # Possible Filters
#     # abs_range_med = scipy.signal.medfilt(abs_range, 5)
#     # abs_range_hat = scipy.signal.savgol_filter(abs_range_med, filter_window, filter_order)


# # bsl = baseline_als(abs_full_range, 1*10**4, 10**-3.5, niter=1000)

# # %%


# def rubberband(x, y):
#     """
#     Rubber band baseline from
#     # Find the convex hull R Kiselev on stack overflow
#     https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
#     """
#     v = ConvexHull(np.array(list(zip(x, y)))).vertices
#     # Rotate convex hull vertices until they start from the lowest one
#     v = np.roll(v, -v.argmin())
#     # Leave only the ascending part
#     v = v[: v.argmax()]

#     # Create baseline using linear interpolation between vertices
#     return np.interp(x, x[v], y[v])


#

# %%

def als_baseline(intensities,
    asymmetry_param=0.05, smoothness_param=5e5,
    max_iters=10, conv_thresh=1e-5, verbose=False):
    """ Computes the asymmetric least squares baseline.
    http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    Smoothness_param: Relative importance of smoothness of the predicted response.
    Asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
    Setting p=1 is effectively a hinge loss. """

    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p to be concise.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])
    
    for i in range(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p * mask + (1 - p) * (~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print(i + 1, conv)
        if conv < conv_thresh:
            break
        w = new_w
    
    else:
        print("ALS did not converge in %d iterations" % max_iters)
    
    return z


class WhittakerSmoother(object):

    def __init__(self, signal, smoothness_param, deriv_order=1):

        self.y = signal
        assert deriv_order > 0, "deriv_order must be an int > 0"
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), "constant")
            for i in range(1, k + 1)])
            
        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1 :] = ds[::-1][: i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)
