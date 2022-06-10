import numpy as np
import warnings
from nearest_psd import nearPD, isPD


class LUSimulator1D:
    """Class for unconditional LU simulation of 1D arrays."""

    def __init__(self):
        self.autocov_fn = None
        self.autocov_mat = None

    def lusim(self, x, seed, nreals):
        """LU simulation via Cholesky decomposition of the covariance matrix"""
        rng = np.random.default_rng(seed)
        covmat = self.autocovariance(x)
        L = np.linalg.cholesky(covmat)

        # draw random numbers
        if nreals == 1:
            w = rng.normal(0, 1, covmat.shape[0])
        else:
            w = rng.normal(0, 1, [covmat.shape[0], nreals])

        # correlate with cholesky
        y = L @ w

        return y

    def autocovariance(self, x):
        """variance-covariance matrix from exhaustive 1D image x"""
        x = np.asarray(x)
        if x.ndim > 1:
            if x.shape[-1] > 1:
                raise ValueError("x must be 1 dimensional")
        x = x.ravel()
        nx = len(x)

        # calculate autocovariance
        autocov = np.correlate(x, x, "full")
        self.autocov_fn = autocov / np.max(autocov)
        lag0 = self.autocov_fn.argmax()  # max cov is lag 0

        # calculate autocovariance
        # self.autocov_fn = self.oned_covariance(x)

        # build the upper triangular autocovariance matrix
        Rxx = np.zeros((nx, nx))
        self.autocov_fn = np.insert(self.autocov_fn, -1, 0)
        for i in range(Rxx.shape[0]):
            Rxx[i, i:] = self.autocov_fn[lag0:][: nx - i]

        # # build the upper triangular autocovariance matrix
        # Rxx = np.zeros((nx, nx))
        # self.autocov_fn = np.insert(self.autocov_fn, -1, 0)
        # for i in range(Rxx.shape[0]):
        #     Rxx[i, i:] = self.autocov_fn[: nx - i]

        # make it symmetric
        Rxx = Rxx + Rxx.T - np.diag(np.diag(Rxx))

        # ensure it's positive semi-definite, correct if not
        if not isPD(Rxx):
            warnings.warn(
                "Covariance matrix not positive definite, correcting to nearest PSD"
            )
            Rxx = nearPD(Rxx)
        self.autocov_mat = Rxx

        return Rxx

    def oned_covariance(self, x):
        """Calculate 1D covariance along x"""
        x = np.asarray(x)
        nx = x.shape[0]
        cx = np.zeros(nx)
        for i in range(0, nx - 1):
            z0 = x[0 : nx - i]
            z1 = x[i:nx]
            dz = (z1 * z0) - np.mean(z0) * np.mean(z1)
            cx[i] = np.sum(dz) / (nx - i)
        return cx
