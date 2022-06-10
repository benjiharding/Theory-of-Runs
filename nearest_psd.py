import numpy as np

""""Calculate the nearest positive semi-defininte correlation matrix 
based on the algorthim in [1]. Python implementation from [2].

[1] Nicholas J. Higham, Computing the Nearest Correlation 
Matrix—A Problem from Finance, IMA J. Numer. Anal. 22, 
329-343, 2002.

[2] https://stackoverflow.com/a/10940283/11842469
"""


# def _getAplus(A):
#     eigval, eigvec = np.linalg.eig(A)
#     Q = np.matrix(eigvec)
#     xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
#     return Q * xdiag * Q.T


# def _getPs(A, W=None):
#     W05 = np.matrix(W ** 0.5)
#     return W05.I * _getAplus(W05 * A * W05) * W05.I


# def _getPu(A, W=None):
#     Aret = np.array(A.copy())
#     Aret[W > 0] = np.array(W)[W > 0]
#     return np.matrix(Aret)


# def nearPD(A, nit=10):
#     n = A.shape[0]
#     W = np.identity(n)
#     # W is the matrix used for the norm (assumed to be Identity matrix here)
#     # the algorithm should work for any diagonal W
#     deltaS = 0
#     Yk = A.copy()
#     for k in range(nit):
#         Rk = Yk - deltaS
#         Xk = _getPs(Rk, W=W)
#         deltaS = Xk - Rk
#         Yk = _getPu(Xk, W=W)
#     return Yk


# def isPD(x):
#     return np.all(np.linalg.eigvals(x) >= 0)


def nearPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
