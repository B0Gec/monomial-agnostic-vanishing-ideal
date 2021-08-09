import jax.numpy as np
from jax.scipy.linalg import eigh
from jax import jit, partial
from sympy.polys.orderings import monomial_key

@jit
def blow(A, B):  
    # A = F1 = {p1(X), p2, ...}, B = F2 = {q1, q2, ...}
    # output: {p1(X)*q1(X), p1*q2, ....}
    A, B = np.asarray(A), np.asarray(B)
    n1, n2 = A.shape[-1], B.shape[-1]
    C = np.repeat(A, n2, axis=-1) * np.tile(B, n1)
    return C

@jit
def dblow(A, B, dA, dB):
    A, B, dA, dB = np.asarray(A), np.asarray(B), np.asarray(dA), np.asarray(dB)
    n1, n2 = A.shape[1], B.shape[1]
    ndims = dA.shape[0]//A.shape[0]

    C = np.repeat(A, n2, axis=-1) * np.tile(B, n1)
    dC1 = np.repeat(np.repeat(A, ndims, axis=0), n2, axis=-1) * np.tile(dB, n1)
    dC2 = np.repeat(dA, n2, axis=-1) * np.tile(np.repeat(B, ndims, axis=0), n1)
    dC = dC1 + dC2 
    return C, dC

## extract residual components and projection operator
@jit
def pres(C, F):
    L = np.linalg.lstsq(F, C, rcond=None)[0]
    res = C - F @ L     # by definition, res == torch.hstack([F, C]) @ L
    return res, L

## project C to residual space
@jit
def res(C, F, L):
    return C - F @ L

@jit
def matrixfact(C):
    _, d, Vt = np.linalg.svd(C, full_matrices=True)
    d = np.append(d, np.zeros(Vt.shape[0] - len(d)))
    return d, Vt.T

@jit
def indirect_ged(A, B, gamma=1e-9):
    '''
    Reduce GED to two SVD
    '''
    Vb, db, _ = np.linalg.svd(B)
    db += gamma 
    Vb_ = Vb / (db**0.5)

    A_ = Vb_.T @ A @ Vb_
    Va, d, _ = np.linalg.svd(A_)
    V = Vb_ @ Va

    return d, V


def matrixfact_gep(C, N, gamma=1e-9):
    '''
    Unfortunately, jax.scipy.linalg.eigh is the only way of solving generalied eigenvalue problem but it is not implmented yet.
    Instead of directly solving generalized eigenvalue problem, we reduce the problem to an eigenvalue problem
    '''
    A = C.T @ C
    B = N.T @ N
    r = np.linalg.matrix_rank(B, gamma)
    gamma_ = np.mean(np.diag(B))*gamma
    d, V = indirect_ged(A, B, gamma=gamma_)
    d = np.sqrt(np.abs(d))

    gnorms = np.diag(V.T@B@V)
    valid = np.argsort(-gnorms)[:r]

    d, V = d[valid], V[:, valid]
    gnorms = gnorms[valid]
   
    perm = np.argsort(-d)

    return d[perm], V[:, perm]

def argsort(arr, key=None, reverse=False):
    arr_ = enumerate(arr)
    arr_ = sorted(arr_, key=lambda x: key(x[1]), reverse=reverse)
    perm = list(zip(*arr_))[0]
    return list(perm)
