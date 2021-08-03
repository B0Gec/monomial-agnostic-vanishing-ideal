import numpy as np
from scipy import linalg
from sympy.polys.orderings import monomial_key


def blow(A, B):  
    # A = F1 = {p1(X), p2, ...}, B = F2 = {q1, q2, ...}
    # output: {p1(X)*q1(X), p1*q2, ....}
    A, B = np.asarray(A), np.asarray(B)
    n1, n2 = A.shape[-1], B.shape[-1]
    C = np.repeat(A, n2, axis=-1) * np.tile(B, n1)
    return C

def dblow(A, B, dA, dB):
    A, B, dA, dB = np.asarray(A), np.asarray(B), np.asarray(dA), np.asarray(dB)
    n1, n2 = A.shape[1], B.shape[1]
    ndims = np.int(dA.shape[0]/A.shape[0])

    C = np.repeat(A, n2, axis=-1) * np.tile(B, n1)
    dC1 = np.repeat(np.repeat(A, ndims, axis=0), n2, axis=-1) * np.tile(dB, n1)
    dC2 = np.repeat(dA, n2, axis=-1) * np.tile(np.repeat(B, ndims, axis=0), n1)
    dC = dC1 + dC2 
    # dC = (np.repeat(np.repeat(A, ndims, axis=0), n2, axis=1) * np.tile(dB, n1) 
    #       + np.repeat(dA, n2, axis=1) * np.tile(np.repeat(B, ndims, axis=0), n1))
    return C, dC

## extract residual components and projection operator
def pres(C, F):
    L = np.linalg.lstsq(F, C, rcond=None)[0]
    resop = np.vstack([-L, np.identity(C.shape[1])])
    res = C - F @ L     # by definition, res == np.hstack([F, C]) @ resop
    # [F C] * [I -L]
    
    return res, resop

## project C to residual space
def res(C, F, R):
    return np.hstack([F, C]) @ R

def matrixfact(C):
    _, d, Vt = np.linalg.svd(C, full_matrices=True)
    d = np.append(d, np.zeros(Vt.shape[0] - len(d)))
    return d, Vt.T

# def matrixfact_gep(C, N, gamma=1e-9):
def matrixfact_gep(C, N, gamma=1e-9):
    # A = Symmetric(C.T@C)
    # B = Symmetric(N.T@N)
    
    # c = np.mean(C, axis=0)
    # C = C - np.mean(C, axis=0)

    A = C.T @ C
    B = N.T @ N
    r = np.linalg.matrix_rank(B, gamma)
    gamma_ = np.mean(np.diag(B))*gamma
    d, V = linalg.eigh(A, B+gamma_*np.identity(B.shape[0]))
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
