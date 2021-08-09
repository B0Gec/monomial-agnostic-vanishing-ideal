import jax.numpy as np 
from copy import deepcopy
from mavi.jax.util.util import blow, dblow

def evaluate(basis, X, target='vanishing'):
    if target == 'nonvanishing':
        return _evaluate_nv(basis, X)
    elif target == 'vanishing':
        return _evaluate_v(basis, X)
    else:
        print('unknown mode: %s' % target)
        exit()


def _evaluate_nv(B, X, device='cpu'):
    F = B.nonvanishings()
    
    N = X.shape[0]

    Z0 = F[0].eval(np.ones((N, 1)))
    if len(F) == 1: return Z0
    
    Z1 = F[1].eval(Z0, X)
    Z = np.hstack([Z0, Z1])

    Zt = np.array(Z1)

    for t in range(2, len(F)):
        C = blow(Z1, Zt)
        Zt = F[t].eval(Z, C)
        Z = np.hstack([Z, Zt])

    return Z


def _evaluate_v(B, X, device='cpu'):
    F = B.nonvanishings()
    G = B.vanishings()

    N = X.shape[0]

    ZF0 = F[0].eval(np.ones((N, 1)))
    ZF1 = F[1].eval(ZF0, X)
    Z1 = G[1].eval(ZF0, X)

    ZF = np.hstack([ZF0, ZF1])
    Z = np.array(Z1)
    # print([f.shape for f in F[1:]])
    ZFt = np.array(ZF1)
    for t in range(2, len(F)):
        C = blow(ZF1, ZFt)
        Zt = G[t].eval(ZF, C)
        ZFt = F[t].eval(ZF, C)
        ZF = np.hstack([ZF, ZFt])
        Z = np.hstack([Z, Zt])

    return Z


def gradient(basis, X, target='vanishing'):
    if target == 'nonvanishing':
        return _gradient_nv(basis, X)
    elif target == 'vanishing':
        return _gradient_v(basis, X)        
    else:
        print('unknown mode %s' % target)

def _gradient_nv(B, X):
    F = B.nonvanishings()
    npoints, ndims = X.shape

    Z0 = F[0].eval(np.ones((npoints, 1)))
    dZ0 = np.zeros((npoints*ndims, 1))
    if len(F) == 1: 
        return dZ0

    Z1 = F[1].eval(Z0, X)
    dZ1 = np.tile(F[1].V, (npoints, 1))
    Z, dZ =  np.hstack((Z0, Z1)), np.hstack((dZ0, dZ1))
    Zt, dZt = deepcopy(Z1), deepcopy(dZ1)

    for t in range(2,len(F)):
        C, dC = dblow(Z1, Zt, dZ1, dZt)
        Zt = F[t].eval(Z, C)
        dZt  = F[t].eval(dZ, dC)
        Z, dZ = np.hstack((Z, Zt)), np.hstack((dZ, dZt))

    return dZ

def _gradient_v(B, X):
    F = B.nonvanishings()
    G = B.vanishings()
    npoints, ndims = X.shape

    ZF0 = F[0].eval(np.ones((npoints, 1)))
    dZF0 = np.zeros((npoints*ndims, 1))
    if len(F) == 1: 
        return dZF0
    
    ZF1 = F[1].eval(ZF0, X)
    dZF1 = np.tile(F[1].V, (npoints, 1))
    ZF, dZF =  np.hstack((ZF0, ZF1)), np.hstack((dZF0, dZF1))
    ZFt, dZFt = deepcopy(ZF1), deepcopy(dZF1)

    dZ = np.tile(G[1].V, (npoints, 1))

    for t in range(2,len(F)):
        C, dC = dblow(ZF1, ZFt, dZF1, dZFt)
        ZFt = F[t].eval(ZF, C)
        dZFt  = F[t].eval(dZF, dC)
        dZt  = G[t].eval(dZF, dC)
        ZF, dZF = np.hstack((ZF, ZFt)), np.hstack((dZF, dZFt))
        dZ = np.hstack((dZ, dZt))
    return dZ

