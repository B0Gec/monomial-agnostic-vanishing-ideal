import numpy as np 
from copy import deepcopy
from mavi.numpy.util.util import blow, dblow

def evaluate(basis, X, target='vanishing'):
    if target == 'nonvanishing':
        return _evaluate_nv(basis, X)
    elif target == 'vanishing':
        return _evaluate_v(basis, X)
    else:
        print('unknown mode: %s' % target)
        exit()
    
def _evaluate_nv(B, X):
    F = B.nonvanishings()
    
    N = X.shape[0]

    Z0 = np.ones((N, 1)) * F[0]
    if len(F) == 1: return Z0
    
    Z1 = np.hstack([Z0, X]) @ F[1]
    Z = np.hstack([Z0, Z1])

    Zt = np.array(Z1)

    for t in range(2, len(F)):
        C = blow(Z1, Zt)
        Zt = np.hstack([Z, C]) @ F[t]
        Z = np.hstack([Z, Zt])

    return Z


def _evaluate_v(B, X):
    F = B.nonvanishings()
    G = B.vanishings()
    # if np.all([gt.size==0 for gt in G]):
        # return np.array([])
    if np.all([gt.size==0 for gt in G]):
        return np.zeros((len(X), 0))

    N = X.shape[0]

    ZF0 = np.ones((N, 1)) * F[0]
    ZF1 = np.hstack([ZF0, X]) @ F[1]
    Z1 = np.hstack([ZF0, X]) @ G[1]

    ZF = np.hstack([ZF0, ZF1])
    Z = np.array(Z1)
    # print([f.shape for f in F[1:]])
    ZFt = np.array(ZF1)
    for t in range(2, len(F)):
        C = blow(ZF1, ZFt)
        Zt = np.hstack([ZF, C]) @ (G[t])
        ZFt = np.hstack([ZF, C]) @ (F[t])
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

    Z0 = np.ones((npoints, 1)) * F[0]
    dZ0 = np.zeros((npoints*ndims, 1))
    if len(F) == 1: 
        return dZ0
    
    Z1 = np.hstack((Z0, X)) @ F[1]
    dZ1 = np.repeat(F[1][1:, :], npoints, axis=0)
    Z, dZ =  np.hstack((Z0, Z1)), np.hstack((dZ0, dZ1))
    Zt, dZt = deepcopy(Z1), deepcopy(dZ1)

    for t in range(2,len(F)):
        C, dC = dblow(Z1, Zt, dZ1, dZt)
        Zt = np.hstack((Z, C)) @ F[t]
        dZt  = np.hstack((dZ, dC)) @ F[t]
        Z, dZ = np.hstack((Z, Zt)), np.hstack((dZ, dZt))

    return dZ

def _gradient_v(B, X):
    F = B.nonvanishings()
    G = B.vanishings()
    npoints, ndims = X.shape

    ZF0 = np.ones((npoints, 1)) * F[0]
    dZF0 = np.zeros((npoints*ndims, 1))
    if len(F) == 1: 
        return dZF0
    
    ZF1 = np.hstack((ZF0, X)) @ F[1]
    dZF1 = np.repeat(F[1][1:, :], npoints, axis=0)
    ZF, dZF =  np.hstack((ZF0, ZF1)), np.hstack((dZF0, dZF1))
    ZFt, dZFt = deepcopy(ZF1), deepcopy(dZF1)

    dZ = np.repeat(G[1][1:,:], npoints, axis=0)

    for t in range(2,len(F)):
        C, dC = dblow(ZF1, ZFt, dZF1, dZFt)
        ZFt = np.hstack((ZF, C)) @ F[t]
        dZFt  = np.hstack((dZF, dC)) @ F[t]
        dZt  = np.hstack((dZF, dC)) @ G[t]
        ZF, dZF = np.hstack((ZF, ZFt)), np.hstack((dZF, dZFt))
        dZ = np.hstack((dZ, dZt))
    return dZ

