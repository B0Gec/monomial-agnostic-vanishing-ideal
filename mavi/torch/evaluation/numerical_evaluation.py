import torch 
from mavi.torch.util.util import blow, dblow

def evaluate(basis, X_, target='vanishing'):
    X = X_.float()
    device = X.device
    if target == 'nonvanishing':
        return _evaluate_nv(basis, X, device=device)
    elif target == 'vanishing':
        return _evaluate_v(basis, X, device=device)
    else:
        print('unknown mode: %s' % target)
        exit()
    
def _evaluate_nv(B, X, device='cpu'):
    F = B.nonvanishings()
    
    N = X.shape[0]

    Z0 = F[0].eval(torch.ones(N, 1, device=device))
    if len(F) == 1: return Z0
    
    # Z1 = torch.hstack([Z0, X]) @ F[1]
    Z1 = F[1].eval(Z0, X)
    Z = torch.hstack([Z0, Z1])

    Zt = Z1.clone()

    for t in range(2, len(F)):
        C = blow(Z1, Zt)
        # Zt = torch.hstack([Z, C]) @ F[t]
        Zt = F[t].eval(Z, C)
        Z = torch.hstack([Z, Zt])

    return Z


def _evaluate_v(B, X, device='cpu'):
    F = B.nonvanishings()
    G = B.vanishings()
    # if torch.all(torch.tensor([gt.numel()==0 for gt in G])):
    #     return torch.zeros(len(X), 0, device=device)

    N = X.shape[0]

    ZF0 = F[0].eval(torch.ones(N, 1, device=device))
    ZF1 = F[1].eval(ZF0, X)
    Z1 = G[1].eval(ZF0, X)

    ZF = torch.hstack([ZF0, ZF1])
    Z = Z1.clone()
    # print([f.shape for f in F[1:]])
    ZFt = ZF1.clone()
    for t in range(2, len(F)):
        C = blow(ZF1, ZFt)
        Zt = G[t].eval(ZF, C)
        ZFt = F[t].eval(ZF, C)
        ZF = torch.hstack([ZF, ZFt])
        Z = torch.hstack([Z, Zt])

    return Z


def gradient(basis, X_, target='vanishing'):
    X = X_.float()
    device = X.device
    if target == 'nonvanishing':
        return _gradient_nv(basis, X, device=device)
    elif target == 'vanishing':
        return _gradient_v(basis, X, device=device)        
    else:
        print('unknown mode %s' % target)

def _gradient_nv(B, X, device='cpu'):
    F = B.nonvanishings()
    npoints, ndims = X.shape

    Z0 = torch.ones(npoints, 1, device=device) * F[0]
    dZ0 = torch.zeros(npoints*ndims, 1, device=device)
    if len(F) == 1: 
        return dZ0
    
    Z1 = torch.hstack((Z0, X)) @ F[1]
    # dZ1 = torch.repeat(F[1][1:, :], npoints, axis=0)
    dZ1 = F[1][1:, :].repeat_interleave(npoints, dim=0)
    Z, dZ =  torch.hstack((Z0, Z1)), torch.hstack((dZ0, dZ1))
    Zt, dZt = Z1.clone(), dZ1.clone()

    for t in range(2,len(F)):
        C, dC = dblow(Z1, Zt, dZ1, dZt)
        Zt = torch.hstack((Z, C)) @ F[t]
        dZt  = torch.hstack((dZ, dC)) @ F[t]
        Z, dZ = torch.hstack((Z, Zt)), torch.hstack((dZ, dZt))

    return dZ

def _gradient_v(B, X, device='cpu'):
    F = B.nonvanishings()
    G = B.vanishings()
    npoints, ndims = X.shape

    ZF0 = torch.ones(npoints, 1, device=device) * F[0]
    dZF0 = torch.zeros(npoints*ndims, 1, device=device)
    if len(F) == 1: 
        return dZF0
    
    ZF1 = torch.hstack((ZF0, X)) @ F[1]
    dZF1 = F[1][1:, :].repeat_interleave(npoints, dim=0)
    ZF, dZF =  torch.hstack((ZF0, ZF1)), torch.hstack((dZF0, dZF1))
    ZFt, dZFt = ZF1.clone(), dZF1.clone()

    dZ = G[1][1:,:].repeat_interleave(npoints, dim=0)

    for t in range(2,len(F)):
        C, dC = dblow(ZF1, ZFt, dZF1, dZFt)
        ZFt = torch.hstack((ZF, C)) @ F[t]
        dZFt  = torch.hstack((dZF, dC)) @ F[t]
        dZt  = torch.hstack((dZF, dC)) @ G[t]
        ZF, dZF = torch.hstack((ZF, ZFt)), torch.hstack((dZF, dZFt))
        dZ = torch.hstack((dZ, dZt))
    return dZ


