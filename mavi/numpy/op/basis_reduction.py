import numpy as np 
from mavi.numpy.evaluation.numerical_evaluation import gradient

def is_generatable(dgX, dGX, tol=1e-6):
    npoints, ndims = dgX.shape
    
    # for each point ...
    for dgx, dGx in zip(dgX, dGX): 
        _, res, _, _ = np.linalg.lstsq(dGx, dgx, rcond=None)
        if res > tol: 
            return False 

    return True 

def basis_reduction(vi, X, tol=1e-6):
    nGt = [Gt.n_bases() for Gt in vi.basis.vanishings()]
    dGX = vi.gradient(X, keep_dim=True)  # npoints x ndims x |G|
    remove = np.repeat([False], sum(nGt))
    for t, n in enumerate(nGt[:-1]): 
        if n == 0: continue 
        
        tau = [i for i, m in enumerate(nGt[t+1:], start=1) if m != 0]
        tau = t + tau[0]
        # remove = np.repeat([False], nGt[tau])
        for m in range(nGt[tau]):
            k = sum(nGt[:tau])
            dGX[:, :, k+m-1]
            dGX[:, :, ~remove]
            dGX[:, :, :k][:, :, ~remove[:k]]
            remove[k+m-1] = is_generatable(dGX[:, :, k+m-1], dGX[:, :, :k][:, :, ~remove[:k]], tol=tol)

    return dGX, remove

        