import torch 

def blow(A, B):
    n1, n2 = A.shape[1], B.shape[1]
    C = A.repeat_interleave((1, n2)) * B.repeat((1, n1))
    return C


def dblow(A, B, dA, dB):
    n1, n2 = A.shape[1], B.shape[1]
    ndims = torch.int(dA.shape[0]/A.shape[0])

    C = A.repeat_interleave((1, n2)) * B.repeat((1, n1))
    dC1 = A.repeat_interleave((ndims, 1)).repeat_interleave((1, n2)) * dB.repeat((1, n1))
    dC2 = dA.repeat((1, n2)) * B.repeat_interleave((ndims, 1)).repeat((1, n1))
    dC = dC1 + dC2

    return C, dC


## extract residual components and projection operator
def pres(C, F):
    # L = torch.linalg.lstsq(F, C, rcond=None)[0]
    L = torch.lstsq(C, F)[0][:F.shape[1], :]
    resop = torch.vstack([-L, torch.eye(C.shape[1])])
    res = C - F @ L     # by definition, res == torch.hstack([F, C]) @ resop
    return res, resop

## project C to residual space
def res(C, F, R):
    return torch.hstack([F, C]) @ R

def matrixfact(C):
    _, d, V = torch.linalg.svd(C, full_matrices=True)
    return d, V.T

def matrixfact_gep(C, N, gamma=1e-9):
    # A = Symmetric(C.T@C)
    # B = Symmetric(N.T@N)

    A = C.T @ C
    B = N.T @ N

    r = torch.linalg.matrix_rank(B, gamma)
    gamma_ = torch.mean(torch.diag(B))*gamma

    d, V = torch.linalg.eigh((A, B+gamma_*torch.eye(B.shape[0])))
    d = torch.sqrt(torch.abs(d))

    gnorms = torch.diag(V.T@B@V)
    valid = torch.argsort(-gnorms)[:r]

    d, V = d[valid], V[:, valid]
    gnorms = gnorms[valid]

    perm = torch.argsort(-d)
    return d[perm], V[:, perm]

