import torch 

def blow(A, B):
    n1, n2 = A.shape[1], B.shape[1]
    # C = A.repeat_interleave((1, n2)) * B.repeat((1, n1))
    C = A.repeat_interleave(n2, dim=1) * B.repeat((1, n1))
    return C


def dblow(A, B, dA, dB):
    n1, n2 = A.shape[1], B.shape[1]
    ndims = dA.shape[0]//A.shape[0]

    C = A.repeat_interleave(n2, dim=1) * B.repeat((1, n1))
    dC1 = A.repeat_interleave(ndims, dim=0).repeat_interleave(n2, dim=1) * dB.repeat((1, n1))
    dC2 = dA.repeat_interleave(n2, dim=1) * B.repeat_interleave(ndims, dim=0).repeat((1, n1))
    dC = dC1 + dC2
    return C, dC


## extract residual components and projection operator
# @profile()
def pres(C, F):
    L = torch.linalg.lstsq(F, C).solution
    res = C - F @ L     # by definition, res == torch.hstack([F, C]) @ L
    return res, L

## project C to residual space
def res(C, F, L):
    return C - F @ L

def matrixfact(C):
    _, d, Vt = torch.linalg.svd(C, full_matrices=True)
    d = torch.cat((d, torch.zeros(Vt.shape[0] - len(d))))
    return d, Vt.T

def indirect_ged(A, B, gamma=1e-9):
    '''
    Reduce GED to two SVD
    '''
    Vb, db, _ = torch.linalg.svd(B)
    db += gamma 
    Vb_ = Vb / (db**0.5)

    A_ = Vb_.T @ A @ Vb_
    Va, d, _ = torch.linalg.svd(A_)
    V = Vb_ @ Va

    return d, V

def matrixfact_gep(C, N, gamma=1e-9):
    '''
    Unfortunately, torch.lobpcg is the only way of solving generalied eigenvalue problem but it cannot perform full decomposition.
    Instead of directly solving generalized eigenvalue problem, we reduce the problem to an eigenvalue problem
    '''
    A = C.T @ C
    B = N.T @ N
    r = torch.linalg.matrix_rank(B, gamma)
    gamma_ = B.diag().mean()*gamma
    d, V = indirect_ged(A, B, gamma=gamma_)
    d = d.abs()**0.5

    gnorms = torch.diag(V.T@B@V)
    valid = torch.argsort(-gnorms)[:r]

    d, V = d[valid], V[:, valid]
    gnorms = gnorms[valid]
   
    perm = torch.argsort(-d)

    return d[perm], V[:, perm]


def jacobian(F, inputs, requires_grad=False):
    '''
    F: n_in -> n_out
    # dot product between jacobians
    J = jacibian(network, inputs)  # n_out x n_points x n_in

    torch.tensordot(J, J, dims=([1, 2], [1, 2])) # n_out x n_out
    '''
    points_jac = torch.autograd.functional.jacobian(
        F, 
        inputs,
        create_graph=requires_grad  # necessary if grad in loss!!
    ).sum(dim=0)  

    return points_jac  # n_out x n_points x n_in

def grad_normalization_mat(F, inputs, requires_grad=False):
    J = jacobian(F, inputs, requires_grad=requires_grad)
    N = torch.tensordot(J, J, dims=([1, 2], [1, 2])) # n_out x n_out
    return N
