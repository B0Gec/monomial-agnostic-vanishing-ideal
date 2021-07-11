import torch 

class NBasis():
    def __init__(self, basist_arr):
        self.B = basist_arr
        self._i = 0

    def nonvanishings(self):
        return [Bt.F for Bt in self.B]
    
    def vanishings(self):
        return [Bt.G for Bt in self.B]
    
    def __getitem__(self, i):
        return self.B[i]

    def __len__(self):
        return len(self.B)

    def __iter__(self):
        return self 
    
    def __next__(self):
        n = len(self.B)
        if self._i == n:
            raise StopIteration()

        val = self.B[self._i]
        self._i += 1
        return val
            

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
    print(F.shape, L.shape, C.shape)
    resop = torch.vstack([-L, torch.eye(C.shape[1])])
    print(resop.shape)
    res = C - F @ L     # by definition, res == torch.hstack([F, C]) @ resop
    return res, resop

# def pres(C, F):
#     L = np.linalg.lstsq(F, C, rcond=None)[0]
#     print(L.shape, C.shape)
#     resop = np.vstack([-L, np.identity(C.shape[1])])
#     print(resop.shape)
#     res = C - F @ L     # by definition, res == np.hstack([F, C]) @ resop
#     # [F C] * [I -L]
#     return res, resop

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
