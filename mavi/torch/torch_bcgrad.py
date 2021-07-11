import torch 
from util.torch_util import res, pres, matrixfact_gep, blow, dblow

class Basist_grad():
    def __init__(self, G, F):
        self.G = G
        self.F = F

    def isemptyF(self):
        return self.F.nelement() == 0

class Intermidiate_grad():
    def __init__(self, FX, dFX):
        self.FX = FX
        self.dFX = dFX
    def extend(self, interm_g):
        self.FX = torch.cat((self.FX, interm_g.FX), dim=1)
        self.dFX = torch.cat((self.dFX, interm_g.dFX), dim=1)

def _initialize_grad(X):
    npoints, ndims = X.shape
    constant = torch.mean(torch.abs(X))

    F = [torch.ones((1,1))*constant]
    G = [torch.zeros((0,0))]

    FX = torch.ones((npoints, 1)) * constant
    dFX = torch.zeros((npoints*ndims, 1))

    interm = Intermidiate_grad(FX, dFX)

    basis0 = Basist_grad(G[0], F[0])
    return [basis0], interm


def _init_candidates_grad(X):
    npoints, ndims = X.shape
    dX = torch.tile(torch.eye(ndims), (npoints, 1))
    return Intermidiate_grad(X, dX)


def _candidates_grad(int_1, int_t):
    return Intermidiate_grad(*dblow(int_1.FX, int_t.FX, int_1.dFX, int_t.dFX))


def _construct_basis_t_grad(cands, intermidiate, eps):
    CtX, dCtX = cands.FX, cands.dFX
    FX, dFX = intermidiate.FX, intermidiate.dFX

    CtX_, R = pres(CtX, FX)
    dCtX_ = res(dCtX, dFX, R)

    d, V = matrixfact_gep(CtX_, dCtX_)
    # print(d)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    dFtX = dCtX_ @ V[:, d>eps]

    return Basist_grad(Gt, Ft), Intermidiate_grad(FtX, dFtX)