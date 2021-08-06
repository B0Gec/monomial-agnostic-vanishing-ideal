import torch
from mavi.torch.util.util import pres, res, matrixfact_gep, blow, dblow, grad_normalization_mat
from mavi.torch.base_class.numerical_basis import NBasist as _Basist
from mavi.torch.base_class.numerical_basis import Intermidiate as _Intermidiate
from mavi.torch.evaluation.numerical_evaluation import evaluate

class Basist(_Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, dFX):
        super().__init__(FX)
        self.dFX = dFX

    def extend(self, interm):
        super().extend(interm)
        self.dFX = torch.hstack((self.dFX, interm.dFX))

def initialize(X, **kwargs):
    device = X.device
    npoints, ndims = X.shape
    constant = X.abs().mean()

    F = [torch.ones(1,1, device=device)*constant]
    G = [torch.zeros(0,0, device=device)]

    FX = torch.ones(npoints, 1, device=device) * constant
    dFX = torch.zeros(npoints*ndims, 1, device=device)

    interm = Intermidiate(FX, dFX)

    basis0 = Basist(G[0], F[0])
    return [basis0], interm


def init_candidates(X, **kwargs):
    device = X.device
    npoints, ndims = X.shape
    # dX = torch.tile(torch.eye(ndims), (npoints, 1))
    dX = torch.eye(ndims, device=device).repeat((npoints, 1))

    return Intermidiate(X, dX)


def candidates(int_1, int_t, **kwargs):
    return Intermidiate(*dblow(int_1.FX, int_t.FX, int_1.dFX, int_t.dFX))


def construct_basis_t(cands, intermidiate, eps, gamma=1e-6):
    CtX, dCtX = cands.FX, cands.dFX
    FX, dFX = intermidiate.FX, intermidiate.dFX

    CtX_, R = pres(CtX, FX)
    dCtX_ = res(dCtX, dFX, R)
    # dCtX_ = grad_normalization_mat(CtX_, cands.X)
    d, V = matrixfact_gep(CtX_, dCtX_, gamma=gamma)
    # print(d)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    dFtX = dCtX_ @ V[:, d>eps]

    return Basist(Gt, Ft), Intermidiate(FtX, dFtX)