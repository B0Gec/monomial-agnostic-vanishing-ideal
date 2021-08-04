import torch
from mavi.torch.base_class.numerical_basis import NBasist as _Basist
from mavi.torch.base_class.numerical_basis import Intermidiate as _Intermidiate
from mavi.torch.util.util import res, pres, matrixfact, blow

class Basist(_Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate(_Intermidiate):
    def __init__(self, FX):
        super().__init__(FX)

    def extend(self, interm):
        super().extend(interm)

def initialize(X, **kwargs):
    npoints, ndims = X.shape
    constant = 1./npoints**0.5

    F = [torch.ones(1,1)*constant]
    G = [torch.zeros(0,0)]

    FX = torch.ones(npoints, 1) * constant

    interm = Intermidiate(FX)

    basis0 = Basist(G[0], F[0])
    return [basis0], interm


def init_candidates(X, **kwargs):
    return Intermidiate(X)


def candidates(int_1, int_t):
    return Intermidiate(blow(int_1.FX, int_t.FX))


def construct_basis_t(cands, intermidiate, eps, **kwargs):
    CtX = cands.FX        # evaluation matrix of candidate polynomials
    FX = intermidiate.FX  # evlauation matrix of nonvanishing polynomials up to degree t-1

    CtX_, R = pres(CtX, FX)  # orthogonal projection

    d, V = matrixfact(CtX_)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    scales = FtX.norm(dim=0)
    FtX /= scales
    Ft /= scales

    return Basist(Gt, Ft), Intermidiate(FtX)