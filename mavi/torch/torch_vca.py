import numpy as np
from util import res, pres, matrixfact, blow

class Basist_vca():
    def __init__(self, G, F):
        self.G = G
        self.F = F

    def isemptyF(self):
        return np.size(self.F) == 0

class Intermidiate_vca():
    def __init__(self, FX):
        self.FX = FX

    def extend(self, interm_g):
        self.FX = np.hstack((self.FX, interm_g.FX))

def _initialize_vca(X):
    npoints, ndims = X.shape
    constant = 1./npoints**0.5

    F = [np.ones((1,1))*constant]
    G = [np.zeros((0,0))]

    FX = np.ones((npoints, 1)) * constant

    interm = Intermidiate_vca(FX)

    basis0 = Basist_vca(G[0], F[0])
    return [basis0], interm


def _init_candidates_vca(X):
    return Intermidiate_vca(X)


def _candidates_vca(int_1, int_t):
    return Intermidiate_vca(blow(int_1.FX, int_t.FX))


def _construct_basis_t_vca(cands, intermidiate, eps):
    CtX = cands.FX
    FX = intermidiate.FX

    CtX_, R = pres(CtX, FX)

    d, V = matrixfact(CtX_)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    scales = np.linalg.norm(FtX, axis=0)
    FtX /= scales
    Ft /= scales

    return Basist_vca(Gt, Ft), Intermidiate_vca(FtX)