import numpy as np
from mavi.numpy.util.util import res, pres, matrixfact_gep, blow, dblow
from mavi.numpy.base_class.numerical_basis import NBasist as Basist
from mavi.numpy.base_class.numerical_basis import Intermidiate

class Basist_grad(Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate_grad(Intermidiate):
    def __init__(self, FX, dFX):
        super().__init__(FX)
        self.dFX = dFX

    def extend(self, interm):
        super().extend(interm)
        self.dFX = np.hstack((self.dFX, interm.dFX))

def _initialize_grad(X):
    npoints, ndims = X.shape
    constant = np.mean(np.abs(X))

    F = [np.ones((1,1))*constant]
    G = [np.zeros((0,0))]

    FX = np.ones((npoints, 1)) * constant
    dFX = np.zeros((npoints*ndims, 1))

    interm = Intermidiate_grad(FX, dFX)

    basis0 = Basist_grad(G[0], F[0])
    return [basis0], interm


def _init_candidates_grad(X):
    npoints, ndims = X.shape
    dX = np.tile(np.identity(ndims), (npoints, 1))
    return Intermidiate_grad(X, dX)


def _candidates_grad(int_1, int_t):
    return Intermidiate_grad(*dblow(int_1.FX, int_t.FX, int_1.dFX, int_t.dFX))


def _construct_basis_t_grad(cands, intermidiate, eps, gamma=1e-9):
    CtX, dCtX = cands.FX, cands.dFX
    FX, dFX = intermidiate.FX, intermidiate.dFX

    CtX_, R = pres(CtX, FX)
    dCtX_ = res(dCtX, dFX, R)

    d, V = matrixfact_gep(CtX_, dCtX_, gamma=gamma)
    # print(d)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    dFtX = dCtX_ @ V[:, d>eps]

    return Basist_grad(Gt, Ft), Intermidiate_grad(FtX, dFtX)