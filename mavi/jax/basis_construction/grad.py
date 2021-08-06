import jax.numpy as np
from mavi.jax.base_class.numerical_basis import Nbasist_fn
from mavi.jax.base_class.numerical_basis import NBasist as _Basist
from mavi.jax.base_class.numerical_basis import Intermidiate as _Intermidiate
from mavi.jax.util.util import res, pres, matrixfact_gep, blow, dblow

class Basist(_Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, dFX):
        super().__init__(FX)
        self.dFX = dFX

    def extend(self, interm):
        super().extend(interm)
        self.dFX = np.hstack((self.dFX, interm.dFX))

def initialize(X):
    npoints, ndims = X.shape
    constant = np.mean(np.abs(X))

    F0 = Nbasist_fn(np.ones((1,1))*constant)
    G0 = Nbasist_fn(np.zeros((0,0)))

    FX = np.ones((npoints, 1)) * constant
    dFX = np.zeros((npoints*ndims, 1))

    interm = Intermidiate(FX, dFX)

    basis0 = Basist(G0, F0)
    return [basis0], interm


def init_candidates(X, **kwargs):
    npoints, ndims = X.shape
    dX = np.tile(np.identity(ndims), (npoints, 1))
    return Intermidiate(X, dX)


def candidates(int_1, int_t, **kwargs):
    return Intermidiate(*dblow(int_1.FX, int_t.FX, int_1.dFX, int_t.dFX))


def construct_basis_t(cands, intermidiate, eps, gamma=1e-9):
    CtX, dCtX = cands.FX, cands.dFX
    FX, dFX = intermidiate.FX, intermidiate.dFX

    CtX_, L = pres(CtX, FX)
    dCtX_ = res(dCtX, dFX, L)

    d, V = matrixfact_gep(CtX_, dCtX_, gamma=gamma)
    # print(d)

    FtX = CtX_ @ V[:, d>eps]
    dFtX = dCtX_ @ V[:, d>eps]
    Ft = Nbasist_fn(V[:, d>eps], L)
    Gt = Nbasist_fn(V[:, d<=eps], L)

    return Basist(Gt, Ft), Intermidiate(FtX, dFtX)