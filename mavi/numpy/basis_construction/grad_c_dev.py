import numpy as np
from mavi.numpy.base_class.numerical_basis import Nbasist_fn
from mavi.numpy.base_class.numerical_basis import NBasist as _Basist
from mavi.numpy.base_class.numerical_basis import Intermidiate as _Intermidiate
from mavi.numpy.util.util import res, pres, matrixfact_gep, dblow, dblow1
from mavi.numpy.util.symbolic_util import sblow1, sblow
import sympy as sm 

class Basist(_Basist):
    def __init__(self, G, F, Gsymb=None, Fsymb=None):
        super().__init__(G, F, Gsymb=Gsymb, Fsymb=Fsymb)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, dFX, Fsymb=None):
        super().__init__(FX, Fsymb=Fsymb)
        self.dFX = dFX

    def extend(self, interm):
        super().extend(interm)
        self.dFX = np.hstack((self.dFX, interm.dFX))

def initialize(X, **kwargs):
    npoints, ndims = X.shape
    constant = np.mean(np.abs(X))

    F0 = Nbasist_fn(np.ones((1,1))*constant)
    G0 = Nbasist_fn(np.zeros((0,0)))

    FX = np.ones((npoints, 1)) * constant
    dFX = np.zeros((npoints*ndims, 1))

    gens = sm.symbols(f'x:{ndims}')
    F0symb = np.array([sm.Poly(constant, gens=gens)])
    G0symb = np.array([])

    interm = Intermidiate(FX, dFX, Fsymb=F0symb)

    basis0 = Basist(G0, F0, Gsymb=G0symb, Fsymb=F0symb)
    return [basis0], interm


def init_candidates(X, **kwargs):
    npoints, ndims = X.shape
    dX = np.tile(np.identity(ndims), (npoints, 1))
    
    gens = sm.symbols(f'x:{ndims}')
    C1symb = np.asarray([sm.Poly(s,gens=gens) for s in gens])
    return Intermidiate(X, dX, Fsymb=C1symb)


def candidates(int_1, int_t, degree=None):
    if degree == 2: 
        return Intermidiate(*dblow1(int_1.FX, int_1.dFX), Fsymb=sblow1(int_1.Fsymb))
    else:
        return Intermidiate(*dblow(int_1.FX, int_t.FX, int_1.dFX, int_t.dFX), Fsymb=sblow(int_1.Fsymb, int_t.Fsymb))


def construct_basis_t(cands, intermidiate, eps, gamma=1e-9, z=1.0):
    CtX, dCtX = cands.FX, cands.dFX
    FX, dFX = intermidiate.FX, intermidiate.dFX

    CtX_, L = pres(CtX, FX)
    dCtX_ = res(dCtX, dFX, L)
    dCtX_, dL = pres(dCtX_, dFX)
    CtX_ = res(CtX_, FX, dL)
    

    nsamples = CtX_.shape[0]
    d, V = matrixfact_gep(CtX_, dCtX_/nsamples**0.5/z, gamma=gamma)
    # print(d)

    FtX = CtX_ @ V[:, d>eps]
    dFtX = dCtX_ @ V[:, d>eps]
    Ft = Nbasist_fn(V[:, d>eps], L)
    Gt = Nbasist_fn(V[:, d<=eps], L)

    Ctsymb = res(cands.Fsymb, intermidiate.Fsymb, L)
    Ftsymb = Ctsymb @ V[:, d>eps] 
    Gtsymb = Ctsymb @ V[:, d<=eps] 

    return Basist(Gt, Ft, Gsymb=Gtsymb, Fsymb=Ftsymb), Intermidiate(FtX, dFtX, Fsymb=Ftsymb)