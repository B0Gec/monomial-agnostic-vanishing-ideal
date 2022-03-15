import numpy as np
from mavi.numpy.base_class.numerical_basis import Nbasist_fn
from mavi.numpy.base_class.numerical_basis import NBasist as _Basist
from mavi.numpy.base_class.numerical_basis import Intermidiate as _Intermidiate
from mavi.numpy.util.util import res, pres, matrixfact_gep, blow, blow1
from mavi.numpy.util.symbolic_util import coeff_corr_mat, sblow, sblow1
import sympy as sm 

class Basist(_Basist):
    def __init__(self, G, F, Gsymb=None, Fsymb=None):
        super().__init__(G, F, Gsymb=Gsymb, Fsymb=Fsymb)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, Fsymb=None):
        super().__init__(FX, Fsymb=Fsymb)

    def extend(self, interm):
        super().extend(interm)

def initialize(X, **kwargs):
    npoints, ndims = X.shape
    constant = 1.0 

    F0 = Nbasist_fn(np.ones((1,1))*constant)
    G0 = Nbasist_fn(np.zeros((0,0)))

    FX = np.ones((npoints, 1)) * constant

    gens = sm.symbols(f'x:{ndims}')
    F0symb = np.array([sm.Poly(constant, gens=gens)])
    G0symb = np.array([])

    interm = Intermidiate(FX, Fsymb=F0symb)

    basis0 = Basist(G0, F0, Gsymb=G0symb, Fsymb=F0symb)
    return [basis0], interm


def init_candidates(X, **kwargs):
    npoints, ndims = X.shape
    gens = sm.symbols(f'x:{ndims}')
    C1symb = np.asarray([sm.Poly(s,gens=gens) for s in gens])
    return Intermidiate(X, Fsymb=C1symb)


def candidates(int_1, int_t, degree=None):
    if degree == 2: 
        return Intermidiate(blow1(int_1.FX), Fsymb=sblow1(int_1.Fsymb))
    else:
        return Intermidiate(blow(int_1.FX, int_t.FX), Fsymb=sblow(int_1.Fsymb, int_t.Fsymb))


def construct_basis_t(cands, intermidiate, eps, gamma=1e-9):
    CtX, Ctsymb = cands.FX, cands.Fsymb
    FX, Fsymb = intermidiate.FX, intermidiate.Fsymb

    CtX_, L = pres(CtX, FX)
    Ctsymb_ = res(Ctsymb.reshape(1, -1), Fsymb.reshape(1, -1), L).flatten()

    M = coeff_corr_mat(Ctsymb_)
    nsamples = CtX_.shape[0]
    d, V = matrixfact_gep(CtX_, M, gamma=gamma, preparedB=True)
    # print(V/np.linalg.norm(V, axis=0))
    # print(CtX)
    # print(M)
    # print(d)

    FtX = CtX_ @ V[:, d>eps]
    Ft = Nbasist_fn(V[:, d>eps], L)
    Gt = Nbasist_fn(V[:, d<=eps], L)

    Ftsymb = (Ctsymb_.reshape(1, -1) @ V[:, d>eps]).flatten()
    Gtsymb = (Ctsymb_.reshape(1, -1) @ V[:, d<=eps]).flatten() 

    return Basist(Gt, Ft, Gsymb=Gtsymb, Fsymb=Ftsymb), Intermidiate(FtX, Fsymb=Ftsymb)