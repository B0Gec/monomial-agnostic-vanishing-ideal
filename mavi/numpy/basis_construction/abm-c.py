# from math import perm
# from sympy.logic.boolalg import term_to_integer
# from sympy.utilities.iterables import filter_symbols
from mavi.numpy.base_class.symbolic_basis import SBasist as _Basist
from mavi.numpy.base_class.symbolic_basis import Intermidiate as _Intermidiate
from mavi.numpy.util.util import matrixfact, blow, argsort
from mavi.numpy.util.symbolic_util import border_terms

import numpy as np
import sympy as sm 
from sympy.polys.orderings import monomial_key

class Basist(_Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, Fc):
        super().__init__(FX, Fc)

def initialize(X, **kwargs):  # mush have term_order as keyword arg

    npoints, nvars = X.shape
    constant = 1.
    if 'scaled_const' in kwargs:
        constant = np.mean(np.abs(X)) if kwargs['scaled_const'] else 1
    # constant = np.mean(np.abs(X))
    # print(constant)

    F = [np.ones((1,1))*constant]
    G = [np.zeros((0,0))]

    FX = np.ones((npoints, 1)) * constant
    Fc = np.zeros(X.shape[1])  # exponents of term 1 is (0, 0, ..., 0)
    interm = Intermidiate(FX, Fc)

    basis0 = Basist([], [])
    return [basis0], interm


def init_candidates(X, term_order='grevlex', **kwargs):
    nvars = X.shape[1]
    c = np.hstack([ np.zeros((nvars, 1)), np.eye(nvars)])
    return Intermidiate(X, c)


def candidates(int_1, int_t):
    cands, cands_symb = border_terms(int_1.FX, int_t.FX, int_1.Fsymb, int_t.Fsymb, int_1.gens, int_1.term_order)
    return Intermidiate(cands, cands_symb, int_1.gens, int_1.term_order)


def construct_basis_t(cands, intermidiate, eps, **kwargs):
    CtX = cands.FX        # evaluation matrix of candidate polynomials
    FX = intermidiate.FX  # evlauation matrix of nonvanishing polynomials up to degree t-1
    FtX = np.zeros((FX.shape[0], 0))
    
    Ot = []
    # print(f'--- degree {degree} ---------------')
    for i, bterm in enumerate(cands.Fsymb): 
        bX = CtX[:, i].reshape(-1, 1)
        M = np.hstack([FX, FtX, bX])
        M = M
        d, V = matrixfact(M)

        if np.min(d) > eps: 
            Ot.append(bterm)
            FtX = np.hstack([FtX, bX])
        else: # ここまで書いた．
            g = sum((Fsymb + Ftsymb + [bterm]) * V[:, np.argmin(d)])
            Gtsymb.append(g)

    return (Basist(Gtsymb, Ftsymb), 
            Intermidiate(FtX, Ftsymb, intermidiate.gens, intermidiate.term_order))