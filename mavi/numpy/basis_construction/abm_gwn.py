from mavi.numpy.base_class.symbolic_basis import SBasist as _Basist
from mavi.numpy.base_class.symbolic_basis import Intermidiate as _Intermidiate
from mavi.numpy.util.util import matrixfact_gep, blow, argsort
from mavi.numpy.util.symbolic_util import grad_weighted_norm, border_terms
import numpy as np
import sympy as sm 
from sympy.polys.orderings import monomial_key

class Basist(_Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate(_Intermidiate):
    def __init__(self, FX, Fsymb, grad_weights, gens, term_order, X=None):
        super().__init__(FX, Fsymb, gens)
        self.term_order = term_order
        self.grad_weights = grad_weights
        self.X = X
        
    def extend(self, interm):
        super().extend(interm)
        self.grad_weights = np.append(self.grad_weights, interm.grad_weights)

def initialize(X, term_order='grevlex'):  # mush have term_order as keyword arg
    npoints, nvars = X.shape
    constant = 1.

    F = [np.ones((1,1))*constant]
    G = [np.zeros((0,0))]

    FX = np.ones((npoints, 1)) * constant

    gens = sm.symbols(f'x:{nvars}')
    Fsymb = [sm.poly(constant, gens=gens, domain='RR')]
    gwn = np.zeros(1)
    interm = Intermidiate(FX, Fsymb, gwn, gens, term_order)

    basis0 = Basist([], [sm.poly(constant, gens=gens, domain='RR')])
    return [basis0], interm


def init_candidates(X, term_order='grevlex'):  # mush have gen & term_order as keyword arg
    nsamples, nvars = X.shape

    gens = sm.symbols(f'x:{nvars}')
    cands_symb = [sm.poly(gen, gens=gens, domain='RR') for gen in gens]
    
    perm = argsort(cands_symb, key=monomial_key(term_order, gens), reverse=True)  # accesnding
    cands = X[:, perm]
    cands_symb = np.asarray(cands_symb)[perm].tolist()

    gwn = np.ones(len(perm)) # / nsamples**0.5
    
    return Intermidiate(cands, cands_symb, gwn, gens, term_order, X=X)


def candidates(int_1, int_t):
    cands, cands_symb = border_terms(int_1.FX, int_t.FX, int_1.Fsymb, int_t.Fsymb, int_1.gens, int_1.term_order)
    gwns = np.asarray([grad_weighted_norm(cand, int_1.X) for cand in cands_symb])
    return Intermidiate(cands, cands_symb, gwns, int_1.gens, int_1.term_order)


def construct_basis_t(cands, intermidiate, eps, **kwargs):
    CtX = cands.FX        # evaluation matrix of candidate polynomials
    CtXgwn = cands.grad_weights
    FX = intermidiate.FX  # evlauation matrix of nonvanishing polynomials up to degree t-1
    FXgwn = intermidiate.grad_weights
    Fsymb = intermidiate.Fsymb
    Ftsymb = []  # order ideal of degree t
    Gtsymb = []
    FtX = np.zeros((FX.shape[0], 0))
    FtXgwn = np.zeros(0)

    for i, bterm in enumerate(cands.Fsymb): 
        bX = CtX[:, i].reshape(-1, 1)
        bXgwn = CtXgwn[i]

        M = np.hstack([FX, FtX, bX])
        D = np.diag(np.hstack([FXgwn, FtXgwn, [bXgwn]]))
        d, V = matrixfact_gep(M, D)
        if np.min(d) > eps: 
            Ftsymb.append(bterm)
            FtX = np.hstack([FtX, bX])
            FtXgwn = np.append(FtXgwn, [bXgwn])
        else: 
            g = sum((Fsymb + Ftsymb + [bterm]) * V[:, np.argmin(d)])
            Gtsymb.append(g)

    return (Basist(Gtsymb, Ftsymb), 
            Intermidiate(FtX, Ftsymb, FtXgwn, intermidiate.gens, intermidiate.term_order, X=cands.X))