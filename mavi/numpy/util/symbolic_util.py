import numpy as np
import itertools as itr 
import sympy as sm 
from sympy.polys.orderings import monomial_key

from mavi.numpy.util.util import blow, argsort
from mavi.numpy.evaluation.symbolic_evaluation import gradient

def border_terms(AX, BX, Asymb, Bsymb, gens, term_order):
    CX = blow(AX, BX)
    Csymb = blow(Asymb, Bsymb)  # numpy array

    ## keep highest-degree terms
    mdeg = 1 + max(Asymb[0].total_degree(), Bsymb[0].total_degree())
    isborder = [cs.total_degree() == mdeg for cs in Csymb]
    CX, Csymb = CX[:, isborder], Csymb[isborder]

    CX, Csymb = delete_symb_duplicants(CX, Csymb)

    perm = argsort(Csymb, key=monomial_key(term_order, gens))[::-1]  # accesnding
    CX, Csymb = CX[:, perm], np.asarray(Csymb)[perm].tolist()

    return CX, Csymb

# def border_terms_(AX, BX, Asymb, Bsymb, gens, term_order):
#     CX = blow(AX, BX)
#     Csymb = blow(Asymb, Bsymb)  # numpy array

#     ## keep highest-degree terms
#     mdeg = 1 + max(Asymb[0].total_degree(), Bsymb[0].total_degree())
#     isborder = [cs.total_degree() == mdeg for cs in Csymb]
#     CX, Csymb = CX[:, isborder], Csymb[isborder]

#     CX, Csymb = delete_symb_duplicants(CX, Csymb)

#     perm = argsort(Csymb, key=monomial_key(term_order, gens))[::-1]  # accesnding
#     CX, Csymb = CX[:, perm], np.asarray(Csymb)[perm].tolist()

#     gwns = [gradient(h, X) for h in Csymb]

#     return CX, Csymb, gwns

def delete_symb_duplicants(FX, Fsymb):
    FX_ = np.zeros((FX.shape[0], 0))
    Fsymb_ = []
    for (i, f) in enumerate(Fsymb):
        if not (f in Fsymb_):
            FX_ = np.hstack([FX_, FX[:, i].reshape(-1,1)])
            Fsymb_.append(f)
    return FX_, Fsymb_

# def delete_symb_duplicants_(FX, Fsymb, gwns):
#     FX_ = np.zeros((FX.shape[0], 0))
#     gwns_ = []
#     Fsymb_ = []
#     for (i, f) in enumerate(Fsymb):
#         if not (f in Fsymb_):
#             FX_ = np.hstack([FX_, FX[:, i].reshape(-1,1)])
#             Fsymb_.append(f)
#             gwns_.append(gwns[i])
#     return FX_, Fsymb_, gwns_

def grad_weighted_norm(f, X): 
    gwn = np.linalg.norm(gradient(f, X))
    Z = np.linalg.norm(f.degree_list()) * X.shape[0]**0.5

    return gwn / Z

def coefficent_norm(g, numpy=True):
    assert(type(g) is sm.Poly)
    return np.linalg.norm(np.asarray(g.coeffs(), dtype=float))

def coeff_set_distance(G1, G2, unsigned=False, normalize=False):
    ''' 
    - calcuate the minumum average distance of coefficient vectors aross G1 and G2
    '''
    # print(len(G1), len(G2))
    if len(G1) != len(G2):
        print(f'G1: {[g.total_degree() for g in G1]}')
        print(f'G2: {[g.total_degree() for g in G2]}')
        assert(len(G1) == len(G2))
    ds = []
    for G2_ in itr.permutations(G2):
        d = 0.0
        for g1, g2 in zip(G1, G2_):
            d += coeff_distance(g1, g2, unsigned=unsigned, normalize=normalize)
        ds.append(d / len(G1))

    return np.min(ds)

def coeff_distance(g1, g2, unsigned=False, normalize=False): 

    if unsigned: 
        return min(coeff_distance(g1, g2, unsigned=False, normalize=normalize), 
                   coeff_distance(g1, -g2, unsigned=False, normalize=normalize))

    if normalize: 
        z1 = np.linalg.norm(np.asarray(g1.coeffs(), dtype=float))
        z2 = np.linalg.norm(np.asarray(g2.coeffs(), dtype=float))
        g1 = sm.Poly(g1/z1, gens=g1.gens)
        g2 = sm.Poly(g2/z2, gens=g2.gens)

    '''
    calculate the dist of coefficient vectors of two polynomials
    '''
    return coefficent_norm(g1 - g2, numpy=True)

def chop_minor_terms(g, tol=1e-6):
    d = g.as_dict()
    gens = g.gens
    for k in d.keys(): 
        if d[k] < tol and d[k] > - tol:
            d[k] = 0
    h = sm.Poly.from_dict(d, gens=gens)
    return h 

def support(g):
    d = g.as_dict()
    gens = g.gens
    supp = []
    for k in d.keys(): 
        dd = {k: 1}
        h = sm.Poly.from_dict(dd, gens=gens)
        supp.append(h)
    return supp 
