import jax.numpy as np
from sympy.polys.orderings import monomial_key

from mavi.jax.util.util import blow, argsort
from mavi.jax.evaluation.symbolic_evaluation import gradient

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