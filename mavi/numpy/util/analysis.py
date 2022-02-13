import numpy as np
from pandas import concat
from mavi.vanishing_ideal import VanishingIdeal
from mavi.numpy.util.symbolic_util import support
# from numba import jit

def argfirstnonzero(arr):
    return (np.asarray(arr) != 0).tolist().index(True)


def find_eps_range(X, method, conds, criterion='config', lb=1e-3, ub=1.0, step=1e-2, quiet=False, **kwargs): 
    if criterion == 'config': 
        return find_eps_range_c(X, method, conds, lb=lb, ub=ub, step=step, quiet=quiet, **kwargs)
    if criterion == 'order': 
        return find_eps_range_o(X, method, conds, lb=lb, ub=ub, step=step, quiet=quiet, **kwargs)
    if criterion == 'structure': 
        return find_eps_range_s(X, method, conds, lb=lb, ub=ub, step=step, quiet=quiet, **kwargs)


def find_eps_range_c(X, method, conds, lb=1e-3, ub=1.0, step=1e-2, quiet=False, **kwargs): 

    assert(conds['npolys'][0] == 0)
    
    ## First determine the (lb, ub), where linear condition holds.
    if conds['npolys'][1] > 0:
        lconds = {'npolys': conds['npolys'][:2], 'cmp': conds['cmp'][:2]}
        lb, ub = _find_eps_range_c(X, method, lconds, lb, ub, step*10, **kwargs)
        # print(lconds)
        lb, ub = lb - 10*step, ub + 10*step
        if not quiet:
            print(f'new range: {lb}, {ub}', flush=True)

    if np.any(np.isnan([lb, ub])):
        return (lb, ub)
    else:
        return _find_eps_range_c(X, method, conds, lb, ub, step, **kwargs)


def find_eps_range_o(X, method, O, lb=1e-3, ub=1.0, step=1e-2, quiet=False, **kwargs): 

    ## First determine the (lb, ub), where linear condition holds.
    O1 = [o for o in O if o.total_degree() <= 1]
    lb, ub = _find_eps_range_o(X, method, O1, lb, ub, step*10, **kwargs)
    # print(O1)
    lb, ub = lb - 10*step, ub + 10*step
    if not quiet:
        print(f'new range: {lb}, {ub}', flush=True)

    if np.any(np.isnan([lb, ub])):
        return (lb, ub)
    else:
        return _find_eps_range_o(X, method, O, lb, ub, step, **kwargs)

def find_eps_range_s(X, method, S, lb=1e-3, ub=1.0, step=1e-2, quiet=False, **kwargs): 

    ## First determine the (lb, ub), where linear condition holds.
    S1 = [s for s in S if s.total_degree() <= 1]
    lb, ub = _find_eps_range_o(X, method, S1, lb, ub, step*10, **kwargs)
    # print(O1)
    lb, ub = lb - 10*step, ub + 10*step
    if not quiet:
        print(f'new range: {lb}, {ub}', flush=True)

    if np.any(np.isnan([lb, ub])):
        return (lb, ub)
    else:
        return _find_eps_range_o(X, method, S, lb, ub, step, **kwargs)

# @jit
def _find_eps_range_c(X, method, conds, lb, ub, step, **kwargs):
    npolys  = conds['npolys']
    cmps = conds['cmp']

    assert(len(npolys) == len(cmps))
    lb_, ub_ = (np.nan, np.nan)
    for eps in np.arange(lb, ub, step):
        # print(f'eps = {eps}')
        vi = VanishingIdeal()
        vi.fit(X, eps, method=method, max_degree=len(npolys)-1, **kwargs)

        if len(vi.basis) < len(cmps): continue

        if (np.isnan(lb_) 
            and condition_checker(vi, npolys, cmps)):
            lb_ = eps
            # print(f'{lb_} ---> ', end='')

        if (np.isnan(ub_) 
            and not np.isnan(lb_)
            and not condition_checker(vi, npolys, cmps)):

            ub_ = max(lb_, eps - step)
            break
            
    if np.isnan(ub_) and not np.isnan(lb_):
        ub_ = eps
        # print(ub)

    return (lb_, ub_)

def condition_checker(vi, npolys, cmps, relaxed=False): 
    ngts = np.asarray([np.asarray(gt).shape[-1] for gt in vi.basis.vanishings()])
    # print(ngts)
    for npoly, ngt, cmp in zip(npolys, ngts, cmps):
        if not cmp_func(npoly, ngt, cmp): return False
        
    return True

def cmp_func(x, y, cmp='='):
    return eval(f"{x} {cmp} {y}")


def _find_eps_range_o(X, method, O, lb, ub, step, **kwargs):

    max_deg = np.max([o.total_degree() for o in O])
    O = set(O)
    # print(lb, ub, step)
    lb_, ub_ = (np.nan, np.nan)
    for eps in np.arange(lb, ub, step):
        # print(f'eps = {eps}')
        vi = VanishingIdeal()
        vi.fit(X, eps, method=method, max_degree=max_deg, **kwargs)

        ok = set(vi.basis.nonvanishings(concat=True)) == O
        if not ok: continue

        if (np.isnan(lb_) and ok):
            lb_ = eps
            # print(f'{lb_} ---> ', end='')

        if (np.isnan(ub_) 
            and not np.isnan(lb_)
            and not ok):

            ub_ = max(lb_, eps - step)
            break
            
    if np.isnan(ub_) and not np.isnan(lb_):
        ub_ = eps
        # print(ub)

    return (lb_, ub_)

def _find_eps_range_s(X, method, S, lb, ub, step, **kwargs):

    max_deg = np.max([s.total_degree() for s in S])
    S = set(S)
    # print(lb, ub, step)
    lb_, ub_ = (np.nan, np.nan)
    for eps in np.arange(lb, ub, step):
        # print(f'eps = {eps}')
        vi = VanishingIdeal()
        vi.fit(X, eps, method=method, max_degree=max_deg, **kwargs)

        ok = set().union(*[set(support(g)) for g in vi.basis.vanishings(concat=True)]) == S
        if not ok: continue

        if (np.isnan(lb_) and ok):
            lb_ = eps
            # print(f'{lb_} ---> ', end='')

        if (np.isnan(ub_) 
            and not np.isnan(lb_)
            and not ok):

            ub_ = max(lb_, eps - step)
            break
            
    if np.isnan(ub_) and not np.isnan(lb_):
        ub_ = eps
        # print(ub)

    return (lb_, ub_)


if __name__ == '__main__':
    # circle
    N = 24
    theta = [np.pi*i/N*2 for i in range(N)]
    X = np.vstack((np.cos(theta), np.sin(theta))).T

    conds = {'npolys': [0, 1, 0, 1], 'cmp': ['==', '==', '==', '==']}
    find_eps_range(X, 'abm', conds, lb=1e-3, ub=2.0, step=1e-2, term_order='grevlex')    