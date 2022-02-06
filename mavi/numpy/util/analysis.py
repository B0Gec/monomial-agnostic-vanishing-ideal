from operator import length_hint
import numpy as np
from mavi.vanishing_ideal import VanishingIdeal
# from numba import jit

def argfirstnonzero(arr):
    return (np.asarray(arr) != 0).tolist().index(True)

def find_eps_range(X, method, conds, lb=1e-3, ub=1.0, step=1e-2, **kwargs): 

    assert(conds['npolys'][0] == 0)

    ## First determine the (lb, ub), where linear condition holds.
    if conds['npolys'][1] > 0:
        lconds = {'npolys': conds['npolys'][:2], 'cmp': conds['cmp'][:2]}
        # print(lconds)
        lb, ub = _find_eps_range(X, method, lconds, lb, ub, step*10, **kwargs)
        lb, ub = lb - 10*step, ub + 10*step
        print(f'new range: {lb}, {ub}', flush=True)

    if np.any(np.isnan([lb, ub])):
        return (lb, ub)
    else:
        return _find_eps_range(X, method, conds, lb, ub, step, **kwargs)

# @jit
def _find_eps_range(X, method, conds, lb, ub, step, **kwargs):
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
        
    # ret = ((len(ngts) == min_deg + 1) 
    #        and np.all(ngts[:-1] == 0))
    # ret = ret and ((ngts[-1] >= npolys) if relaxed else (ngts[-1] == npolys))
#     print(f'{eps:.4f}, ({lb_:.4f},{ub_:.4f})', ngts)
#     print(ngts[:-1] == 0)
#     print(min_deg, npolys)
    return True

def cmp_func(x, y, cmp='='):
    return eval(f"{x} {cmp} {y}")

if __name__ == '__main__':
    # circle
    N = 24
    theta = [np.pi*i/N*2 for i in range(N)]
    X = np.vstack((np.cos(theta), np.sin(theta))).T

    conds = {'npolys': [0, 1, 0, 1], 'cmp': ['==', '==', '==', '==']}
    find_eps_range(X, 'abm', conds, lb=1e-3, ub=2.0, step=1e-2, term_order='grevlex')    