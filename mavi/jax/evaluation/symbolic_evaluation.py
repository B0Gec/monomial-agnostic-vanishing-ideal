import jax.numpy as np 
import sympy as sm 
from jax import jit 
from mavi.jax.util.util import blow, dblow
# from functools import lru_cache

def evaluate(basis, X, target='vanishing'):
    if target == 'nonvanishing':
        return _evaluate(basis.nonvanishings(concat=True), X)
    elif target == 'vanishing':
        return _evaluate(basis.vanishings(concat=True), X)
    else:
        print('unknown mode: %s' % target)
        exit()

@jit
def _evaluate(F, X):
    return np.vstack([__evaluate(f, X) for f in F]).T

@jit
def __evaluate(f, X):
    assert(type(f).__name__ == 'Poly')
    if f.total_degree() == 0:
        return np.ones(X.shape[0]) * float(f.as_expr())

    f = sm.lambdify(f.gens, f.as_expr(), 'numpy')
    return np.array(f(*X.T))

def gradient(f, X):
    return _evaluate(_gradient_op(f), X)
    # dfX = __evaluate(_gradient_op(f)[0], X)
    # return np.vstack([__evaluate(df, X) for df in _gradient_op(f)])

@jit
def _gradient_op(f):
    return [sm.diff(f, x) for x in f.gens]
