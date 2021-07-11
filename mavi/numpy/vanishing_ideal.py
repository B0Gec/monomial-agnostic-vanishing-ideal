# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
from copy import deepcopy
# from functools import lru_cache


from mavi.numpy.util.util import blow, dblow
from mavi.numpy.util.cache import memoize
from mavi.numpy.util.plot import plot
from mavi.numpy.base_class.basis import Basis
from mavi.numpy.evaluation.numerical_evaluation import evaluate as n_evaluate
from mavi.numpy.evaluation.numerical_evaluation import gradient as n_gradient
from mavi.numpy.evaluation.symbolic_evaluation import evaluate as s_evaluate
from mavi.numpy.evaluation.symbolic_evaluation import gradient as s_gradient

from mavi.numpy.basis_construction.bcgrad import Basist_grad, Intermidiate_grad
from mavi.numpy.basis_construction.bcgrad import _initialize_grad, _init_candidates_grad, _candidates_grad
from mavi.numpy.basis_construction.bcgrad import _construct_basis_t_grad

from mavi.numpy.basis_construction.vca import Basist_vca, Intermidiate_vca
from mavi.numpy.basis_construction.vca import _initialize_vca, _init_candidates_vca, _candidates_vca
from mavi.numpy.basis_construction.vca import _construct_basis_t_vca

from mavi.numpy.basis_construction.abm import Basist_abm, Intermidiate_abm
from mavi.numpy.basis_construction.abm import _initialize_abm, _init_candidates_abm, _candidates_abm
from mavi.numpy.basis_construction.abm import _construct_basis_t_abm

from mavi.numpy.basis_construction.abm_gwn import Basist_abm_gwn, Intermidiate_abm_gwn
from mavi.numpy.basis_construction.abm_gwn import _initialize_abm_gwn, _init_candidates_abm_gwn, _candidates_abm_gwn
from mavi.numpy.basis_construction.abm_gwn import _construct_basis_t_abm_gwn

# class ExactApproximateVI():
class VanishingIdeal():
    def __init__(self):
        self.basis  = []
        self.eps    = None 
        self.method = None 
        ''
        # 'This is a class to compute exact approximate vanishing ideal'
        # '''
        # fit(X, eps, dlt, misc)
        #     This computes 
        # '''

    # def __len(self)__:
    #     return len(self.basis)

    def fit(self, X, eps, method="grad", max_degree=15, gamma=1e-9, **kwargs):
        ## set attributes
        self.X = np.asarray(X)
        self.eps = eps
        self.method = method
        self.max_degree = max_degree
        self.gamma = gamma
        self.symbolic = method in ("abm", "abm-gwn")
        self.kwargs = kwargs

        ## initialization
        basis, intermidiate = self._initialize(X, method, **kwargs)
        
        for t in range(1, max_degree+1):
            # print("\ndegree %d" % t)
            cands = self._init_candidates(X, method, intermidiate) if t == 1 else self._candidates(intermidiate_1, intermidiate_t, method, X)
            # print('border', [c.as_expr() for c in cands.Fsymb])
            basist, intermidiate_t = self._construct_basis_t(cands, intermidiate, eps, method, gamma=gamma)
            
            basis.append(basist)
            intermidiate.extend(intermidiate_t)
            if t == 1:
                # basis1 = deepcopy(basist)
                intermidiate_1 = deepcopy(intermidiate_t)

            if basist.isemptyF(): 
                break 
        
        self.basis = Basis(basis)
        return self

    def _initialize(self, X, method, **kwargs):
        if method == "grad":
            return _initialize_grad(X, **kwargs)
        elif method == "vca": 
            return _initialize_vca(X, **kwargs)
        elif method == 'abm':
            return _initialize_abm(X, **kwargs)
        elif method == 'abm-gwn':
            return _initialize_abm_gwn(X, **kwargs)
        else:
            print("unknown method: %s", method)
        
    def _init_candidates(self, X, method, intermidiate):
        if method == "grad":
            return _init_candidates_grad(X)
        elif method == "vca": 
            return _init_candidates_vca(X)
        elif method == "abm": 
            return _init_candidates_abm(X, gens=intermidiate.gens, term_order=intermidiate.term_order)
        elif method == "abm-gwn": 
            return _init_candidates_abm_gwn(X, gens=intermidiate.gens, term_order=intermidiate.term_order)
        else:
            print("unknown method: %s", method)

    def _candidates(self, intermidiate_1, intermidiate_t, method, X):
        if method == "grad":
            return _candidates_grad(intermidiate_1, intermidiate_t)
        elif method == "vca": 
            return _candidates_vca(intermidiate_1, intermidiate_t)
        elif method == "abm": 
            return _candidates_abm(intermidiate_1, intermidiate_t)
        elif method == "abm-gwn": 
            return _candidates_abm_gwn(intermidiate_1, intermidiate_t, X)    
        else:
            print("unknown method: %s", method)
    
    def _construct_basis_t(self, cands, intermidiate, eps, method, gamma=1e-9):
        if method == "grad":
            return _construct_basis_t_grad(cands, intermidiate, eps, gamma=gamma)
        elif method == "vca":
            return _construct_basis_t_vca(cands, intermidiate, eps)
        elif method == "abm":
            return _construct_basis_t_abm(cands, intermidiate, eps)
        elif method == "abm-gwn":
            return _construct_basis_t_abm_gwn(cands, intermidiate, eps)
        else:
            print("unknown method: %s", method)
    
    # @lru_cache()
    # @memoize
    def evaluate(self, X, target='vanishing'):
        if self.symbolic: 
            return s_evaluate(self.basis, X, target=target)
        else:
            return n_evaluate(self.basis, X, target=target)
    
    # @lru_cache()
    # @memoize
    def gradient(self, X, target='vanishing'):
        if self.symbolic: 
            return s_gradient(self.basis, X, target=target)
        else:
            return n_gradient(self.basis, X, target=target)

    # @memoize
    def plot(self, X, target='vanishing', 
            n=1000, scale=1.5, x_max=1.0, y_max=1.0,
            z_func=lambda x_, y_: 0.0,
            show=False, splitshow=False):

        plot(self, X, target=target, 
            n=n, scale=scale, x_max=x_max, y_max=y_max,
            z_func=z_func,
            show=show, splitshow=splitshow)

    def set_weight(self, F=None, G=None, start=0):
        if F != None:
            assert(start + len(F) == len(self.basis))
            for Bt, Ft in zip(self.basis[start:], F):
                Bt.F = Ft
                
        if G != None:
            assert(start + len(G) == len(self.basis))
            for Bt, Gt in zip(self.basis[start:], G):
                Bt.G = Gt