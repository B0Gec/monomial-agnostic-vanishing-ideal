# coding: utf-8

import itertools as itr
from copy import deepcopy
from mavi.numpy.util.plot import plot
from mavi.numpy.base_class.basis import Basis

class VanishingIdeal():
    def __init__(self):
        self.basis  = []
        self.eps    = None 
        self.method = None 
        
    def fit(self, X, eps, method="grad", max_degree=15, gamma=1e-9, backend='numpy', **kwargs):

        self.load_modules(method, backend)

        ## set attributes
        self.eps = eps
        self.method = method
        self.max_degree = max_degree
        self.gamma = gamma
        self.symbolic = method in ("abm", "abm-gwn")
        self.kwargs = kwargs

        ## initialization
        basis, intermidiate = self.initialize(X, **kwargs)
        
        for t in range(1, max_degree+1):
            # print("\ndegree %d" % t)
            cands = self.init_candidates(X, **kwargs) if t == 1 else self.candidates(intermidiate_1, intermidiate_t)
            # print('border', [c.as_expr() for c in cands.Fsymb])
            basist, intermidiate_t = self.construct_basis_t(cands, intermidiate, eps, gamma=gamma)
            
            basis.append(basist)
            intermidiate.extend(intermidiate_t)
            if t == 1:
                # basis1 = deepcopy(basist)
                intermidiate_1 = deepcopy(intermidiate_t)

            if basist.isemptyF(): 
                break 
        
        self.basis = Basis(basis)
        return self

    def evaluate(self, X, target='vanishing'):
        return self._evaluate(self.basis, X, target=target)
    
    def gradient(self, X, target='vanishing'):
        '''
        Not implemented for symbolic case. Use ```symbolic_evalutation.gradient``` instead.
        '''
        return self._gradient(self.basis, X, target=target)

    def load_modules(self, method, backend):
        self.method = method 
        self.backend = backend

        if method == "grad":
            if backend == 'numpy':
                from mavi.numpy.basis_construction.grad import Basist, Intermidiate
                from mavi.numpy.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.grad import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

            if backend == 'pytorch':
                from mavi.pytorch.basis_construction.grad import Basist, Intermidiate
                from mavi.pytorch.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.pytorch.basis_construction.grad import construct_basis_t
                from mavi.pytorch.evaluation.numerical_evaluation import evaluate
                from mavi.pytorch.evaluation.numerical_evaluation import gradient


        elif method == "vca": 
            if backend == 'numpy':
                from mavi.numpy.basis_construction.vca import Basist, Intermidiate
                from mavi.numpy.basis_construction.vca import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.vca import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

            if backend == 'pytorch':
                from mavi.pytorch.basis_construction.vca import Basist, Intermidiate
                from mavi.pytorch.basis_construction.vca import initialize, init_candidates, candidates
                from mavi.pytorch.basis_construction.vca import construct_basis_t
                from mavi.pytorch.evaluation.numerical_evaluation import evaluate
                from mavi.pytorch.evaluation.numerical_evaluation import gradient

        elif method == 'abm':
            if backend == 'numpy':
                from mavi.numpy.basis_construction.abm import Basist, Intermidiate
                from mavi.numpy.basis_construction.abm import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.abm import construct_basis_t
                from mavi.numpy.evaluation.symbolic_evaluation import evaluate
                from mavi.numpy.evaluation.symbolic_evaluation import gradient

            if backend == 'pytorch':
                ''

        elif method == 'abm_gwn':
            if backend == 'numpy':
                from mavi.numpy.basis_construction.abm_gwn import Basist, Intermidiate
                from mavi.numpy.basis_construction.abm_gwn import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.abm_gwn import construct_basis_t
                from mavi.numpy.evaluation.symbolic_evaluation import evaluate
                from mavi.numpy.evaluation.symbolic_evaluation import gradient

            if backend == 'pytorch':
                ''
        else:
            print("unknown method: %s", method)


        self.initialize = initialize
        self.init_candidates = init_candidates
        self.candidates = candidates
        self.construct_basis_t = construct_basis_t

        self._evaluate = evaluate
        self._gradient = gradient


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