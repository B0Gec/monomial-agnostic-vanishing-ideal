# coding: utf-8
from mavi.base_class.basis import Basis
import itertools as itr
from copy import deepcopy
# from jax import jit, partial
import torch.nn as nn 
import torch 

class VanishingIdeal():
    def __init__(self):
        super().__init__()
        self.basis  = []
        self.eps    = None 
        self.method = None 

    def fit(self, X, eps, method="grad", max_degree=12, gamma=1e-6, with_coeff=False, backend='numpy', **kwargs):
        self.load_modules(method, backend)

        if backend=='torch': 
            self.to(X.device)
            torch.no_grad()
            
        ## set attributes
        self.eps = eps
        self.method = method
        self.max_degree = max_degree
        self.gamma = gamma  

        # NOTE: smaller gamma (e.g., 1e-9) also works for numpy backend
        #       but not for torch because pytorch uses float (not double)
        self.symbolic = method in ("abm", "abm-gwn")
        self.kwargs = kwargs

        ## initialization
        basis, intermidiate = self.initialize(X, **self.kwargs)
        
        for t in range(1, self.max_degree+1):
            # print("\ndegree %d" % t)
            cands = self.init_candidates(X, **self.kwargs) if t == 1 else self.candidates(intermidiate_1, intermidiate_t, degree=t)
            # print('border', [c.as_expr() for c in cands.Fsymb])
            basist, intermidiate_t = self.construct_basis_t(cands, intermidiate, eps, gamma=self.gamma)
            
            basis.append(basist)
            intermidiate.extend(intermidiate_t)
            if t == 1:
                # basis1 = deepcopy(basist)
                intermidiate_1 = deepcopy(intermidiate_t)

            if basist.isemptyF(): 
                break 
        
        self.basis = Basis(basis)

        return self

    # Uncomment if you use jax backend only
    # @partial(jit, static_argnums=(0,2,))
    def evaluate(self, X, target='vanishing'):
        return self._evaluate(self.basis, X, target=target)
    
    # Uncomment if you use jax backend only
    # @partial(jit, static_argnums=(0,2,))
    def gradient(self, X, target='vanishing', keep_dim=False):
        '''
        Not implemented for symbolic case. Use ```symbolic_evalutation.gradient``` instead.
        '''
        return self._gradient(self.basis, X, target=target, keep_dim=keep_dim)

    def load_modules(self, method, backend):
        self.method = method 
        self.backend = backend

        if backend == 'numpy':
            from mavi.numpy.util.plot import plot
        if backend == 'jax':
            from mavi.jax.util.plot import plot
        if backend == 'torch':
            from mavi.torch.util.plot import plot

        if method == "grad":
            if backend == 'numpy':
                from mavi.numpy.basis_construction.grad import Basist, Intermidiate
                from mavi.numpy.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.grad import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

            if backend == 'jax':
                from mavi.jax.basis_construction.grad import Basist, Intermidiate
                from mavi.jax.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.jax.basis_construction.grad import construct_basis_t
                from mavi.jax.evaluation.numerical_evaluation import evaluate
                from mavi.jax.evaluation.numerical_evaluation import gradient

            if backend == 'torch':
                from mavi.torch.basis_construction.grad import Basist, Intermidiate
                from mavi.torch.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.torch.basis_construction.grad import construct_basis_t
                from mavi.torch.evaluation.numerical_evaluation import evaluate
                from mavi.torch.evaluation.numerical_evaluation import gradient

        elif method == "grad-c":
            if backend == 'numpy':
                from mavi.numpy.basis_construction.grad_c import Basist, Intermidiate
                from mavi.numpy.basis_construction.grad_c import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.grad_c import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

        elif method == "coeff":
            if backend == 'numpy':
                from mavi.numpy.basis_construction.coeff import Basist, Intermidiate
                from mavi.numpy.basis_construction.coeff import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.coeff import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

            if backend == 'jax':
                from mavi.jax.basis_construction.grad import Basist, Intermidiate
                from mavi.jax.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.jax.basis_construction.grad import construct_basis_t
                from mavi.jax.evaluation.numerical_evaluation import evaluate
                from mavi.jax.evaluation.numerical_evaluation import gradient

            if backend == 'torch':
                from mavi.torch.basis_construction.grad import Basist, Intermidiate
                from mavi.torch.basis_construction.grad import initialize, init_candidates, candidates
                from mavi.torch.basis_construction.grad import construct_basis_t
                from mavi.torch.evaluation.numerical_evaluation import evaluate
                from mavi.torch.evaluation.numerical_evaluation import gradient

        elif method == "vca": 
            if backend == 'numpy':
                from mavi.numpy.basis_construction.vca import Basist, Intermidiate
                from mavi.numpy.basis_construction.vca import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.vca import construct_basis_t
                from mavi.numpy.evaluation.numerical_evaluation import evaluate
                from mavi.numpy.evaluation.numerical_evaluation import gradient

            if backend == 'jax':
                from mavi.jax.basis_construction.vca import Basist, Intermidiate
                from mavi.jax.basis_construction.vca import initialize, init_candidates, candidates
                from mavi.jax.basis_construction.vca import construct_basis_t
                from mavi.jax.evaluation.numerical_evaluation import evaluate
                from mavi.jax.evaluation.numerical_evaluation import gradient

            if backend == 'torch':
                from mavi.torch.basis_construction.vca import Basist, Intermidiate
                from mavi.torch.basis_construction.vca import initialize, init_candidates, candidates
                from mavi.torch.basis_construction.vca import construct_basis_t
                from mavi.torch.evaluation.numerical_evaluation import evaluate
                from mavi.torch.evaluation.numerical_evaluation import gradient

        elif method == 'abm':
            if backend == 'numpy':
                from mavi.numpy.basis_construction.abm import Basist, Intermidiate
                from mavi.numpy.basis_construction.abm import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.abm import construct_basis_t
                from mavi.numpy.evaluation.symbolic_evaluation import evaluate
                from mavi.numpy.evaluation.symbolic_evaluation import gradient

            if backend == 'torch':
                ''

        elif method in ('abm-gwn', 'abm_gwn'):
            if backend == 'numpy':
                from mavi.numpy.basis_construction.abm_gwn import Basist, Intermidiate
                from mavi.numpy.basis_construction.abm_gwn import initialize, init_candidates, candidates
                from mavi.numpy.basis_construction.abm_gwn import construct_basis_t
                from mavi.numpy.evaluation.symbolic_evaluation import evaluate
                from mavi.numpy.evaluation.symbolic_evaluation import gradient

            if backend == 'jax':  # sympy objects are not compatible with jax array
                ''
                # from mavi.jax.basis_construction.abm_gwn import Basist, Intermidiate
                # from mavi.jax.basis_construction.abm_gwn import initialize, init_candidates, candidates
                # from mavi.jax.basis_construction.abm_gwn import construct_basis_t
                # from mavi.jax.evaluation.symbolic_evaluation import evaluate
                # from mavi.jax.evaluation.symbolic_evaluation import gradient

            if backend == 'torch':
                ''
        else:
            print(f"unknown method: {method}")


        self.initialize = initialize
        self.init_candidates = init_candidates
        self.candidates = candidates
        self.construct_basis_t = construct_basis_t

        self._evaluate = evaluate
        self._gradient = gradient

        self._plot = plot

        if backend == 'jax':
            self._evaluate_jit = None 

    def plot(self, X, target='vanishing', 
            n=1000, scale=1.5, x_max=1.0, y_max=1.0,
            z_func=lambda x_, y_: 0.0,
            show=False, splitshow=False):

        self._plot(self, X, target=target, 
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

    def to(self, device):
        assert(self.backend == 'torch')
        if self.basis: self.basis.to(device)
        self.device = device



def main(X):
    vi = VanishingIdeal()
    vi.fit(X.detach().numpy(), 10.0, method="vca", backend='numpy', max_degree=2) 

if __name__ == '__main__':
    import torch
    from torchvision.datasets import MNIST
    from torchvision import transforms 
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = MNIST('.', train=False, download=True, transform=transform)
    dataloader = DataLoader(mnist_test, batch_size=100)

    dataiter = iter(dataloader)
    images, labels = dataiter.next()  # ミニバッチを一つ取り出す
    X = images.view(-1, 28*28)

    main(X)
    print('done!')