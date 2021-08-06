import jax.numpy as np

class Intermidiate():
    '''
    Structure  : Matrix (ndarray)
    Implication: Intermidiate results of basis construction process
    '''
    def __init__(self, FX):
        self.FX = FX
    def extend(self, interm):
        self.FX = np.hstack((self.FX, interm.FX))

class NBasist():
    '''
    Structure  : Pair of matrices (ndarrays)
    Implication: bases of vanishing and nonvanishing polynomials of degree t
    '''
    def __init__(self, G, F):
        self.G = G  # vanishing functions
        self.F = F  # non-vanishing functions

    def isemptyF(self):
        return self.F.isempty()
    
    def n_nonvanishings(self):
        return self.F.n_bases()
    
    def n_vanishings(self):
        return self.G.n_bases()

class Nbasist_fn():
    '''
    Structure  : Pair of matrices (ndarrays)
    Implication: bases of vanishing and nonvanishing polynomials of degree t
    '''
    def __init__(self, V, L=None):
        self.V = V
        self.L = L  

    def eval(self, Z, C=None):
        # F0 and G0
        if self.L is None:  
            assert(C is None)
            return Z * self.V 

        assert(not (Z is None))
        assert(not (C is None))
        return (C - Z @ self.L) @ self.V

    def isempty(self):
        return self.V.size == 0

    def n_bases(self):
        return self.V.shape[1]

    def matrix_form(self):
        return np.vstack([-self.L, np.eye(self.L.shape[1])]) @ self.V

