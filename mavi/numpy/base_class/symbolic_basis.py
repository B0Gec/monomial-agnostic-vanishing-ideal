import numpy as np
from collections import UserList

class Intermidiate():
    '''
    Structure  : List of polynomials (sympy objects)
    Implication: Intermidiate results of basis construction process
    '''
    def __init__(self, FX, Fsymb, gens):
        self.FX = FX
        self.Fsymb = Fsymb
        self.gens = gens 

    def extend(self, interm):
        self.FX = np.hstack((self.FX, interm.FX))
        self.Fsymb.extend(interm.Fsymb)

class SBasist():
    '''
    Structure  : Pair of lists of polynomials (sympy objects)
    Implication: bases of vanishing and nonvanishing polynomials of degree t
    '''
    def __init__(self, G, F):
        self.G = G  # vanishing functions
        self.F = F  # non-vanishing functions

    def isemptyF(self):
        return len(self.F) == 0

    def n_vanishings(self):
        return len(self.G)

class SBasis(UserList):
    '''
    Structure  : List of SBasis_t
    Implication: bases of vanishing and nonvanishing polynomials
    '''
    def __init__(self, basist_arr=[], term_order='grevlex'):
        super().__init__(basist_arr)
        self.term_order = term_order

    def nonvanishings(self):
        return [Bt.F for Bt in self.data]
    
    def vanishings(self):
        return [Bt.G for Bt in self.data]
