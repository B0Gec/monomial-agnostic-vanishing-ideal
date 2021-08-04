import torch

class Intermidiate():
    '''
    Structure  : Matrix (ndarray)
    Implication: Intermidiate results of basis construction process
    '''
    def __init__(self, FX):
        self.FX = FX
    def extend(self, interm):
        self.FX = torch.hstack((self.FX, interm.FX))

class NBasist():
    '''
    Structure  : Pair of matrices (ndarrays)
    Implication: bases of vanishing and nonvanishing polynomials of degree t
    '''
    def __init__(self, G, F):
        self.G = G  # vanishing functions
        self.F = F  # non-vanishing functions

    def isemptyF(self):
        return self.F.numel() == 0

