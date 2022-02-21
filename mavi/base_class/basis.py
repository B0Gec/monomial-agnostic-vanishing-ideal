from collections import UserList
from itertools import chain

class Basis(UserList):
    '''
    Structure  : List of NBasis_t
    Implication: bases of vanishing and nonvanishing polynomials
    '''
    def __init__(self, basist_arr=[]):
        super().__init__(basist_arr)

    def nonvanishings(self, concat=False):
        ret = [Bt.F for Bt in self.data] 
        return list(chain(*ret)) if concat else ret
    
    def vanishings(self, concat=False):
        ret = [Bt.G for Bt in self.data] 
        return list(chain(*ret)) if concat else ret

    def to(self, device):
        for Bt in self.data:
            Bt.F = Bt.F.to(device)
            Bt.G = Bt.G.to(device)

    def cuda(self, device):
        for Bt in self.data:
            Bt.F = Bt.F.cuda()
            Bt.G = Bt.G.cuda()

    def detach(self, device):
        for Bt in self.data:
            Bt.F = Bt.F.detach()
            Bt.G = Bt.G.detach()

    def cpu(self, device):
        for Bt in self.data:
            Bt.F = Bt.F.cpu()
            Bt.G = Bt.G.cpu()
    