'''
Building blocks to manipulate terms in a numerical fashion. 
'''

from calendar import c
import numpy as np 
from scipy.special import comb
import itertools as itr 

def n_terms_d(n, d):
    return comb(n+d, d)

def n_terms_ud(n, d):
    return comb(n+d+1, d)

class Term():
    def __init__(self, exponents):
        self.exponents = np.array(exponents)
        self.total_degree = sum(exponents)

    def evaluate(self, X):
        ev = np.ones((X.shape[0], 1))
        for i in range(X.shape[1]): 
            ev *= X[:, i:i+1] ** self.exponents[i]
        
        return ev 

    def __repr__(self) -> str:
        return f'{self.exponents}'

    def mult(self, t): 
        return Term(self.exponents + t.exponents)


class Monom(Term):
    def __init__(self, exponents, c=1.0): 
        super().__init__(exponents)
        self.c = c

    def evaluate(self, X):
        return super().evaluate(X) * self.c

    def __repr__(self) -> str:
        return f'{self.c} * {self.exponents}'

    def mult(self, m): 
        return Monom(self.exponents + m.exponents, c=self.c * m.c)

class Poly(): 
    def __init__(self, monoms, homo=False):
        if type(monoms) is dict: 
            monoms = [Monom(expo, c=c) for expo, c in monoms.items()]
        if not (isinstance(monoms[0], Monom) or isinstance(monoms[0], Term)): 
            monoms = [Monom(m) for m in monoms]  # from exponents
        self.monoms = monoms 

    def evaluate(self, X): 
        return np.sum([ m.evaluate(X) for m in self.monoms ])

    def __repr__(self) -> str:
        return f'{[m.__repr__() for m in self.monoms]}'

    def mult(self, p): 
        pp = {}
        for m1, m2 in itr.product(self.monoms, p.monoms): 
            m = m1.mult(m2)

            if tuple(m.exponents) in pp: 
                pp[tuple(m.exponents)] += m.c 
            else: 
                pp[tuple(m.exponents)] = m.c 

        return Poly( pp )
    

if __name__ == '__main__': 
    e1 = [0, 1, 1]
    e2 = [1, 1, 1]
    t = Term(e1) 
    m = Monom(e2, 2.0)

    # X = np.random.randn(100, 3)
    X = np.array([1, 2, 1]).reshape(-1, 1)
    p = Poly([t, m])
    # print(p.evaluate(X))
    print(m)
    print(p)

    p = Poly([e1, e2])
    print('poly p:', p)
    print('p*p   :', p.mult(p))


    
        