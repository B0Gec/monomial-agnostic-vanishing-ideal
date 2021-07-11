import numpy as np
from mavi.numpy.base_class.numerical_basis import NBasist as Basist
from mavi.numpy.base_class.numerical_basis import Intermidiate
from mavi.numpy.util.util import res, pres, matrixfact, blow

class Basist_vca(Basist):
    def __init__(self, G, F):
        super().__init__(G, F)

class Intermidiate_vca(Intermidiate):
    def __init__(self, FX):
        super().__init__(FX)

    def extend(self, interm):
        super().extend(interm)

def _initialize_vca(X):
    npoints, ndims = X.shape
    constant = 1./npoints**0.5

    F = [np.ones((1,1))*constant]
    G = [np.zeros((0,0))]

    FX = np.ones((npoints, 1)) * constant

    interm = Intermidiate_vca(FX)

    basis0 = Basist_vca(G[0], F[0])
    return [basis0], interm


def _init_candidates_vca(X):
    return Intermidiate_vca(X)


def _candidates_vca(int_1, int_t):
    return Intermidiate_vca(blow(int_1.FX, int_t.FX))


def _construct_basis_t_vca(cands, intermidiate, eps):
    CtX = cands.FX        # evaluation matrix of candidate polynomials
    FX = intermidiate.FX  # evlauation matrix of nonvanishing polynomials up to degree t-1

    CtX_, R = pres(CtX, FX)  # orthogonal projection

    d, V = matrixfact(CtX_)

    Ft = R @ V[:, d>eps]
    Gt = R @ V[:, d<=eps]
    FtX = CtX_ @ V[:, d>eps]
    scales = np.linalg.norm(FtX, axis=0)
    FtX /= scales
    Ft /= scales

    return Basist_vca(Gt, Ft), Intermidiate_vca(FtX)