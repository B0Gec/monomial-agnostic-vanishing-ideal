import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

from torch.nn.functional import relu

# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
from copy import deepcopy

from torch_util import NBasis, blow, dblow
from torch_bcgrad import Basist_grad, Intermidiate_grad
from torch_bcgrad import _initialize_grad, _init_candidates_grad, _candidates_grad
from torch_bcgrad import _construct_basis_t_grad

from torch_vca import Basist_vca, Intermidiate_vca
from torch_vca import _initialize_vca, _init_candidates_vca, _candidates_vca
from torch_vca import _construct_basis_t_vca


# class ExactApproximateVI():
class VanishingIdeal():
    def __init__(self):
        ''
    def fit(self, X, eps, method="grad", max_degree=20):
        ## set attributes
        self.X = (X)
        self.eps = eps
        self.method = method
        self.max_degree = max_degree

        ## initialization
        basis, intermidiate = self._initialize(X, method)
        
        for t in range(1, max_degree+1):
            # print("degree %d" % t)
            cands = self._init_candidates(X, method) if t == 1 else self._candidates(intermidiate_1, intermidiate_t, method)

            basist, intermidiate_t = self._construct_basis_t(cands, intermidiate, eps, method)
            
            basis.append(basist)
            intermidiate.extend(intermidiate_t)
            if t == 1:
                # basis1 = deepcopy(basist)
                intermidiate_1 = deepcopy(intermidiate_t)

            if basist.isemptyF(): 
                break 
        
        self.basis = NBasis(basis)
        return self


    def _initialize(self, X, method):
        if method == "grad":
            return _initialize_grad(X)
        elif method == "vca": 
            return _initialize_vca(X)
        else:
            print("unknown method: %s", method)
        
    def _init_candidates(self, X, method):
        if method == "grad":
            return _init_candidates_grad(X)
        elif method == "vca": 
            return _init_candidates_vca(X)
        else:
            print("unknown method: %s", method)

    def _candidates(self, intermidiate_1, intermidiate_t, method):
        if method == "grad":
            return _candidates_grad(intermidiate_1, intermidiate_t)
        elif method == "vca": 
            return _candidates_vca(intermidiate_1, intermidiate_t)
        else:
            print("unknown method: %s", method)
    
    def _construct_basis_t(self, cands, intermidiate, eps, method):
        if method == "grad":
            return _construct_basis_t_grad(cands, intermidiate, eps)
        elif method == "vca":
            return _construct_basis_t_vca(cands, intermidiate, eps)
        else:
            print("unknown method: %s", method)
    
    def evaluate(self, X, target='vanishing'):
        if target == 'nonvanishing':
            return self._evaluate_nv(self.basis, X)
        elif target == 'vanishing':
            return self._evaluate_v(self.basis, X)
        else:
            print('unknown mode: %s' % target)
            exit()
        
    def _evaluate_nv(self, B, X):
        F = B.nonvanishings()
        
        N = X.shape[0]

        Z0 = np.ones((N, 1)) * F[0]
        if len(F) == 1: return Z0
        
        Z1 = np.hstack([Z0, X]) @ F[1]
        Z = np.hstack([Z0, Z1])

        Zt = np.array(Z1)

        for t in range(2, len(F)):
            C = blow(Z1, Zt)
            Zt = np.hstack([Z, C]) @ F[t]
            Z = np.hstack([Z, Zt])

        return Z


    def _evaluate_v(self, B, X):
        F = B.nonvanishings()
        G = B.vanishings()
        # if np.all([gt.size==0 for gt in G]):
            # return np.array([])
        if np.all([gt.size==0 for gt in G]):
            return np.zeros((len(X), 0))

        N = X.shape[0]

        ZF0 = np.ones((N, 1)) * F[0]
        ZF1 = np.hstack([ZF0, X]) @ F[1]
        Z1 = np.hstack([ZF0, X]) @ G[1]

        ZF = np.hstack([ZF0, ZF1])
        Z = np.array(Z1)
        # print([f.shape for f in F[1:]])
        ZFt = np.array(ZF1)
        for t in range(2, len(F)):
            C = blow(ZF1, ZFt)
            Zt = np.hstack([ZF, C]) @ (G[t])
            ZFt = np.hstack([ZF, C]) @ (F[t])
            ZF = np.hstack([ZF, ZFt])
            Z = np.hstack([Z, Zt])

        return Z


    def gradient(self, X, target='vanishing'):
        if target == 'nonvanishing':
            return self._gradient_nv(self.basis, X)
        elif target == 'vanishing':
            return self._gradient_v(self.basis, X)        
        else:
            print('unknown mode %s' % target)

    def _gradient_nv(self, B, X):
        F = B.nonvanishings()
        npoints, ndims = X.shape

        Z0 = np.ones((npoints, 1)) * F[0]
        dZ0 = np.zeros((npoints*ndims, 1))
        if len(F) == 1: 
            return dZ0
        
        Z1 = np.hstack((Z0, X)) @ F[1]
        dZ1 = np.repeat(F[1][1:, :], npoints, axis=0)
        Z, dZ =  np.hstack((Z0, Z1)), np.hstack((dZ0, dZ1))
        Zt, dZt = deepcopy(Z1), deepcopy(dZ1)

        for t in range(2,len(F)):
            C, dC = dblow(Z1, Zt, dZ1, dZt)
            Zt = np.hstack((Z, C)) @ F[t]
            dZt  = np.hstack((dZ, dC)) @ F[t]
            Z, dZ = np.hstack((Z, Zt)), np.hstack((dZ, dZt))

        return dZ

    def _gradient_v(self, B, X):
        F = B.nonvanishings()
        G = B.vanishings()
        npoints, ndims = X.shape

        ZF0 = np.ones((npoints, 1)) * F[0]
        dZF0 = np.zeros((npoints*ndims, 1))
        if len(F) == 1: 
            return dZF0
        
        ZF1 = np.hstack((ZF0, X)) @ F[1]
        dZF1 = np.repeat(F[1][1:, :], npoints, axis=0)
        ZF, dZF =  np.hstack((ZF0, ZF1)), np.hstack((dZF0, dZF1))
        ZFt, dZFt = deepcopy(ZF1), deepcopy(dZF1)

        dZ = np.repeat(G[1][1:,:], npoints, axis=0)

        for t in range(2,len(F)):
            C, dC = dblow(ZF1, ZFt, dZF1, dZFt)
            ZFt = np.hstack((ZF, C)) @ F[t]
            dZFt  = np.hstack((dZF, dC)) @ F[t]
            dZt  = np.hstack((dZF, dC)) @ G[t]
            ZF, dZF = np.hstack((ZF, ZFt)), np.hstack((dZF, dZFt))
            dZ = np.hstack((dZ, dZt))
        return dZ

    def plot(self, X, target='vanishing', n=1000, scale=1.5, x_max=1.0, y_max=1.0, show=False, splitshow=False):

        ## set plot range
        m = np.mean(X, axis=0)
        x_max = y_max = np.max(np.abs(X))
        x = np.arange(-scale*x_max, scale*x_max, 0.1)
        y = np.arange(-scale*y_max, scale*y_max, 0.1)
        Z1, Z2 = np.meshgrid(x, y)

        ## set plot setting
        npolys = 0
        if target == 'vanishing':
            npolys = sum([Gt.shape[1] for Gt in self.basis.vanishings()])
        elif target == 'nonvanishing':
            npolys = sum([Ft.shape[1] for Ft in self.basis.nonvanishings()])
        else:
            print('unknown target: %s' % target)

        colors = plt.cm.Dark2(np.linspace(0,1,8))
        linestyles = ['solid','dashed','dashdot', 'dotted']
        nfigs = min(npolys, n)

        for i in range(nfigs):
            f = lambda x_, y_: self.evaluate(np.array([[x_,y_]]), target=target)[0,i]
            f = np.vectorize(f)
            plt.contour(Z1,Z2,f(Z1, Z2), levels=[0], colors=[colors[i%len(colors)]], linewidths=[1.], linestyles=[linestyles[i%4]])
            if splitshow:
                plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()

        if not splitshow:
            plt.plot(X[:,0], X[:,1], 'o', mfc='none', alpha=0.8)
            plt.gca().set_aspect('equal', adjustable='box')      
#         plt.savefig('graph_Z.pdf') 
        
        if not splitshow and show: 
            plt.show()

    def set_weight(self, F=None, G=None, start=0):
        if F != None:
            assert(start + len(F) == len(self.basis))
            for Bt, Ft in zip(self.basis[start:], F):
                Bt.F = Ft
                
        if G != None:
            assert(start + len(G) == len(self.basis))
            for Bt, Gt in zip(self.basis[start:], G):
                Bt.G = Gt




class polylayer(torch.nn.Module):
    def __init__(self, F, G, bias=True):
        super(polylayer, self).__init__()

        self.linear_f = torch.nn.Linear(*np.shape(F), bias=bias)
        self.linear_g = torch.nn.Linear(*np.shape(G), bias=bias)
        with torch.no_grad():
            self.linear_f.weight.copy_(F.T)
            self.linear_g.weight.copy_(G.T)

    def forward(self, ot, o1, o):  # assume ot  = kt x 1
        x = [(o1[i].unsqueeze(1) @ ot[i].unsqueeze(0)).reshape(1,-1) for i in range(o1.size()[0])]  # C_pre
        x = torch.cat(x, dim=0)
        x = torch.cat([o, x], dim=1)
        xf = self.linear_f(x)
        xg = self.linear_g(x)

        return xf, xg
    
    def weight(self):
        return self.linear_f.weight, self.linear_g.weight

    def nparameters(self):
        return np.prod(self.linear_f), np.prod(self.linear_g)

class PolyNet(nn.Module):
    def __init__(self, D_in, D_out, Fs, Gs, bias=True, task="regression", target="full"):
        super(PolyNet, self).__init__()
        
        self.target = target
        
        self.layers = nn.ModuleList()

        self.bias0 = Fs[0].squeeze()
        self.linear_f = torch.nn.Linear(*np.shape(Fs[1]), bias=bias)
        self.linear_g = torch.nn.Linear(*np.shape(Gs[1]), bias=bias)
        self.task = task
        
        with torch.no_grad():
            self.linear_f.weight.copy_(Fs[1].T)
            self.linear_g.weight.copy_(Gs[1].T)
        
        dims_f = []
        dims_g = []
        for F, G in zip(Fs[1:], Gs[1:]):
            dims_f.append(F.shape[1])
            dims_g.append(G.shape[1])

        for F, G in zip(Fs[2:], Gs[2:]):
            self.layers.append(polylayer(F, G, bias=bias))
        
        if target == 'full':
            self.fc = nn.Linear(sum(dims_f)+sum(dims_g)+1, D_out, bias=bias)
        elif target == 'nonvanishing':
            self.fc_f = nn.Linear(sum(dims_f)+1, D_out, bias=bias)
        elif target == 'vanishing':
            self.fc_g = nn.Linear(sum(dims_g), D_out, bias=bias)            
        else:
            print("unknown keyword %s" % target)
            

    def forward(self, x):
        o = torch.ones(x.size()[0], 1, device=x.device) * self.bias0
        ot = x
        
        # degree 1
        # tmp = torch.cat((o, ot), dim=1)
        # og = self.linear_g(tmp)
        og = self.linear_g(torch.cat((o, ot), dim=1))
        ot = self.linear_f(torch.cat((o, ot), dim=1))
        
        o1 = ot.clone()
        o = torch.cat((o, ot), dim=1)
        
        # degree > 1
        for t, pl in enumerate(self.layers):
            ot, otg = pl(ot, o1, o)
            o = torch.cat((o, ot), dim=1)
            og = torch.cat((og, otg), dim=1)

        output = None

        target = self.target    
        if target == 'full':
            output = self.fc(torch.cat((o, og), dim=1))
        elif target == 'nonvanishing':
            output = self.fc_f(o)
        elif target == 'vanishing':
            output = self.fc_g(og)
        else:
            print("unknown keyword %s" % target)

        if self.task == "classification":
            output = relu(output)

        return output
    
    def weight(self, target=None, form='numpy'):
        target = self.target if target == None else target
        
        res = None
        if target == 'full':
            res = [(self.linear_f.weight, self.linear_g.weight)]
            res.extend([l.weight() for l in self.layers])
            res.extend([self.fc.weight])            
        elif target == 'nonvanishing':
            res = [self.linear_f.weight]
            res.extend([l.weight()[0] for l in self.layers])
            res.extend([self.fc_f.weight])
        elif target == 'vanishing':
            res = [self.linear_g.weight]
            res.extend([l.weight()[1] for l in self.layers])
            res.extend([self.fc_g.weight])
        else:
            print("unknown keyword %s" % target)

        if form == 'numpy':
            res = [r.to('cpu').detach().numpy().copy().T for r in res]
        
        return res

    # def nparameters(self):
    #     np.prod(self.linear_g)
    #     # sum([np.prod(p.size()) for p in l.parameters()])
    #     target = self.target
    #     if target == 'full':
    #         output = self.fc.parameters()
    #         np.prod(l.weight.size())
    #     elif target == 'nonvanishing':
    #         output = self.fc_f(o)
    #     elif target == 'vanishing':
    #         output = self.fc_g(og)
    #     else:
    #         print("unknown keyword %s" % target)



if __name__ == '__main__':
    import sys
    # sys.path.append("/Users/kera/Dropbox/RESEARCH/Experiments/javi/avipy/src")
    sys.path.append('/home3/kera/workspace/avipy/src/')
    from vanishing_ideal import VanishingIdeal

    ## Data preparation
    N = 64
    D_in, D_out = 10, 1
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    ## VI basis set computation
    X = x.to('cpu').detach().numpy().copy()
    B = VanishingIdeal()
    B.fit(X, 0.5)
    F = [torch.from_numpy(ft.astype(np.float64)).clone() for ft in B.basis.nonvanishings()]
    G = [torch.from_numpy(ft.astype(np.float64)).clone() for ft in B.basis.vanishings()]

    ## Training
    # Construct our model by instantiating the class defined above
    model = PolyNet(D_in, D_out, F, G, bias=False)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(300):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()