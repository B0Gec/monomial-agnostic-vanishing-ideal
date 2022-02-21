from turtle import back
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC, SVC
from mavi.vanishing_ideal import VanishingIdeal
from mavi.util.preprocessing import Preprocessor
import time 
import torch 

class VanishingComponentSVM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        ''

    def fit(self, X, y, th, method='vca', z=1.0, max_degree=10, preprocessing=False, backend='numpy', linear_energy_based_threshold=False):
        '''
        th corresponds to the 'epsilon' of the basis computation. 
        if linear_energy_based_threshold=True, 
        th is instead used to determine epsilons based on PCA. 
        Note that in this case, at least one linear polynomial will be classified vanishing. 
        Thus, linear_energy_based_threshold=False provides more general classifier.
        ''' 

        self.z = z
        if backend == 'torch':
            class_ids = np.sort(np.unique(y.detach().cpu().numpy()))
        else:
            class_ids = np.sort(np.unique(y))

        vis = [VanishingIdeal() for _ in class_ids]
        preps = [Preprocessor(backend=backend) for _ in class_ids]
        
        Zs = []
        for prep in preps: 
            for c in class_ids:
                prep.fit(X[y==c], th, quiet=True)
        
        epsilons = None 
        if np.isscalar(th): 
            th = [th for _ in class_ids]
            epsilons = th
        if linear_energy_based_threshold: 
            # epsilons = [min(*prep.eps_range) + 1e-9 for prep in preps]
            epsilons = [(prep.eps_range[0]+prep.eps_range[-1])*0.5 for prep in preps]
            # epsilons = [(prep.eps_range[0]*prep.eps_range[-1])**0.5 for prep in preps]

        start = time.time()
        if preprocessing:
            for c, eps, vi, prep in zip(class_ids, epsilons, vis, preps):
                vi.fit(prep.transform(X[y==c])/z, eps/z, method=method, max_degree=max_degree, backend=backend)
        else: 
            for c, eps, vi in zip(class_ids, epsilons, vis):
                vi.fit(X[y==c]/z, eps/z, method=method, max_degree=max_degree, backend=backend)
        end = time.time()

        # print(f'vis: {end - start} [sec]')

        self.vis = vis
        self.epsilons = epsilons
        self.th = th
        self.z = z
        self.backend = backend
        self.preprocessors = preps 
        self.preprocessing = preprocessing

        start = time.time()
        svc = LinearSVC()
        # svc = SVC(**{'kernel':'poly', 'degree':1,'coef0':0,'decision_function_shape':'ovr'})
        F = self.features(X)
        if self.backend == 'torch': 
            F = F.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

        # print(np.any(np.isnan(F)), np.max(np.abs(F)))
        # print(f'{method}: {[len(vi.basis) for vi in vis]}')

        svc.fit(F, y)
        end = time.time()

        # print(f'svc: {end - start} [sec]')


        self.svc = svc
        
        self.stat()

        return self

    def predict(self,X):
        F = self.features(X)
        if self.backend == 'torch': 
            F = F.detach().cpu().numpy()
        return self.svc.predict(F)
    
    def score(self, X, y):
        F = self.features(X)
        if self.backend == 'torch': 
            F = F.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        return self.svc.score(F, y)

    def features(self, X, deg=-1):
        ks = []
        for vi in self.vis:
            k = [Gt.n_bases() for Gt in vi.basis.vanishings()]
            k = k[:deg+1] if deg > 0 else 100000
            ks.append(k)

        F = None 
        if self.preprocessing: 
            Ls = [ X @ prep.Vg / self.z for prep in self.preprocessors ]
            Fs = [ vi.evaluate(prep.transform(X)/self.z)[:, :k] for k, vi, prep in zip(ks, self.vis, self.preprocessors) ]
            if self.backend == 'torch': 
                F = torch.hstack([ torch.hstack([L, F]) for L, F in zip(Ls, Fs) ]).abs()
            else: 
                F = np.abs(np.hstack([ np.hstack([L, F]) for L, F in zip(Ls, Fs) ]))
        else:
            if self.backend == 'torch': 
                F = torch.hstack( [ vi.evaluate(X/self.z)[:, :k] for k, vi in zip(ks, self.vis) ] ).abs()
            else:
                F = np.abs(np.hstack( [ vi.evaluate(X/self.z)[:, :k] for k, vi in zip(ks, self.vis) ] ))

        return F 
        
    def stat(self):

        ndegs = [np.array([G.n_bases() for G in vi.basis.vanishings()]) for vi in self.vis]
        degs = [np.array([i for i, G in enumerate(vi.basis.vanishings()) if G.n_bases() > 0]) for vi in self.vis]

        mean_deg = np.mean([np.sum([i*d for i,d in enumerate(ndeg)])/np.sum(ndeg) for ndeg in ndegs])
        max_deg = np.max(np.hstack(degs))
        min_deg = np.min(np.hstack(degs))
        ## classwise feat dim.
        cwdim = [np.sum([G.n_bases() for G in vi.basis.vanishings()]) for vi in self.vis]
        dim = np.sum(cwdim)

        self.stats = {'dim': dim, 'nfeat': list(cwdim), 'meandeg': mean_deg, 'maxdeg':max_deg, 'mindeg':min_deg, 'ndegrees': list(ndegs)}
