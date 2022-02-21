# from jax import jit 
class Preprocessor():
    def __init__(self, backend='numpy'):
        self.backend = backend

    def fit(self, X, th=.95, keep_dim=False, quiet=False):
        self.fit_transform(X, th=th, keep_dim=keep_dim, quiet=quiet)

    def fit_transform(self, X, th=.95, keep_dim=False, quiet=False):
        '''
        retain th % (defalut 95%) power
        '''

        if self.backend in ('numpy', 'jax'):
            return self._fit_transform_numpy(X, th, keep_dim, quiet=quiet)
        if self.backend in ('torch'):
            return self._fit_transform_torch(X, th, keep_dim, quiet=quiet)

    def _fit_transform_numpy(self, X, th=.95, keep_dim=False, quiet=False):
        if self.backend == 'numpy': import numpy as np
        if self.backend == 'jax': import jax.numpy as np

        m = np.mean(X, axis=0)
        X_ = X - m 
        _, d, Vt = np.linalg.svd(X_)
        th_id = np.where(np.cumsum(d**2) / np.sum(d**2) >= th**2)[0]
        th_id = th_id[0] if len(th_id) > 0 else -1

        # print(d)
        # print(th_id)

        self.d = d 
        self.th = th
        self.th_id = th_id
        self.eps_range = (d[th_id] if th_id < len(d) else 0, d[th_id-1])
        self.eps = (self.eps_range[0] + self.eps_range[1]) * 0.5
        self.m = m 
        self.V = Vt[:th_id].T if th_id > 0 else Vt.T
        self.Vg = Vt[th_id:].T if th_id > 0 else Vt[:, 0:0].T 
        self.keep_dim = keep_dim

        if not quiet:
            print('----------------------')
            print(f'top {th_id-1} components has {th*100}% power in total')
            print(f'correpsonding range of epsilon is {self.eps_range} (mean: {self.eps})')
            print('----------------------')

        return self.transform(X, keep_dim=keep_dim)

    def _fit_transform_torch(self, X, th=.95, keep_dim=False, quiet=False):
        import torch

        m = X.mean(dim=0)
        X_ = X - m 
        _, d, Vt = torch.linalg.svd(X_)

        th_id = 1 + torch.where((d**2).cumsum(dim=0) / (d**2).sum() >= th**2)[0]
        th_id = th_id[0] if len(th_id) > 0 else -1

        self.d = d 
        self.th = th
        self.th_id = th_id
        self.eps_range = (d[th_id] if th_id < len(d) else 0, d[th_id-1])
        self.eps = (self.eps_range[0] + self.eps_range[1]) * 0.5
        self.m = m 
        self.V = Vt[:th_id].T if th_id > 0 else Vt.T
        self.Vg = Vt[th_id:].T if th_id > 0 else Vt[:, 0:0].T 
        self.keep_dim = keep_dim

        if not quiet:
            print('----------------------')
            print(f'top {th_id-1} components has {th*100}% power in total')
            print(f'correpsonding range of epsilon is {self.eps_range} (mean: {self.eps})')
            print('----------------------')
        
        return self.transform(X, keep_dim=keep_dim)

    def transform(self, X, keep_dim=False):
        if keep_dim: 
            return (X - self.m) @ self.V @ self.V.T
        else:
            return (X - self.m) @ self.V 


