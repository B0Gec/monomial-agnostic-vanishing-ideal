output from error when running mavi!

does it uses least squares?!!!


---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
Cell In[36], line 5
      1 vi = VanishingIdeal()
      2 # vi.fit(X, 0.01, method="grad")
      3 # vi.fit(X, 0.01, method="grad", max_degree=3)
      4 # vi.fit(X, 0.01, method="grad", max_degree=2)
----> 5 vi.fit(data, 0.01, method="grad", max_degree=2)
      6 vi.basis[0]

File ~/monomial-agnostic-vanishing-ideal/example/../mavi/vanishing_ideal.py:41, in VanishingIdeal.fit(self, X, eps, method, max_degree, gamma, with_coeff, backend, **kwargs)
     39 cands = self.init_candidates(X, **self.kwargs) if t == 1 else self.candidates(intermidiate_1, intermidiate_t, degree=t)
     40 # print('border', [c.as_expr() for c in cands.Fsymb])
---> 41 basist, intermidiate_t = self.construct_basis_t(cands, intermidiate, eps, gamma=self.gamma)
     43 basis.append(basist)
     44 intermidiate.extend(intermidiate_t)

File ~/monomial-agnostic-vanishing-ideal/example/../mavi/numpy/basis_construction/grad.py:53, in construct_basis_t(cands, intermidiate, eps, gamma, z)
     50 CtX, dCtX = cands.FX, cands.dFX
     51 FX, dFX = intermidiate.FX, intermidiate.dFX
---> 53 CtX_, L = pres(CtX, FX)
     54 dCtX_ = res(dCtX, dFX, L)
     56 nsamples = CtX_.shape[0]

File ~/monomial-agnostic-vanishing-ideal/example/../mavi/numpy/util/util.py:59, in pres(C, F)
     58 def pres(C, F):
---> 59     L = np.linalg.lstsq(F, C, rcond=None)[0]
     60     res = C - F @ L     # by definition, res == torch.hstack([F, C]) @ L
     61     return res, L

File ~/Documents/py-envs/vii/lib/python3.11/site-packages/numpy/linalg/linalg.py:2326, in lstsq(a, b, rcond)
   2323 if n_rhs == 0:
   2324     # lapack can't handle n_rhs = 0 - so allocate the array one larger in that axis
   2325     b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)
-> 2326 x, resids, rank, s = gufunc(a, b, rcond, signature=signature, extobj=extobj)
   2327 if m == 0:
   2328     x[...] = 0

UFuncTypeError: Cannot cast ufunc 'lstsq_n' input 0 from dtype('O') to dtype('float64') with casting rule 'same_kind'
