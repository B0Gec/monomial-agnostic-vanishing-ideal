import numpy as np
import matplotlib.pyplot as plt

from mavi.vanishing_ideal import VanishingIdeal

# X = [np.ndarray([x, x*x] for x in range(10)]
X = [[x, x*x*x] for x in range(-10, 10)]
X = [[x, x*x] for x in range(-100, 100)]
# X = [[x,x] for x in range(10)]
Xn = np.array(X)
# print(X)
# print(type(Xn))
# print(Xn)
# plt.plot(Xn[:, 0], Xn[:, 1], 'o')
# plt.show()

# vi.plot(X, target="vanishing", splitshow=True)
vi = VanishingIdeal()
method = "grad"
# methods = ["grad", "vca", "abm", "abm-gwn"]
methods = ["vca", "abm", "abm_gwn", "abm-gwn", "coeff", "grad", "grad-c"]
# vi.fit(Xn, 0.01, method="vca", backend='numpy')
# vi.fit(Xn, 0.01, method="vca", backend='numpy', max_degree=3)
for method in methods:
    print('\n' * 3, 'each method\n')
    # print(vi.basis.n_vanishings())
    # print(vi.basis)
    print('method:', method)
    vi.fit(Xn, 0.01, method=method, backend='numpy', max_degree=3)
    print('\nprint each:', method)
    print([elt.G for elt in vi.basis])

# vi.fit(Xn, 0.01, method="abm", backend='numpy', max_degree=3)
# vi.plot(Xn, target="vanishing", splitshow=True)
# print(vi.basis[0].Fsymb)
print('\n'*5, 'after plot\n')
# print(vi.basis.n_vanishings())
# print(vi.basis)
print('method:', method)
# self.symbolic = method in ("abm", "abm-gwn")
# elif method in ('abm-gwn', 'abm_gwn'):
# grad-c, coeff
print([elt.G for elt in vi.basis])
# for i in vi.basis:
#     # print(i.basis)
#     print(i.G)
#     print('fail, i.e. vanishing:', i.F)
# print(vi.basis[0].vanishings())
# print(vi.basis[0])

# go in vanishing_ideal.py and print basis

1/0


# print(vi.gamma)
# print(2)
print(vi.basis)
# for i in vi.basis:
#     # # print(i.attributes)
#     print(vi.basis[0])
#     print(vi.basis[0].isemptyF())
#     # print(vi.basis[0].n_vanishings())
#     # print(vi.basis[0].n_nonvanishings())

#     print(vi.basis[0].F)
#     print(vi.basis[0].G)
#     print(vi.basis[0].F.__dir__())
#     # print(vi.basis[0].__dir__())
#     # print(vi.basis[0].F.V)
#     print('after')
#     # print(vi.intermidiate)
# 
print(202)
vi.evaluate(Xn, target='vanishing')
print(3)


