import math
import sympy as sp
import numpy as np


def round_expr(expr, num_digits):  # author: https://stackoverflow.com/a/48491897
    # return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})
    return expr.xreplace({n : round(n, num_digits) if abs(n) > 10**(-num_digits) else round(n, -min(0, math.floor(math.log10(abs(n)))))
                          for n in expr.atoms(sp.Number)})


def simpl_disp(expr, verbosity=0, num_digits=3, epsilon=1e-10):
    smds = sp.Add.make_args(expr)
    if verbosity > 0:
        print('smds:', smds)
    # display(smds)
    # li[0] < 1e-10
    li = [sp.Mul.make_args(smd)[0] for smd in smds]
    if verbosity > 0:
        # print(f'li:{li:< 10}')
        print(f'li:          {li}')

    # display(li)
    # li = [(abs(smd) > 1e-10) for smd in li]
    li = [smds[i] for i, smd in enumerate(li) if (abs(smd) > epsilon)]
    if verbosity > 0:
        print('li filtered:', li)
    # li = [smd for smd in li ]
    # li = [f'{smd:e}' for smd in li]
    expr = sum([round_expr(smd, num_digits=num_digits) for smd in li])
    # if verbosity > 0:
    #     print('li:', li)
    eq = sp.Eq(expr, 0)
    if verbosity > 0:
        print('eq', eq)
        print('expr', expr)
    # display(li)
    # display(expr)
    # display(eq)
    return eq, expr

# display(simpl_disp(g)[0])
#%%
