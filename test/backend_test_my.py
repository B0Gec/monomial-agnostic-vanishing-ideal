from mavi.vanishing_ideal import VanishingIdeal

def main():
    test_numpy()
    # jax_test()
    # torch_test()

def test_numpy():
    import numpy as np
    X = np.random.randn(20, 2)

    vi = VanishingIdeal()
    vi.fit(X, 0.1, method='vca', backend='numpy')
    eval_test(vi, X, do=('eval', 'grad'))

    vi.plot(X, target='vanishing', show=False)
    vi.plot(X, target='nonvanishing', show=False)

    vi = VanishingIdeal()
    vi.fit(X, 0.1, method='grad', backend='numpy')
    eval_test(vi, X, do=('eval', 'grad'))

    vi = VanishingIdeal()
    vi.fit(X, 0.1, method='abm', backend='numpy')
    eval_test(vi, X, do=('eval'))

    vi = VanishingIdeal()
    vi.fit(X, 0.1, method='abm_gwn', backend='numpy')
    eval_test(vi, X, do=('eval'))

    print('numpy backend test done')

# def test_jax():
#     import jax.numpy as jnp
#
#     key = jnp.random.PRNGKey(1)
#     X = jnp.random.normal(key, (20, 2))
#
#     vi = VanishingIdeal()
#     vi.fit(X, 0.1, method='vca', backend='jax')
#     eval_test(vi, X, do=('eval', 'grad'))
#
#     vi.plot(X, target='vanishing', show=False)
#     vi.plot(X, target='nonvanishing', show=False)
#
#     vi = VanishingIdeal()
#     vi.fit(X, 0.1, method='grad', backend='jax')
#     eval_test(vi, X, do=('eval', 'grad'))
#
#     print('jax backend test done')


# def test_torch():
#     import torch
#
#     X = torch.randn(20, 2)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     X = X.to(X)
#
#     vi = VanishingIdeal()
#     vi.fit(X, 0.1, method='vca', backend='torch')
#     eval_test(vi, X, do=('eval', 'grad'))
#
#     vi.plot(X, target='vanishing', show=False)
#     vi.plot(X, target='nonvanishing', show=False)
#
#     vi = VanishingIdeal()
#     vi.fit(X, 0.1, method='grad', backend='torch')
#     eval_test(vi, X, do=('eval', 'grad'))
#
#     print('pytorch backend test done')


def eval_test(vi, X, do=('eval', 'grad')):
    if 'eval' in do: 
        vi.evaluate(X, target='vanishing')
        vi.evaluate(X, target='nonvanishing')
    if 'grad' in do: 
        vi.gradient(X, target='vanishing')
        vi.gradient(X, target='nonvanishing')


if __name__ == '__main__':
    main()