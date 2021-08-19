# MAVI: Monomial Agnostic Vanishing Ideal

## Overview
MAVI provides numerical, GPU-backed implementation of approximate basis computation of vanishing ideals. 

MAVI includes the following methods: 
- Numerical methods: Vanishing Component Analysis (VCA; [Livni+, '13]) and its variants Simple Basis Computaition (SBC; [Kera+, '19, '20, '21a]). 
- Symbolic methods: Approximate Buchberger–M"oller algorithm (ABM; [Limbeck+, '13]) and its variant [Kera, '21b]. 

## Installation
```
$git clone https://github.com/HiroshiKERA/monomial-agnostic-vanishing-ideal.git
```

## Usage
The simplest usage is as follows:
```
from mavi.vanishing_ideal import VanishingIdeal

# X: data matrix, eps: threshold (positive real)
vi = VanishingIdeal()
vi.fit(X, eps, method='vca', backend='numpy')
vi.plot(X, target='vanishing', spiltshow=True)
```
Please refer to the Jupyter notebooks in `example/`. 

## Backend
To meet the various usages, MAVI is implemented in three different backends `numpy`, `jax`, and `pytorch`. Users are recommended to use `jax` because it provides the fastest computation backed by GPU and JIT compilation. `pytorch` can also work with GPU but symbolic methods are not implemented yet. If you use `jax` backend only, you can uncomment two `@partial` decorators at `evaluate()` and `gradient()` for further accelaration. 

## Citation
If you find this useful, please cite:
```
@misc{kera2021mavi,
   author = {Hiroshi Kera},
   title = {{MAVI: Monomial Agnostic Vanishing Ideal}},
   url = {https://github.com/HiroshiKERA/monomial-agnostic-vanishing-ideal},
}

@article{kera2021monomial,
  title={Monomial-agnostic computation of vanishing ideals},
  author={Hiroshi Kera and Yoshihiko Hasegawa},
  journal={arXiv preprint arXiv:2101.00243},
  year={2021}
}

```
## References 
1. [Livni+, '13]: Livni, R., Lehavi, D., Schein, S., Nachliely, H., Shalev-Shwartz, S., Globerson, A., 2013. Vanishing component analysis. In:  Proceedings of theThirteenth International Conference on Machine Learning (ICML). PMLR, pp. 597–605.
2. [Limbeck+, '13]: Limbeck, J., 2013. Computation of approximate border bases and applications. Ph.D. thesis, Passau, Universit ̈at Passau.
3. [Kera+, '19]: Kera, H., Hasegawa, Y., 2019. Spurious vanishing problem in approximate vanishing ideal. IEEE Access 7, pp. 178961–178976.
4. [Kera+, '20]: Kera, H., Hasegawa, Y., 2020. Gradient boosts the approximate vanishing ideal. In: Proceedings of the Thirty-Fourth AAAI Conference on ArtificialIntelligence (AAAI). AAAI Press, pp. 4428–4425.
5. [Kera+, '21a]: Kera, H., Hasegawa, Y., 2021. Monomial-agnostic computation of vanishing ideals. arXiv preprint arXiv:2101.00243
6. [Kera, '21b]: Kera, H., 2021. Approximate border basis with gradient normalization. arXiv preprint 	arXiv:2101.00401
