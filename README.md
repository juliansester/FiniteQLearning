# Code for "Q-Learning under Finite Model Uncertainty"

##  Julian Sester, Cécile Decker

# Abstract
We propose a robust Q-learning algorithm for Markov decision processes under model uncertainty when each state–action pair is associated with a finite ambiguity set of candidate transition kernels. This finite-measure framework enables highly flexible, user-designed uncertainty models and goes beyond the common KL/Wasserstein-ball formulations. We establish almost sure convergence of the learned Q-function to the robust optimum, and derive non-asymptotic high-probability error bounds that separate stochastic approximation error from transition-kernel estimation error. Finally, we show that Wasserstein-ball and parametric ambiguity sets can be approximated by finite ambiguity sets, allowing our algorithm to be used as a generic solver beyond the finite setting.

# Preprint

[Link](https://arxiv.org/abs/2407.04259)

# Content

The examples from the paper are provided as separate Jupyter notebooks, each with a unique name that specifies the example covered:
- An [example](https://github.com/juliansester/FiniteQLearning/blob/main/Example_cointoss.ipynb) covering finite Q-learning for a coin toss game.
- An [example](https://github.com/juliansester/FiniteQLearning/blob/main/Example_stockinvesting.ipynb) covering finite Q-learning for a stock investing example.

Core Python implementations included in this folder:
- `finite_q_learning.py`: robust Q-learning under finite ambiguity sets.
- `q_learning.py`: baseline Q-learning implementation used in the examples.
- `wasserstein_q_learning.py`: Wasserstein-ball robust Q-learning variant.

# Quick start

1. Open the notebooks in VS Code or Jupyter and run all cells.
2. Or run the scripts directly from this folder:
	 - `python finite_q_learning.py`
	 - `python q_learning.py`
	 - `python wasserstein_q_learning.py`

# Citation

If you use this code, please cite the preprint:

```bibtex
@article{SesterDecker2026FiniteQ,
	title   = {Q-Learning under Finite Model Uncertainty},
	author  = {Sester, Julian and Decker, C\'ecile},
	journal = {arXiv preprint arXiv:2407.04259},
	year    = {2026}
}
```
