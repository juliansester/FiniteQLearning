# Code for "Q-Learning under Finite Model Uncertainty"

##  Julian Sester, Cécile Decker

# Abstract
We propose a robust Q-learning algorithm for Markov decision processes under model uncertainty when each state–action pair is associated with a finite ambiguity set of candidate transition kernels. This finite-measure framework enables highly flexible, user-designed uncertainty models and goes beyond the common KL/Wasserstein-ball formulations. We establish almost sure convergence of the learned Q-function to the robust optimum, and derive non-asymptotic high-probability error bounds that separate stochastic approximation error from transition-kernel estimation error. Finally, we show that Wasserstein-ball and parametric ambiguity sets can be approximated by finite ambiguity sets, allowing our algorithm to be used as a generic solver beyond the finite setting.

# Preprint

[Link](https://arxiv.org/abs/2407.04259)

# Content

The Examples from the paper are provided as seperate jupyter notebooks, each with a unique name, exactly specifying which example is covered therein. These are:
- An [Example](https://github.com/juliansester/FiniteQLearning/blob/main/Example_cointoss.ipynb) covering finite Q learning for a coin toss game.
- An [Example(https://github.com/juliansester/FiniteQLearning/blob/main/Example_stockinvesting.ipynb) covering finite Q learning for a stock investing example.
