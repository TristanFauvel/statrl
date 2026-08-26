# Thompson Sampling (TS)

## Overview

**Thompson Sampling (TS)** is one of the most influential algorithms for stochastic multi-armed bandits. Introduced by William R. Thompson in 1933, it is a Bayesian algorithm that naturally balances exploration and exploitation through posterior sampling.

For Bernoulli rewards, Thompson Sampling assumes that the unknown mean reward of each arm follows a Beta prior. After each observation, the posterior distribution is updated analytically, and one random sample is drawn from every posterior. The arm whose sampled value is largest is selected.

Because the amount of uncertainty decreases automatically as more observations are collected, Thompson Sampling provides an elegant and parameter-free exploration strategy.

---

# Mathematical Formulation

Assume that each arm has an unknown Bernoulli mean

\[
\mu_a \in [0,1].
\]

A uniform prior is assigned:

\[
\mu_a \sim \mathrm{Beta}(1,1).
\]

After observing

- \(S_a\) successes,
- \(F_a\) failures,

the posterior becomes

\[
\mu_a
\mid
\mathcal D
\sim
\mathrm{Beta}(S_a+1,\;F_a+1).
\]

At each iteration, the algorithm independently samples

\[
\theta_a
\sim
\mathrm{Beta}(S_a+1,F_a+1)
\]

for every arm, and selects

\[
A_t
=
\arg\max_a \theta_a.
\]

---

# Algorithm

For every interaction:

1. Draw one posterior sample from each arm.
2. Select the arm with the largest sampled value.
3. Observe the reward.
4. Update the Beta posterior.

Unlike UCB, Thompson Sampling does not explicitly compute confidence intervals. Exploration naturally emerges from posterior uncertainty.

---

# Class Reference

## `TS(nbArms)`

Constructs a Thompson Sampling learner.

### Parameters

| Parameter | Description |
|------------|-------------|
| `nbArms` | Number of bandit arms |

---

# Attributes

| Attribute | Description |
|------------|-------------|
| `nbDraws` | Number of pulls for each arm |
| `cumRewards` | Number of observed successes |
| `theta` | Posterior samples used for arm selection |

---

# Methods

## `reset()`

Resets all posterior statistics.

Initially every arm follows

\[
\mathrm{Beta}(1,1),
\]

corresponding to a uniform prior.

---

## `play()`

Returns the arm with the largest sampled posterior value.

---

## `update(arm, reward)`

Updates the posterior distribution after observing a Bernoulli reward.

The posterior parameters become

- success parameter:

\[
\alpha = S + 1
\]

- failure parameter:

\[
\beta = F + 1.
\]

A new posterior sample is then generated for every arm.

---

# Computational Complexity

Let \(K\) denote the number of arms.

| Operation | Complexity |
|-----------|------------|
| Arm selection | \(O(K)\) |
| Posterior update | \(O(K)\) |
| Memory | \(O(K)\) |

---

# Theoretical Properties

Thompson Sampling enjoys strong theoretical guarantees for stochastic bandits.

Under standard assumptions:

- Bayesian regret is nearly optimal.
- Frequentist regret is logarithmic.
- The algorithm is asymptotically optimal for Bernoulli bandits.

Unlike deterministic index policies such as UCB, Thompson Sampling uses randomized action selection derived directly from posterior uncertainty.

---

# References

- Thompson, W. R. (1933). *On the likelihood that one unknown probability exceeds another in view of the evidence of two samples.*
- Kaufmann, Cappé & Garivier (2012). *On Bayesian Upper Confidence Bounds for Bandit Problems.*
- Russo, Van Roy, Kazerouni, Osband & Wen (2018). *A Tutorial on Thompson Sampling.*