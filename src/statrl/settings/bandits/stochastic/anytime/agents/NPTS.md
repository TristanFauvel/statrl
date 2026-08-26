# NPTS: Non-Parametric Thompson Sampling (Bounded Dirichlet Sampling)

## Overview

**Non-Parametric Thompson Sampling (NPTS)**, also known as **Bounded Dirichlet Sampling (BDS)**, is a stochastic multi-armed bandit algorithm that extends the idea of Thompson Sampling to a fully **distribution-free setting**.

Instead of maintaining a parametric posterior (e.g., Beta distributions for Bernoulli rewards), NPTS constructs **random empirical distributions** directly from observed data using Dirichlet-weighted resampling.

---

## Core Idea

Given a history of rewards for arm \( a \):

\[
\mathcal{D}_a = \{r_1, r_2, \dots, r_n\}
\]

NPTS generates a random estimate of the mean:

\[
\tilde{\mu}_a = \sum_{i=1}^{n} w_i r_i
\quad \text{where } w \sim \mathrm{Dirichlet}(1, \dots, 1)
\]

This produces a **random convex combination of past observations**, which acts as a non-parametric posterior sample.

---

## Algorithm Intuition

NPTS replaces Bayesian conjugate updates with:
- empirical history storage
- randomized resampling of observed rewards
- implicit uncertainty via Dirichlet weights

This yields:
- no parametric assumptions
- natural variance-driven exploration
- robustness to misspecified reward distributions

---

## Class Reference

### `NPTS(nbArms, bound=1.0)`

#### Parameters
- `nbArms` (int): number of arms
- `bound` (float): initial pseudo-reward for stabilization

---

## Attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `nbDraws` | `np.ndarray` | Number of pulls per arm |
| `cumRewards` | `np.ndarray` | Cumulative reward per arm |
| `meanRewards` | `np.ndarray` | Empirical mean reward per arm |
| `rewardHistory` | `list[list]` | Observed rewards per arm |
| `playbuffer` | `list` | Exploration scheduling buffer |
| `bound` | `float` | Initial reward seed |

---

## Methods

### `reset()`
Initializes all internal structures.

Each arm starts with a single artificial reward equal to `bound` to ensure numerical stability.

---

### `play()`

Selects the next arm using a **buffered exploration mechanism**.

#### Procedure

1. If `playbuffer` is non-empty:
   - return next scheduled arm

2. Otherwise:
   - identify leader arm (most pulled)
   - compare against less-sampled arms
   - compute Dirichlet-resampled means
   - selectively repopulate buffer

---

### `_dirichletmean(rewards)`

Computes a stochastic estimate of the mean reward:

\[
\tilde{\mu} = w^\top r, \quad w \sim \mathrm{Dirichlet}(1)
\]

This induces randomness over empirical support without parametric assumptions.

---

### `update(arm, reward)`

Updates empirical statistics.

#### Steps
1. Increment pull count
2. Update cumulative reward
3. Update empirical mean
4. Append reward to history

---

## Theoretical Interpretation

NPTS can be viewed as:
- a **bootstrap approximation of posterior sampling**
- a **Bayesian non-parametric method**
- a stochastic convex-hull estimator over empirical rewards

Unlike classical Thompson Sampling, it does not assume:
- Bernoulli rewards
- Gaussian rewards
- conjugate priors

---

## Computational Complexity

Let:
- \( n_a \): number of observations per arm

Then:
- Sampling step: \( O(n_a) \)
- Update step: \( O(1) \)
- Memory: \( O(\sum_a n_a) \)

---

## Remarks

- The Dirichlet sampling introduces additional variance for exploration
- The `playbuffer` mechanism stabilizes decision ordering
- The initial `bound` prevents degenerate early behavior
- Performance depends on reward history richness

---