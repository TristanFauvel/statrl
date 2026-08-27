# UCB: Upper Confidence Bound Algorithm

## Overview

The **Upper Confidence Bound (UCB)** algorithm is a foundational method in stochastic multi-armed bandits. It is based on the principle of **optimism in the face of uncertainty**, selecting actions according to the highest statistically plausible reward estimate.

Each arm is assigned an index composed of:
- empirical mean reward
- an exploration bonus derived from concentration inequalities

---

## Core Principle

For each arm \( a \), the UCB index for [0,1] bounded or 1/2-subGaussian distributions is defined as:

\[
I_a(t) = \hat{\mu}_a(t) + \sqrt{\frac{\log(1/\delta(t))}{2 N_a(t)}}
\]

where:
- \( \hat{\mu}_a(t) \): empirical mean reward
- \( N_a(t) \): number of pulls of arm \( a \)
- \( \delta(t) \): confidence schedule
- \( t \): global time step

The selected action is:

\[
a_t = \arg\max_a I_a(t)
\]

---

## Algorithm Intuition

UCB balances:
- exploitation via empirical mean
- exploration via uncertainty bounds

The exploration bonus shrinks as:
- the number of pulls increases
- confidence level tightens over time

This ensures:
- logarithmic regret in stationary stochastic settings
- automatic exploration decay

---

## Class Reference

### `UCB(nbArms, delta)`

#### Parameters
- `nbArms` (int): number of arms
- `delta` (callable): confidence schedule function δ(t)

---

## Attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `time` | int | Global time step |
| `nbDraws` | `np.ndarray` | Number of pulls per arm |
| `cumRewards` | `np.ndarray` | Cumulative rewards per arm |
| `means` | `np.ndarray` | Empirical mean reward per arm |
| `indexes` | `np.ndarray` | UCB index per arm |
| `delta` | callable | Confidence schedule |

---

## Methods

### `reset()`
Reinitializes all statistics before a new experiment.

---

### `play()`
Selects the arm with the highest UCB index:

\[
\arg\max_a I_a(t)
\]

---

### `update(arm, reward)`

Updates internal statistics.

#### Steps
1. Increment time step
2. Update cumulative reward
3. Update empirical mean
4. Recompute confidence bounds
5. Update UCB indices

---

## Theoretical Guarantees

Under standard assumptions (bounded rewards, stochastic stationarity), UCB achieves:

- **logarithmic regret**:
  \[
  R(T) = O(\log T)
  \]

- asymptotic optimality up to constant factors

---

## Computational Complexity

Let \( K \) be the number of arms:

- Update step: \( O(K) \)
- Memory: \( O(K) \)

---

## Remarks

- Performance depends strongly on the choice of \( \delta(t) \)
- The algorithm is sensitive to scaling of rewards
- Initialization ensures all arms are eventually explored

---