statrl
======

.. rst-class:: lead

   The **Statistical Reinforcement Learning Toolkit** — a research library for
   bandit and reinforcement-learning algorithms, organised as a taxonomy of
   *settings*, each with a matching environment, agent, and interaction loop.

``statrl`` gives you a small, explicit protocol shared across settings: an
**environment** holding the problem, an **agent** that acts and learns, and an
**interaction loop** that runs the two against each other and returns a
cumulative-score time series. On top of that sit reference algorithms (IMED,
PSRL, IMED-RL, BIMED, BABA, an adversarial Lipschitz forecaster) and an
``experiments`` harness for running many replicates in parallel and plotting
regret.

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: quickstart
      :link-type: doc

      Install ``statrl`` and measure your first regret curve.

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Narrative walkthrough of each setting and the algorithms it ships.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Auto-generated reference for every public module, class, and function.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: auto_examples
      :link-type: doc

      Runnable scripts benchmarking agents and plotting regret.

.. toctree::
   :hidden:
   :caption: Getting started

   quickstart

.. toctree::
   :hidden:
   :caption: User guide

   user_guide/index
   auto_examples

.. toctree::
   :hidden:
   :caption: Reference

   api/index
   references

.. toctree::
   :hidden:
   :caption: Project

   developing
   changelog
