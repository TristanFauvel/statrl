Utilities
=========

Numerical helpers shared by every setting.

Tie-breaking selection
----------------------

.. currentmodule:: statrl.settings

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.randmax
   utils.randmin
   utils.allmax

Divergences
-----------

Parametric divergences for the usual one-parameter exponential families, plus
the non-parametric :math:`K_{\inf}` used by the batched IMED variants. Pick the
one matching your rewards — an index algorithm is asymptotically optimal only
for the right divergence.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.klBern
   utils.klGauss
   utils.klPoisson
   utils.klExp
   utils.KLinf_threshold

Sampling
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.categorical_sample
   utils.Dirac

Setting validation
------------------

Checks that a setting folder follows the layout described in
:doc:`../developing`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   validator.validate_setting
   validator.validate_structure
   validator.validate_environment
   validator.validate_agent
   validator.validate_interaction
   validator.validate_environment_yaml
   validator.load_module
   validator.report
