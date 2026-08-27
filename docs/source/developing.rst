Contributing
============

Setting up
----------

.. code-block:: bash

   git clone https://github.com/StatisticalRL/statrl.git
   cd statrl
   pip install -e ".[test,lint]"
   pip install -r docs/requirements.txt

Checks
------

The same checks run in CI. Run them before opening a pull request:

.. code-block:: bash

   pytest                                              # unit tests
   ruff check src && mypy                              # lint and types
   sphinx-build -b html -W --keep-going docs/source docs/_build/html
   sphinx-build -b doctest docs/source docs/_build/doctest

Adding a setting
----------------

A setting is a folder under ``src/statrl/settings/`` with a fixed layout, which
:mod:`statrl.settings.validator` checks:

.. code-block:: text

   mysetting/
     agent.py            # base agent class for the setting
     environment.py      # environment class
     interaction.py      # Interaction subclass
     _test.py            # smoke script
     agents/             # concrete algorithms
     envs/               # concrete instances, plus environments.yaml
     renderers/          # optional display
     wrappers/           # optional adapters to other settings

.. code-block:: bash

   python -m statrl.settings.validator

Documentation
-------------

Docstrings are `numpydoc <https://numpydoc.readthedocs.io/>`_ style, rendered
through :mod:`sphinx.ext.napoleon`. Every public object must have a docstring
and be listed in ``docs/source/api/``.

Follow the conventions the existing docstrings use:

- Document constructor parameters on the **class**, not on ``__init__``.
- Give every algorithm a ``References`` section citing its paper, and add the
  entry to :doc:`references`.
- Use ``See Also`` to link variants and counterparts; it is what makes the
  reference navigable.
- Put runnable ``Examples`` in docstrings where the object can be exercised in
  a couple of lines. They run under ``pytest --doctest-modules`` in CI. 

When adding a setting, add both an ``api/<setting>.rst`` and a
``user_guide/<setting>.rst``, and list them in the two ``index.rst`` files. An
API page that no ``user_guide`` page explains is a reference nobody can enter.

