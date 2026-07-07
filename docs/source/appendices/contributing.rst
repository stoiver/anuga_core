.. _contributing:

Contributing
============

Contributions to ANUGA are welcome — bug reports, documentation, and code. The
full guide (forking, branching, and opening a pull request) is in the
repository's `CONTRIBUTING.rst
<https://github.com/anuga-community/anuga_core/blob/main/CONTRIBUTING.rst>`_.

In brief:

#. Fork and clone the
   `repository <https://github.com/anuga-community/anuga_core>`_, and set up a
   development install (see :doc:`../installation/install_anuga_developers`).

#. Create a feature branch and make your change. Add or update tests under the
   relevant ``anuga/*/tests/`` directory, and keep the change focused.

#. Run the test suite before submitting:

   .. code-block:: bash

      pytest --pyargs anuga --run-fast     # quick check (~40 s)
      pytest --pyargs anuga                # full suite (~1600 tests)

   and lint the files you touched:

   .. code-block:: bash

      ruff check anuga/path/to/module.py

#. Push to your fork and open a pull request against
   ``anuga-community/anuga_core``, describing what the change does and why.

Bug reports and feature requests can be raised on the
`issue tracker <https://github.com/anuga-community/anuga_core/issues>`_.
